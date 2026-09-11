"""Battery state-of-charge and internal-resistance estimation."""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


CAPACITY_AH = 2.4
POLARIZATION_RESISTANCE_OHM = 0.018
POLARIZATION_TIME_CONSTANT_S = 55.0


def open_circuit_voltage(state_of_charge):
    """Approximate a lithium-ion open-circuit-voltage curve in volts."""
    state_of_charge = np.clip(np.asarray(state_of_charge), -0.05, 1.05)
    return (
        3.15
        + 0.85 * state_of_charge
        + 0.10 * np.tanh(8.0 * (state_of_charge - 0.5))
        - 0.06 * np.exp(-25.0 * state_of_charge)
        + 0.05 * np.exp(-25.0 * (1.0 - state_of_charge))
    )


def open_circuit_voltage_derivative(state_of_charge):
    """Differentiate :func:`open_circuit_voltage` with respect to charge."""
    state_of_charge = np.clip(np.asarray(state_of_charge), -0.05, 1.05)
    hyperbolic_tangent = np.tanh(8.0 * (state_of_charge - 0.5))
    return (
        0.85
        + 0.8 * (1.0 - hyperbolic_tangent**2)
        + 1.5 * np.exp(-25.0 * state_of_charge)
        + 1.25 * np.exp(-25.0 * (1.0 - state_of_charge))
    )


def propagate_battery_state(state, delta_t_s):
    """Propagate ``[charge, polarization, log-resistance, current]``."""
    charge, polarization_voltage, log_resistance, current_a = np.asarray(state)
    decay = np.exp(-delta_t_s / POLARIZATION_TIME_CONSTANT_S)
    return np.array(
        [
            charge - delta_t_s * current_a / (3600.0 * CAPACITY_AH),
            decay * polarization_voltage
            + POLARIZATION_RESISTANCE_OHM * (1.0 - decay) * current_a,
            log_resistance,
            current_a,
        ]
    )


def battery_transition_jacobian(_state, delta_t_s):
    """Return the analytic Jacobian of the battery transition."""
    decay = np.exp(-delta_t_s / POLARIZATION_TIME_CONSTANT_S)
    jacobian = np.eye(4)
    jacobian[0, 3] = -delta_t_s / (3600.0 * CAPACITY_AH)
    jacobian[1, 1] = decay
    jacobian[1, 3] = POLARIZATION_RESISTANCE_OHM * (1.0 - decay)
    return jacobian


def observe_voltage_and_current(state):
    """Observe terminal voltage and load current from the battery state."""
    charge, polarization_voltage, log_resistance, current_a = np.asarray(state)
    internal_resistance = np.exp(log_resistance)
    terminal_voltage = (
        open_circuit_voltage(charge)
        - polarization_voltage
        - internal_resistance * current_a
    )
    return np.array([terminal_voltage, current_a])


def voltage_current_jacobian(state):
    """Return the terminal-voltage/current observation Jacobian."""
    charge, _polarization_voltage, log_resistance, current_a = np.asarray(state)
    internal_resistance = np.exp(log_resistance)
    return np.array(
        [
            [
                open_circuit_voltage_derivative(charge),
                -1.0,
                -internal_resistance * current_a,
                -internal_resistance,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def drive_cycle_current(times_s):
    """Return a repeatable discharge, pulse-load, and regeneration cycle."""
    times_s = np.asarray(times_s)
    phase_s = np.mod(times_s, 300.0)
    current_a = np.select(
        [
            phase_s < 90.0,
            phase_s < 140.0,
            phase_s < 200.0,
            phase_s < 240.0,
        ],
        [0.8, 2.8, 1.4, -0.4],
        default=0.2,
    )
    return current_a + 0.12 * np.sin(2.0 * np.pi * times_s / 47.0)


def simulate_battery_truth(times_s, initial_charge=0.92, resistance_ohm=0.062):
    """Simulate a battery under the deterministic current drive cycle."""
    times_s = np.asarray(times_s)
    current_a = drive_cycle_current(times_s)
    truth = np.empty((len(times_s), 4))
    truth[0] = np.array([initial_charge, 0.0, np.log(resistance_ohm), current_a[0]])
    for index, delta_t_s in enumerate(np.diff(times_s)):
        truth[index + 1] = propagate_battery_state(truth[index], delta_t_s)
        truth[index + 1, 3] = current_a[index + 1]
    return truth


def run_battery_state_of_charge(
    seed=801,
    duration_s=3600.0,
    delta_t_s=5.0,
    use_jacobian=False,
):
    """Estimate battery charge and resistance from voltage/current measurements."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s / delta_t_s) + 1) * delta_t_s
    true_resistance_ohm = 0.062
    truth = simulate_battery_truth(times, resistance_ohm=true_resistance_ohm)

    voltage_std_v = 0.012
    current_std_a = 0.04
    measurement_covariance = np.diag([voltage_std_v**2, current_std_a**2])
    noiseless_measurements = np.array(
        [observe_voltage_and_current(state) for state in truth[1:]]
    )
    measurements = noiseless_measurements + rng.normal(
        0.0,
        [voltage_std_v, current_std_a],
        noiseless_measurements.shape,
    )
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_voltage_and_current,
            voltage_current_jacobian,
        )
        for measurement in measurements
    ]

    model = StateTransitionModel(
        propagate_battery_state,
        np.diag([1.5e-8, 2.5e-6, 1.0e-5, 0.65**2]),
        battery_transition_jacobian,
    )
    initial_mean = truth[0] + np.array(
        [-0.13, 0.025, np.log(0.038 / true_resistance_ohm), 0.20]
    )
    initial_covariance = np.diag([0.14**2, 0.05**2, 0.55**2, 0.35**2])
    bayes_filter = BayesianFilter(model, Gaussian(initial_mean, initial_covariance))
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=use_jacobian,
    )

    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])

    def charge_rmse(states):
        return np.sqrt(np.mean((states[:, 0] - truth[:, 0]) ** 2))

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "true_resistance_ohm": true_resistance_ohm,
        "filtered_resistance_ohm": np.exp(filtered[:, 2]),
        "smoothed_resistance_ohm": np.exp(smoothed[:, 2]),
        "filtered_voltage_v": np.array(
            [observe_voltage_and_current(state)[0] for state in filtered]
        ),
        "smoothed_voltage_v": np.array(
            [observe_voltage_and_current(state)[0] for state in smoothed]
        ),
        "true_voltage_v": np.array(
            [observe_voltage_and_current(state)[0] for state in truth]
        ),
        "filtered_charge_rmse": charge_rmse(filtered),
        "smoothed_charge_rmse": charge_rmse(smoothed),
        "voltage_std_v": voltage_std_v,
        "current_std_a": current_std_a,
        "use_jacobian": use_jacobian,
    }


if __name__ == "__main__":
    from examples.battery_plotting import plot_battery_state_of_charge

    result = run_battery_state_of_charge()
    path = Path(__file__).parent / "figures" / "battery-state-of-charge.png"
    plot_battery_state_of_charge(result, path)
    print(
        f"charge RMSE: {100.0 * result['filtered_charge_rmse']:.2f} -> "
        f"{100.0 * result['smoothed_charge_rmse']:.2f} percentage points"
    )
    print(
        f"internal resistance: {1e3 * result['filtered_resistance_ohm'][0]:.1f} -> "
        f"{1e3 * result['filtered_resistance_ohm'][-1]:.1f} mΩ "
        f"(true {1e3 * result['true_resistance_ohm']:.1f} mΩ)"
    )
    print(f"saved {path}")

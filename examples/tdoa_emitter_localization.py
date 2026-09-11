"""Moving-emitter localization from time-difference-of-arrival measurements."""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS
from examples.linear_tracking import (
    constant_velocity_matrix,
    white_acceleration_covariance,
)


PROPAGATION_SPEED_KM_S = 299792.458
MICROSECONDS_PER_SECOND = 1e6
MICROSECONDS_PER_KM = MICROSECONDS_PER_SECOND / PROPAGATION_SPEED_KM_S
RECEIVER_POSITIONS_KM = np.array(
    [
        [0.0, -12.0],
        [-12.0, -5.0],
        [12.0, -6.0],
        [11.0, 11.0],
        [-10.0, 10.0],
    ]
)
TRUE_CLOCK_BIASES_US = np.array([0.35, -0.22, 0.15, -0.30])


def propagate_emitter_state(state, delta_t_s):
    """Propagate position/velocity while holding receiver clock biases fixed."""
    state = np.asarray(state)
    propagated = state.copy()
    propagated[:2] += delta_t_s * state[2:4]
    return propagated


def emitter_transition_jacobian(state, delta_t_s):
    """Return the augmented constant-velocity transition matrix."""
    jacobian = np.eye(len(state))
    jacobian[:4, :4] = constant_velocity_matrix(2, delta_t_s)
    return jacobian


def observe_tdoa(state, receiver_positions=RECEIVER_POSITIONS_KM):
    """Observe non-reference arrival times minus the reference arrival time."""
    receiver_positions = np.asarray(receiver_positions)
    ranges_km = np.linalg.norm(receiver_positions - np.asarray(state)[:2], axis=1)
    geometric_tdoa_us = MICROSECONDS_PER_KM * (ranges_km[1:] - ranges_km[0])
    return geometric_tdoa_us + np.asarray(state)[4:]


def tdoa_jacobian(state, receiver_positions=RECEIVER_POSITIONS_KM):
    """Return the TDOA Jacobian, including non-reference clock biases."""
    receiver_positions = np.asarray(receiver_positions)
    displacement = np.asarray(state)[:2] - receiver_positions
    directions = displacement / np.linalg.norm(displacement, axis=1)[:, None]
    num_differences = len(receiver_positions) - 1
    jacobian = np.zeros((num_differences, len(state)))
    jacobian[:, :2] = MICROSECONDS_PER_KM * (directions[1:] - directions[0])
    jacobian[:, 4:] = np.eye(num_differences)
    return jacobian


def tdoa_noise_covariance(arrival_time_std_us, num_receivers):
    """Covariance induced by differencing independent receiver arrival times."""
    num_differences = num_receivers - 1
    return arrival_time_std_us**2 * (
        np.eye(num_differences) + np.ones((num_differences, num_differences))
    )


def simulate_emitter_truth(initial_state, times):
    """Simulate a constant-velocity emitter with fixed receiver clock biases."""
    truth = [np.asarray(initial_state)]
    for delta_t_s in np.diff(times):
        truth.append(propagate_emitter_state(truth[-1], delta_t_s))
    return np.array(truth)


def run_tdoa_emitter_localization(
    seed=801,
    duration_s=180.0,
    delta_t_s=0.5,
    use_jacobian=False,
):
    """Track a moving emitter and infer four receiver clock offsets."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s / delta_t_s) + 1) * delta_t_s
    initial_truth = np.concatenate(
        (
            np.array([-9.0, -6.0, 0.095, 0.075]),
            TRUE_CLOCK_BIASES_US,
        )
    )
    truth = simulate_emitter_truth(initial_truth, times)

    arrival_time_std_us = 0.06
    measurement_covariance = tdoa_noise_covariance(
        arrival_time_std_us,
        len(RECEIVER_POSITIONS_KM),
    )
    measurements = []
    for state in truth[1:]:
        arrival_noise_us = rng.normal(
            0.0,
            arrival_time_std_us,
            len(RECEIVER_POSITIONS_KM),
        )
        differenced_noise_us = arrival_noise_us[1:] - arrival_noise_us[0]
        measurements.append(observe_tdoa(state) + differenced_noise_us)
    measurements = np.array(measurements)
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_tdoa,
            tdoa_jacobian,
        )
        for measurement in measurements
    ]

    process_covariance = np.zeros((8, 8))
    process_covariance[:4, :4] = white_acceleration_covariance(
        2,
        delta_t_s,
        spectral_density=1e-7,
    )
    process_covariance[4:, 4:] = 1e-6 * np.eye(4)
    model = StateTransitionModel(
        propagate_emitter_state,
        process_covariance,
        emitter_transition_jacobian,
    )
    initial_mean = initial_truth + np.array(
        [1.5, -1.1, 0.018, -0.012, -0.35, 0.22, -0.15, 0.30]
    )
    initial_covariance = np.diag(
        [2.0**2, 2.0**2, 0.04**2, 0.04**2, *([0.8**2] * 4)]
    )
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
    filtered_covariances = np.array(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.array(
        [state.covariance() for state in smoother_states]
    )

    def position_rmse(states):
        return np.sqrt(np.mean(np.sum((states[:, :2] - truth[:, :2]) ** 2, axis=1)))

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "receiver_positions": RECEIVER_POSITIONS_KM.copy(),
        "true_clock_biases_us": TRUE_CLOCK_BIASES_US.copy(),
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "arrival_time_std_us": arrival_time_std_us,
        "measurement_covariance": measurement_covariance,
        "use_jacobian": use_jacobian,
    }


if __name__ == "__main__":
    from examples.plotting import plot_tdoa_emitter_localization

    result = run_tdoa_emitter_localization()
    path = Path(__file__).parent / "figures" / "tdoa-emitter-localization.png"
    plot_tdoa_emitter_localization(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f} km"
    )
    print(
        "clock biases [μs]: "
        f"{np.array2string(result['filtered'][-1, 4:], precision=3)} "
        f"(true {np.array2string(result['true_clock_biases_us'], precision=3)})"
    )
    print(f"saved {path}")

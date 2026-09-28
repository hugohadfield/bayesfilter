"""Estimate aerodynamic drag and rolling resistance from a flat-road coast-down.

A vehicle starts at motorway speed and coasts on a perfectly flat road with no
engine, braking, or regenerative torque. The only measurement is noisy GPS
speed. Because rolling resistance contributes approximately constant
deceleration while aerodynamic drag scales with speed squared, a coast-down
that spans a wide speed range makes both effects observable.

The hidden state is ``[speed, rolling_resistance_coefficient, drag_area]``,
where ``drag_area`` is ``C_d A`` in square metres. The idealized dynamics are

``m dv/dt = -m g C_rr - 0.5 rho C_d A v |v|``.

This deliberately ignores road grade, wind, drivetrain drag, tyre-temperature
changes, and other practical complications so the example stays focused on
online physical-parameter estimation.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


SPEED = 0
ROLLING_RESISTANCE = 1
DRAG_AREA = 2

GRAVITY_MPS2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
VEHICLE_MASS_KG = 1700.0

TRUE_ROLLING_RESISTANCE = 0.012
TRUE_DRAG_AREA_M2 = 0.65
INITIAL_SPEED_MPS = 40.0
DURATION_S = 145.0

GPS_INTERVAL_S = 1.0
GPS_TIMING_JITTER = 0.15
GPS_SPEED_STD_MPS = 0.25


def coastdown_acceleration(state):
    """Return longitudinal acceleration from rolling and aerodynamic drag."""
    speed, rolling_resistance, drag_area = np.asarray(state)
    rolling_deceleration = GRAVITY_MPS2 * rolling_resistance
    aerodynamic_deceleration = (
        0.5
        * AIR_DENSITY_KG_M3
        * drag_area
        / VEHICLE_MASS_KG
        * speed
        * abs(speed)
    )
    return -(rolling_deceleration + aerodynamic_deceleration)


def propagate_coastdown_state(state, delta_t_s):
    """Propagate the coast-down dynamics with one RK4 step."""
    state = np.asarray(state, dtype=float)

    def derivative(current_state):
        return np.array([coastdown_acceleration(current_state), 0.0, 0.0])

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * delta_t_s * k1)
    k3 = derivative(state + 0.5 * delta_t_s * k2)
    k4 = derivative(state + delta_t_s * k3)
    return state + delta_t_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def process_noise_covariance(delta_t_s):
    """Small random walks for speed and the nominally constant parameters."""
    diffusion_std = np.array([0.015, 1e-6, 1e-5])
    return np.diag(diffusion_std**2 * max(delta_t_s, 1e-9))


def observe_gps_speed(state):
    """GPS observes speed only."""
    return np.array([np.asarray(state)[SPEED]])


def simulate_coastdown_truth(delta_t_s=0.05):
    """Generate the ideal flat-road coast-down used by the example."""
    times = np.arange(0.0, DURATION_S + delta_t_s, delta_t_s)
    truth = np.empty((len(times), 3))
    truth[0] = np.array(
        [
            INITIAL_SPEED_MPS,
            TRUE_ROLLING_RESISTANCE,
            TRUE_DRAG_AREA_M2,
        ]
    )
    for index in range(len(times) - 1):
        truth[index + 1] = propagate_coastdown_state(truth[index], delta_t_s)
    return times, truth


TRUTH_TIMES, TRUTH_STATES = simulate_coastdown_truth()


def truth_state_at_time(time_s):
    """Interpolate the dense truth at an arbitrary GPS timestamp."""
    return np.array(
        [
            np.interp(time_s, TRUTH_TIMES, TRUTH_STATES[:, dimension])
            for dimension in range(TRUTH_STATES.shape[1])
        ]
    )


def make_gps_measurements(seed=4):
    """Generate jittered 1 Hz noisy GPS speed measurements."""
    rng = np.random.default_rng(seed)
    gps_times = []
    current_time = 0.0
    while True:
        current_time += GPS_INTERVAL_S * rng.uniform(
            1.0 - GPS_TIMING_JITTER,
            1.0 + GPS_TIMING_JITTER,
        )
        if current_time >= DURATION_S:
            break
        gps_times.append(current_time)

    gps_times = np.asarray(gps_times)
    true_speeds = np.asarray(
        [truth_state_at_time(time_s)[SPEED] for time_s in gps_times]
    )
    measurements = true_speeds + rng.normal(
        0.0,
        GPS_SPEED_STD_MPS,
        size=len(gps_times),
    )
    return gps_times, measurements


def initial_distribution():
    """Start with deliberately poor drag and rolling-resistance guesses."""
    return Gaussian(
        mean=np.array([36.0, 0.020, 0.90]),
        covariance=np.diag([2.0**2, 0.008**2, 0.40**2]),
    )


def _make_filter():
    transition_model = StateTransitionModel(
        propagate_coastdown_state,
        process_noise_covariance(1.0),
    )
    return BayesianFilter(transition_model, initial_distribution())


def _run_filter(gps_times, gps_measurements):
    bayes_filter = _make_filter()
    state = bayes_filter.state
    previous_time = 0.0
    filtered_states = []

    for time_s, measurement in zip(gps_times, gps_measurements):
        delta_t_s = float(time_s) - previous_time
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_s)
        )
        predicted_state = bayes_filter.predict(
            state,
            delta_t_s,
            use_jacobian=False,
        )
        observation = Observation(
            np.array([measurement]),
            np.array([[GPS_SPEED_STD_MPS**2]]),
            observe_gps_speed,
        )
        state = bayes_filter.update(
            observation,
            predicted_state,
            use_jacobian=False,
        )
        bayes_filter.set_state(state)
        filtered_states.append(state)
        previous_time = float(time_s)

    return bayes_filter, filtered_states


def _rts_smooth(bayes_filter, filtered_states, filter_times):
    """RTS smoothing with process noise matched to each irregular interval."""
    smoothed_states = list(filtered_states)
    current_smoothed = filtered_states[-1]

    for index in range(len(filtered_states) - 2, -1, -1):
        delta_t_s = filter_times[index + 1] - filter_times[index]
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_s)
        )
        predicted_state, cross_covariance = (
            bayes_filter.transition_model.predict_with_cross_covariance(
                filtered_states[index],
                delta_t_s,
                use_jacobian=False,
            )
        )
        smoother_gain = cross_covariance @ np.linalg.inv(
            predicted_state.covariance()
        )
        smoothed_mean = filtered_states[index].mean() + smoother_gain @ (
            current_smoothed.mean() - predicted_state.mean()
        )
        smoothed_covariance = filtered_states[index].covariance() + smoother_gain @ (
            current_smoothed.covariance() - predicted_state.covariance()
        ) @ smoother_gain.T
        current_smoothed = Gaussian(smoothed_mean, smoothed_covariance)
        smoothed_states[index] = current_smoothed

    return smoothed_states


def _state_array(states):
    return np.asarray([state.mean() for state in states])


def _covariance_array(states):
    return np.asarray([state.covariance() for state in states])


def run_vehicle_coastdown(seed=4):
    """Infer ``C_rr`` and ``C_d A`` from one noisy GPS coast-down."""
    gps_times, gps_measurements = make_gps_measurements(seed)
    bayes_filter, filtered_states = _run_filter(gps_times, gps_measurements)
    smoothed_states = _rts_smooth(
        bayes_filter,
        filtered_states,
        gps_times,
    )

    filtered = _state_array(filtered_states)
    smoothed = _state_array(smoothed_states)
    truth = np.asarray([truth_state_at_time(time_s) for time_s in gps_times])

    filtered_speed_rmse = float(
        np.sqrt(np.mean((filtered[:, SPEED] - truth[:, SPEED]) ** 2))
    )
    smoothed_speed_rmse = float(
        np.sqrt(np.mean((smoothed[:, SPEED] - truth[:, SPEED]) ** 2))
    )

    return {
        "times": gps_times,
        "truth": truth,
        "gps_measurements": gps_measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": _covariance_array(filtered_states),
        "smoothed_covariances": _covariance_array(smoothed_states),
        "filtered_speed_rmse_mps": filtered_speed_rmse,
        "smoothed_speed_rmse_mps": smoothed_speed_rmse,
        "final_rolling_resistance": float(filtered[-1, ROLLING_RESISTANCE]),
        "final_drag_area_m2": float(filtered[-1, DRAG_AREA]),
    }


def _print_summary(result):
    print("Flat-road coast-down parameter estimation")
    print(
        f"  speed range: {INITIAL_SPEED_MPS:.1f} -> "
        f"{result['truth'][-1, SPEED]:.1f} m/s"
    )
    print(
        f"  C_rr: 0.020 initial -> {result['final_rolling_resistance']:.5f} "
        f"(truth {TRUE_ROLLING_RESISTANCE:.5f})"
    )
    print(
        f"  C_d A: 0.90 initial -> {result['final_drag_area_m2']:.3f} m^2 "
        f"(truth {TRUE_DRAG_AREA_M2:.3f} m^2)"
    )
    print(
        f"  speed RMSE: {result['filtered_speed_rmse_mps']:.3f} m/s filtered, "
        f"{result['smoothed_speed_rmse_mps']:.3f} m/s RTS"
    )


if __name__ == "__main__":
    _print_summary(run_vehicle_coastdown())

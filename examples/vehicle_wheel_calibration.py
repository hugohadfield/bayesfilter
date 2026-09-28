"""Estimate wheel radius and speedometer bias while driving.

A synthetic vehicle carries three sensors:

* a high-rate wheel angular-rate sensor,
* a dashboard speedometer with an unknown additive bias,
* sparse GPS position and speed fixes.

The hidden state is

``[position, speed, acceleration, log(wheel_radius), speedometer_bias]``.

The example is deliberately a calibration problem rather than a pure tracking
problem. GPS provides an occasional absolute speed reference, while the wheel
sensor and dashboard speedometer keep running during a six-minute GPS outage.
The filter therefore learns both the effective rolling radius and the
speedometer bias, then uses those calibrated quantities to maintain speed and
position estimates when GPS disappears.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


POSITION = 0
SPEED = 1
ACCELERATION = 2
LOG_WHEEL_RADIUS = 3
SPEEDOMETER_BIAS = 4

DURATION_S = 30.0 * 60.0
TRUE_WHEEL_RADIUS_M = 0.335
TRUE_SPEEDOMETER_BIAS_MPS = 0.45

WHEEL_INTERVAL_S = 1.0
SPEEDOMETER_INTERVAL_S = 2.0
GPS_INTERVAL_S = 20.0
TIMING_JITTER = 0.10
GPS_TIMING_JITTER = 0.25
GPS_OUTAGE_START_S = 12.0 * 60.0
GPS_OUTAGE_END_S = 18.0 * 60.0

WHEEL_RATE_STD_RAD_S = 0.45
SPEEDOMETER_STD_MPS = 0.25
GPS_POSITION_STD_M = 3.0
GPS_SPEED_STD_MPS = 0.30


def propagate_vehicle_state(state, delta_t_s):
    """Constant-acceleration transition for the calibration state."""
    position, speed, acceleration, log_radius, bias = np.asarray(state)
    return np.array(
        [
            position + speed * delta_t_s + 0.5 * acceleration * delta_t_s**2,
            speed + acceleration * delta_t_s,
            acceleration,
            log_radius,
            bias,
        ]
    )


def process_noise_covariance(delta_t_s):
    """Scale random-walk process noise by the irregular event interval."""
    diffusion_std = np.array([0.03, 0.03, 0.10, 2e-5, 2e-4])
    return np.diag(diffusion_std**2 * max(delta_t_s, 1e-9))


def observe_wheel_rate(state):
    """Predict wheel angular rate from vehicle speed and effective radius."""
    speed = np.asarray(state)[SPEED]
    wheel_radius = np.exp(np.asarray(state)[LOG_WHEEL_RADIUS])
    return np.array([speed / wheel_radius])


def observe_speedometer(state):
    """Predict dashboard indicated speed including an additive calibration bias."""
    state = np.asarray(state)
    return np.array([state[SPEED] + state[SPEEDOMETER_BIAS]])


def observe_gps_position_speed(state):
    """GPS observes absolute longitudinal position and ground speed."""
    state = np.asarray(state)
    return np.array([state[POSITION], state[SPEED]])


def acceleration_profile(time_s):
    """Smooth, repeatable acceleration excitation for a varied road drive."""
    time_s = np.asarray(time_s)
    return (
        0.22 * np.sin(2.0 * np.pi * time_s / 170.0)
        + 0.12 * np.sin(2.0 * np.pi * time_s / 67.0)
        + 0.05 * np.sin(2.0 * np.pi * time_s / 310.0)
    )


def simulate_vehicle_truth(delta_t_s=0.25):
    """Generate a 30-minute drive with changing speed and fixed calibration."""
    times = np.arange(0.0, DURATION_S + delta_t_s, delta_t_s)
    acceleration = acceleration_profile(times)
    speed = np.empty_like(times)
    position = np.empty_like(times)
    speed[0] = 16.0
    position[0] = 0.0

    for index in range(len(times) - 1):
        # Mild speed-restoring term keeps the synthetic drive in a plausible
        # range while retaining rich acceleration excitation.
        actual_acceleration = acceleration[index] - 0.012 * (speed[index] - 16.0)
        speed[index + 1] = max(
            5.0,
            speed[index] + actual_acceleration * delta_t_s,
        )
        position[index + 1] = (
            position[index]
            + speed[index] * delta_t_s
            + 0.5 * actual_acceleration * delta_t_s**2
        )
        acceleration[index] = actual_acceleration

    acceleration[-1] = acceleration[-2]
    truth = np.column_stack(
        [
            position,
            speed,
            acceleration,
            np.full_like(times, np.log(TRUE_WHEEL_RADIUS_M)),
            np.full_like(times, TRUE_SPEEDOMETER_BIAS_MPS),
        ]
    )
    return times, truth


TRUTH_TIMES, TRUTH_STATES = simulate_vehicle_truth()


def truth_state_at_time(time_s):
    """Interpolate the dense simulated truth at an arbitrary sensor timestamp."""
    return np.array(
        [
            np.interp(time_s, TRUTH_TIMES, TRUTH_STATES[:, dimension])
            for dimension in range(TRUTH_STATES.shape[1])
        ]
    )


def _jittered_times(rng, mean_interval_s, jitter_fraction):
    times = []
    current_time = 0.0
    while True:
        current_time += mean_interval_s * rng.uniform(
            1.0 - jitter_fraction,
            1.0 + jitter_fraction,
        )
        if current_time >= DURATION_S:
            break
        times.append(current_time)
    return np.asarray(times)


def make_sensor_events(seed=715):
    """Generate asynchronous wheel, speedometer, and GPS measurements."""
    rng = np.random.default_rng(seed)
    wheel_times = _jittered_times(rng, WHEEL_INTERVAL_S, TIMING_JITTER)
    speedometer_times = _jittered_times(
        rng,
        SPEEDOMETER_INTERVAL_S,
        TIMING_JITTER,
    )
    gps_times_all = _jittered_times(rng, GPS_INTERVAL_S, GPS_TIMING_JITTER)
    gps_times = gps_times_all[
        (gps_times_all < GPS_OUTAGE_START_S)
        | (gps_times_all > GPS_OUTAGE_END_S)
    ]

    events = []
    wheel_measurements = []
    for time_s in wheel_times:
        truth = truth_state_at_time(time_s)
        measurement = (
            truth[SPEED] / TRUE_WHEEL_RADIUS_M
            + rng.normal(0.0, WHEEL_RATE_STD_RAD_S)
        )
        wheel_measurements.append(measurement)
        events.append((time_s, "wheel", measurement))

    speedometer_measurements = []
    for time_s in speedometer_times:
        truth = truth_state_at_time(time_s)
        measurement = (
            truth[SPEED]
            + TRUE_SPEEDOMETER_BIAS_MPS
            + rng.normal(0.0, SPEEDOMETER_STD_MPS)
        )
        speedometer_measurements.append(measurement)
        events.append((time_s, "speedometer", measurement))

    gps_measurements = []
    for time_s in gps_times:
        truth = truth_state_at_time(time_s)
        measurement = np.array(
            [
                truth[POSITION] + rng.normal(0.0, GPS_POSITION_STD_M),
                truth[SPEED] + rng.normal(0.0, GPS_SPEED_STD_MPS),
            ]
        )
        gps_measurements.append(measurement)
        events.append((time_s, "gps", measurement))

    events.sort(key=lambda event: event[0])
    return {
        "events": events,
        "wheel_times": wheel_times,
        "wheel_measurements": np.asarray(wheel_measurements),
        "speedometer_times": speedometer_times,
        "speedometer_measurements": np.asarray(speedometer_measurements),
        "gps_times_all": gps_times_all,
        "gps_times": gps_times,
        "gps_measurements": np.asarray(gps_measurements),
    }


def initial_distribution():
    """Deliberately miscalibrated starting point."""
    return Gaussian(
        mean=np.array(
            [
                30.0,
                14.0,
                0.0,
                np.log(0.300),
                -0.30,
            ]
        ),
        covariance=np.diag(
            [
                25.0**2,
                2.0**2,
                0.50**2,
                0.10**2,
                0.80**2,
            ]
        ),
    )


def _make_filter():
    transition_model = StateTransitionModel(
        propagate_vehicle_state,
        process_noise_covariance(1.0),
    )
    return BayesianFilter(transition_model, initial_distribution())


def _observation_for_event(event_type, measurement):
    if event_type == "wheel":
        return Observation(
            np.array([measurement]),
            np.array([[WHEEL_RATE_STD_RAD_S**2]]),
            observe_wheel_rate,
        )
    if event_type == "speedometer":
        return Observation(
            np.array([measurement]),
            np.array([[SPEEDOMETER_STD_MPS**2]]),
            observe_speedometer,
        )
    if event_type == "gps":
        return Observation(
            measurement,
            np.diag([GPS_POSITION_STD_M**2, GPS_SPEED_STD_MPS**2]),
            observe_gps_position_speed,
        )
    raise ValueError(f"Unknown event type: {event_type}")


def _run_filter(events):
    bayes_filter = _make_filter()
    state = bayes_filter.state
    previous_time = 0.0
    filtered_states = []
    filter_times = []

    for time_s, event_type, measurement in events:
        delta_t_s = float(time_s) - previous_time
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_s)
        )
        predicted_state = bayes_filter.predict(
            state,
            delta_t_s,
            use_jacobian=False,
        )
        state = bayes_filter.update(
            _observation_for_event(event_type, measurement),
            predicted_state,
            use_jacobian=False,
        )
        bayes_filter.set_state(state)
        filtered_states.append(state)
        filter_times.append(float(time_s))
        previous_time = float(time_s)

    return bayes_filter, filtered_states, np.asarray(filter_times)


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


def run_vehicle_wheel_calibration(seed=715):
    """Estimate wheel radius and dashboard speed bias over one synthetic drive."""
    sensor_data = make_sensor_events(seed)
    bayes_filter, filtered_states, filter_times = _run_filter(sensor_data["events"])
    smoothed_states = _rts_smooth(
        bayes_filter,
        filtered_states,
        filter_times,
    )

    filtered = _state_array(filtered_states)
    smoothed = _state_array(smoothed_states)
    truth = np.asarray([truth_state_at_time(time_s) for time_s in filter_times])

    filtered_speed_rmse = float(
        np.sqrt(np.mean((filtered[:, SPEED] - truth[:, SPEED]) ** 2))
    )
    smoothed_speed_rmse = float(
        np.sqrt(np.mean((smoothed[:, SPEED] - truth[:, SPEED]) ** 2))
    )

    return {
        "times": filter_times,
        "truth": truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": _covariance_array(filtered_states),
        "smoothed_covariances": _covariance_array(smoothed_states),
        "filtered_speed_rmse_mps": filtered_speed_rmse,
        "smoothed_speed_rmse_mps": smoothed_speed_rmse,
        "final_wheel_radius_m": float(np.exp(filtered[-1, LOG_WHEEL_RADIUS])),
        "final_speedometer_bias_mps": float(filtered[-1, SPEEDOMETER_BIAS]),
        **sensor_data,
    }


def _print_summary(result):
    print("Vehicle wheel-radius and speedometer calibration")
    print(
        f"  wheel radius: {np.exp(result['filtered'][0, LOG_WHEEL_RADIUS]):.3f} -> "
        f"{result['final_wheel_radius_m']:.5f} m "
        f"(truth {TRUE_WHEEL_RADIUS_M:.3f} m)"
    )
    print(
        f"  speedometer bias: {result['filtered'][0, SPEEDOMETER_BIAS]:+.3f} -> "
        f"{result['final_speedometer_bias_mps']:+.3f} m/s "
        f"(truth {TRUE_SPEEDOMETER_BIAS_MPS:+.3f} m/s)"
    )
    print(
        f"  speed RMSE: {result['filtered_speed_rmse_mps']:.3f} m/s filtered, "
        f"{result['smoothed_speed_rmse_mps']:.3f} m/s RTS"
    )
    print(
        f"  GPS outage: {GPS_OUTAGE_START_S / 60:.0f}-{GPS_OUTAGE_END_S / 60:.0f} min"
    )


if __name__ == "__main__":
    _print_summary(run_vehicle_wheel_calibration())

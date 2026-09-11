"""Estimate wheel slip while driving across changing terrain.

A vehicle with a known effective wheel radius drives across asphalt, gravel,
sand, wet grass, and asphalt again. The filter is *not* told which material is
under the vehicle. It only sees frequent noisy wheel angular-rate measurements
and sparse noisy GPS position/speed fixes, and it must infer the changing
longitudinal slip ratio online.

The hidden state is

``[position, speed, acceleration, slip_logit]``.

Slip is represented in logit space so the inferred ratio stays between zero
and one. For positive-driving slip we use

``slip = (r * omega - v) / (r * omega)``,

so the wheel observation model is

``omega = v / (r * (1 - slip))``.

This makes the observation nonlinear and gives a compact example of using the
unscented filter for online traction estimation rather than ordinary tracking.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


POSITION = 0
SPEED = 1
ACCELERATION = 2
SLIP_LOGIT = 3

KNOWN_WHEEL_RADIUS_M = 0.335
DURATION_S = 20.0 * 60.0

WHEEL_INTERVAL_S = 0.5
GPS_INTERVAL_S = 8.0
WHEEL_TIMING_JITTER = 0.10
GPS_TIMING_JITTER = 0.20

WHEEL_RATE_STD_RAD_S = 0.25
GPS_POSITION_STD_M = 2.5
GPS_SPEED_STD_MPS = 0.20

# Material labels are used only by the simulator and reporting code. They are
# never supplied to the filter.
TERRAIN_SEGMENTS = (
    (0.0, 240.0, "asphalt", 0.015),
    (240.0, 480.0, "gravel", 0.075),
    (480.0, 780.0, "sand", 0.240),
    (780.0, 1020.0, "wet grass", 0.140),
    (1020.0, 1200.0, "asphalt", 0.020),
)


def sigmoid(value):
    """Map the unconstrained slip latent variable to a ratio in ``(0, 1)``."""
    value = np.asarray(value)
    return 1.0 / (1.0 + np.exp(-value))


def logit(probability):
    """Map a probability-like slip ratio in ``(0, 1)`` to the real line."""
    probability = np.asarray(probability)
    return np.log(probability / (1.0 - probability))


def propagate_slip_state(state, delta_t_s):
    """Propagate position/speed with constant acceleration and slip random walk."""
    position, speed, acceleration, slip_logit = np.asarray(state)
    return np.array(
        [
            position + speed * delta_t_s + 0.5 * acceleration * delta_t_s**2,
            speed + acceleration * delta_t_s,
            acceleration,
            slip_logit,
        ]
    )


def process_noise_covariance(delta_t_s):
    """Scale random-walk process noise by the irregular sensor interval."""
    diffusion_std = np.array([0.02, 0.02, 0.11, 0.10])
    return np.diag(diffusion_std**2 * max(delta_t_s, 1e-9))


def observe_wheel_rate(state):
    """Predict wheel angular rate from ground speed and longitudinal slip."""
    state = np.asarray(state)
    slip = sigmoid(state[SLIP_LOGIT])
    return np.array(
        [
            state[SPEED]
            / (KNOWN_WHEEL_RADIUS_M * (1.0 - slip))
        ]
    )


def observe_gps_position_speed(state):
    """GPS observes absolute longitudinal position and ground speed."""
    state = np.asarray(state)
    return np.array([state[POSITION], state[SPEED]])


def acceleration_profile(time_s):
    """Smooth throttle/braking excitation used by the synthetic drive."""
    time_s = np.asarray(time_s)
    return (
        0.18 * np.sin(2.0 * np.pi * time_s / 130.0)
        + 0.09 * np.sin(2.0 * np.pi * time_s / 49.0)
        + 0.04 * np.sin(2.0 * np.pi * time_s / 270.0)
    )


def terrain_at_time(time_s):
    """Return ``(name, nominal_slip)`` for the simulated terrain."""
    for start_s, end_s, name, nominal_slip in TERRAIN_SEGMENTS:
        if start_s <= time_s < end_s:
            return name, nominal_slip
    return TERRAIN_SEGMENTS[-1][2], TERRAIN_SEGMENTS[-1][3]


def _smooth_terrain_slip(times, delta_t_s):
    """Give physical terrain transitions a short finite response time."""
    slip = np.empty(len(times))
    slip[0] = TERRAIN_SEGMENTS[0][3]
    response_time_s = 12.0

    for index in range(len(times) - 1):
        _, target_slip = terrain_at_time(times[index])
        slip[index + 1] = slip[index] + (
            delta_t_s / response_time_s * (target_slip - slip[index])
        )

    return slip


def simulate_slip_truth(delta_t_s=0.25):
    """Generate a 20-minute drive across several surfaces."""
    times = np.arange(0.0, DURATION_S + delta_t_s, delta_t_s)
    commanded_acceleration = acceleration_profile(times)

    position = np.empty_like(times)
    speed = np.empty_like(times)
    acceleration = np.empty_like(times)
    position[0] = 0.0
    speed[0] = 13.0

    for index in range(len(times) - 1):
        actual_acceleration = (
            commanded_acceleration[index]
            - 0.015 * (speed[index] - 13.0)
        )
        acceleration[index] = actual_acceleration
        speed[index + 1] = max(
            6.0,
            speed[index] + actual_acceleration * delta_t_s,
        )
        position[index + 1] = (
            position[index]
            + speed[index] * delta_t_s
            + 0.5 * actual_acceleration * delta_t_s**2
        )
    acceleration[-1] = acceleration[-2]

    slip = _smooth_terrain_slip(times, delta_t_s)
    nominal_slip = np.asarray([terrain_at_time(time_s)[1] for time_s in times])

    # Loose surfaces exhibit a little more effective wheel slip under positive
    # drive torque. The filter is not given this rule; it estimates the
    # resulting effective slip directly from the sensors.
    slip += (
        0.012
        * np.maximum(acceleration, 0.0)
        * (1.0 + 3.0 * nominal_slip)
    )
    slip = np.clip(slip, 0.001, 0.45)

    truth = np.column_stack(
        [
            position,
            speed,
            acceleration,
            logit(slip),
        ]
    )
    return times, truth, slip


TRUTH_TIMES, TRUTH_STATES, TRUE_SLIP = simulate_slip_truth()


def truth_state_at_time(time_s):
    """Interpolate the dense synthetic state at a sensor timestamp."""
    return np.array(
        [
            np.interp(time_s, TRUTH_TIMES, TRUTH_STATES[:, dimension])
            for dimension in range(TRUTH_STATES.shape[1])
        ]
    )


def true_slip_at_time(time_s):
    return float(np.interp(time_s, TRUTH_TIMES, TRUE_SLIP))


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


def make_sensor_events(seed=411):
    """Generate asynchronous wheel-rate and GPS measurements."""
    rng = np.random.default_rng(seed)
    wheel_times = _jittered_times(
        rng,
        WHEEL_INTERVAL_S,
        WHEEL_TIMING_JITTER,
    )
    gps_times = _jittered_times(
        rng,
        GPS_INTERVAL_S,
        GPS_TIMING_JITTER,
    )

    events = []
    wheel_measurements = []
    for time_s in wheel_times:
        truth = truth_state_at_time(time_s)
        slip = true_slip_at_time(time_s)
        measurement = (
            truth[SPEED]
            / (KNOWN_WHEEL_RADIUS_M * (1.0 - slip))
            + rng.normal(0.0, WHEEL_RATE_STD_RAD_S)
        )
        wheel_measurements.append(measurement)
        events.append((time_s, "wheel", measurement))

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
        "gps_times": gps_times,
        "gps_measurements": np.asarray(gps_measurements),
    }


def initial_distribution():
    """Start with a deliberately mediocre kinematic and slip estimate."""
    return Gaussian(
        mean=np.array([20.0, 11.5, 0.0, logit(0.04)]),
        covariance=np.diag([20.0**2, 1.5**2, 0.40**2, 1.0**2]),
    )


def _make_filter():
    transition_model = StateTransitionModel(
        propagate_slip_state,
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
    """Run RTS smoothing with process noise matched to each event interval."""
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


def _terrain_summary(times, true_slip, estimated_slip):
    summary = []
    for start_s, end_s, name, _nominal_slip in TERRAIN_SEGMENTS:
        # Skip the first 30 seconds after each boundary so this number describes
        # the settled surface rather than the transition itself.
        mask = (times >= start_s + 30.0) & (times < end_s - 10.0)
        if not np.any(mask):
            continue
        summary.append(
            {
                "terrain": name,
                "start_s": start_s,
                "end_s": end_s,
                "true_mean_slip": float(np.mean(true_slip[mask])),
                "estimated_mean_slip": float(np.mean(estimated_slip[mask])),
                "mean_absolute_error": float(
                    np.mean(np.abs(estimated_slip[mask] - true_slip[mask]))
                ),
            }
        )
    return summary


def run_wheel_slip_estimation(seed=411):
    """Estimate changing wheel slip without using terrain labels in the filter."""
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
    true_slip = np.asarray([true_slip_at_time(time_s) for time_s in filter_times])
    filtered_slip = sigmoid(filtered[:, SLIP_LOGIT])
    smoothed_slip = sigmoid(smoothed[:, SLIP_LOGIT])

    filtered_slip_rmse = float(
        np.sqrt(np.mean((filtered_slip - true_slip) ** 2))
    )
    smoothed_slip_rmse = float(
        np.sqrt(np.mean((smoothed_slip - true_slip) ** 2))
    )

    return {
        "times": filter_times,
        "truth": truth,
        "true_slip": true_slip,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_slip": filtered_slip,
        "smoothed_slip": smoothed_slip,
        "filtered_covariances": _covariance_array(filtered_states),
        "smoothed_covariances": _covariance_array(smoothed_states),
        "filtered_slip_rmse": filtered_slip_rmse,
        "smoothed_slip_rmse": smoothed_slip_rmse,
        "terrain_summary": _terrain_summary(
            filter_times,
            true_slip,
            filtered_slip,
        ),
        **sensor_data,
    }


def _print_summary(result):
    print("Online wheel-slip estimation across changing terrain")
    print(
        f"  slip RMSE: {100 * result['filtered_slip_rmse']:.2f} percentage points "
        f"filtered, {100 * result['smoothed_slip_rmse']:.2f} RTS"
    )
    for segment in result["terrain_summary"]:
        print(
            f"  {segment['terrain']:<10} "
            f"true {100 * segment['true_mean_slip']:.1f}%  "
            f"filtered {100 * segment['estimated_mean_slip']:.1f}%"
        )


if __name__ == "__main__":
    _print_summary(run_wheel_slip_estimation())

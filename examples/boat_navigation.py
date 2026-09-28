"""Sun-aided navigation around an island with sparse human observations.

This example imagines a small sailing vessel before modern electronic
navigation. The boat has a noisy speed log, the crew occasionally estimates a
bearing and rough distance to one charted headland, and a sun-compass-like
measurement supplies the direction of the Sun in the boat frame. The global
Sun azimuth is assumed known from time and an ephemeris.

The filter jointly estimates position, heading, speed, turn rate, and a steady
world-frame ocean current. A no-Sun ablation shows how the global orientation
cue helps both the forward filter and RTS smoother.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


STATE_SIZE = 7
POSITION = slice(0, 2)
HEADING = 2
SPEED = 3
TURN_RATE = 4
CURRENT = slice(5, 7)

DURATION_S = 6.0 * 60.0 * 60.0
TRUE_SPEED_MPS = 3.0
TRUE_TURN_RATE_RAD_S = 2.0 * np.pi / DURATION_S
TRUE_CURRENT_MPS = np.array([0.25, -0.15])
ISLAND_RADIUS_M = 5500.0
ROUTE_RADIUS_M = TRUE_SPEED_MPS / TRUE_TURN_RATE_RAD_S

SPEED_INTERVAL_S = 10.0 * 60.0
LANDMARK_INTERVAL_S = 30.0 * 60.0
SUN_INTERVAL_S = 60.0
TIMING_JITTER = 0.20
SPEED_STD_MPS = 0.60
LANDMARK_BEARING_STD_RAD = np.deg2rad(20.0)
SUN_STD_RAD = np.deg2rad(25.0)
RANGE_STD_FLOOR_M = 500.0
RANGE_STD_FRACTION = 0.40
CLOUD_START_S = 2.5 * 60.0 * 60.0
CLOUD_END_S = 3.0 * 60.0 * 60.0
MISSED_LANDMARK_SIGHTINGS = {4, 5}


def angle_difference(first, second):
    """Return wrapped ``first - second`` in radians."""
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def propagate_boat_state(state, delta_t_s):
    """Propagate [x, y, heading, speed, turn rate, current east, current north]."""
    x, y, heading, speed, turn_rate, current_east, current_north = np.asarray(state)
    heading_increment = turn_rate * delta_t_s

    if abs(turn_rate) < 1e-10:
        delta_x_water = delta_t_s * speed * np.cos(heading)
        delta_y_water = delta_t_s * speed * np.sin(heading)
    else:
        delta_x_water = speed / turn_rate * (
            np.sin(heading + heading_increment) - np.sin(heading)
        )
        delta_y_water = speed / turn_rate * (
            -np.cos(heading + heading_increment) + np.cos(heading)
        )

    return np.array(
        [
            x + delta_x_water + delta_t_s * current_east,
            y + delta_y_water + delta_t_s * current_north,
            heading + heading_increment,
            speed,
            turn_rate,
            current_east,
            current_north,
        ]
    )


def process_noise_covariance(delta_t_s):
    """Continuous-time random-walk noise integrated over an irregular interval."""
    diffusion_std_per_sqrt_s = np.array(
        [
            0.05,
            0.05,
            np.deg2rad(0.03),
            0.002,
            2e-7,
            5e-4,
            5e-4,
        ]
    )
    return np.diag(diffusion_std_per_sqrt_s**2 * max(delta_t_s, 1e-9))


def sun_global_azimuth(time_s):
    """Synthetic ephemeris: Sun azimuth sweeps 90 degrees during the passage."""
    return np.deg2rad(105.0 + 90.0 * time_s / DURATION_S)


def observe_speed(state):
    return np.array([state[SPEED]])


def make_sun_observation_func(time_s):
    global_azimuth = sun_global_azimuth(time_s)

    def observe_sun_vector(state):
        relative = global_azimuth - state[HEADING]
        return np.array([np.cos(relative), np.sin(relative)])

    return observe_sun_vector


def make_landmark_bearing_func(landmark):
    landmark = np.asarray(landmark)

    def observe_landmark_bearing(state):
        delta = landmark - state[POSITION]
        global_bearing = np.arctan2(delta[1], delta[0])
        relative = global_bearing - state[HEADING]
        return np.array([np.cos(relative), np.sin(relative)])

    return observe_landmark_bearing


def make_landmark_range_func(landmark):
    landmark = np.asarray(landmark)

    def observe_landmark_range(state):
        return np.array([np.linalg.norm(landmark - state[POSITION])])

    return observe_landmark_range


def _rotate_unit_vector(vector, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ]
    )


def _unit_vector_covariance(angle_std_rad):
    """Approximate angular uncertainty as isotropic chord uncertainty."""
    vector_std = np.sin(angle_std_rad)
    return vector_std**2 * np.eye(2)


def _jittered_times(rng, mean_interval_s, duration_s=DURATION_S):
    times = []
    current_time = 0.0
    while True:
        current_time += mean_interval_s * rng.uniform(
            1.0 - TIMING_JITTER,
            1.0 + TIMING_JITTER,
        )
        if current_time >= duration_s:
            break
        times.append(current_time)
    return np.asarray(times)


def headland_positions(count=12):
    """Known charted headlands distributed around the island coast."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        [
            ISLAND_RADIUS_M * np.cos(angles),
            ISLAND_RADIUS_M * np.sin(angles),
        ]
    )


def initial_truth_state():
    return np.array(
        [
            ROUTE_RADIUS_M,
            0.0,
            np.pi / 2.0,
            TRUE_SPEED_MPS,
            TRUE_TURN_RATE_RAD_S,
            *TRUE_CURRENT_MPS,
        ]
    )


def truth_state_at_time(time_s):
    return propagate_boat_state(initial_truth_state(), float(time_s))


def _make_sensor_events(rng):
    speed_times = _jittered_times(rng, SPEED_INTERVAL_S)
    sun_times_all = _jittered_times(rng, SUN_INTERVAL_S)
    sun_times = sun_times_all[
        (sun_times_all < CLOUD_START_S) | (sun_times_all > CLOUD_END_S)
    ]
    landmark_schedule = _jittered_times(rng, LANDMARK_INTERVAL_S)
    headlands = headland_positions()

    events = []
    speed_measurements = []
    for time_s in speed_times:
        truth = truth_state_at_time(time_s)
        measurement = truth[SPEED] + rng.normal(0.0, SPEED_STD_MPS)
        speed_measurements.append(measurement)
        events.append((time_s, "speed", {"measurement": measurement}))

    sun_measurements = []
    for time_s in sun_times:
        truth = truth_state_at_time(time_s)
        observation_func = make_sun_observation_func(time_s)
        true_vector = observation_func(truth)
        noisy_vector = _rotate_unit_vector(
            true_vector,
            rng.normal(0.0, SUN_STD_RAD),
        )
        sun_measurements.append(noisy_vector)
        events.append((time_s, "sun", {"measurement": noisy_vector}))

    landmark_records = []
    for sighting_index, time_s in enumerate(landmark_schedule):
        if sighting_index in MISSED_LANDMARK_SIGHTINGS:
            continue
        truth = truth_state_at_time(time_s)
        distances = np.linalg.norm(headlands - truth[POSITION], axis=1)
        landmark_index = int(np.argmin(distances))
        landmark = headlands[landmark_index]

        bearing_func = make_landmark_bearing_func(landmark)
        true_bearing_vector = bearing_func(truth)
        bearing_measurement = _rotate_unit_vector(
            true_bearing_vector,
            rng.normal(0.0, LANDMARK_BEARING_STD_RAD),
        )

        true_range = distances[landmark_index]
        range_std = RANGE_STD_FLOOR_M + RANGE_STD_FRACTION * true_range
        range_measurement = max(100.0, true_range + rng.normal(0.0, range_std))

        record = {
            "time": time_s,
            "landmark_index": landmark_index,
            "landmark": landmark,
            "bearing_measurement": bearing_measurement,
            "range_measurement": range_measurement,
            "true_range": true_range,
        }
        landmark_records.append(record)
        events.append((time_s, "landmark", record))

    events.sort(key=lambda event: event[0])
    return {
        "events": events,
        "speed_times": speed_times,
        "speed_measurements": np.asarray(speed_measurements),
        "sun_times_all": sun_times_all,
        "sun_times": sun_times,
        "sun_measurements": np.asarray(sun_measurements),
        "landmark_schedule": landmark_schedule,
        "landmark_records": landmark_records,
        "headlands": headlands,
    }


def _initial_distribution():
    truth = initial_truth_state()
    initial_mean = truth + np.array(
        [
            1500.0,
            -1000.0,
            np.deg2rad(15.0),
            -0.40,
            -0.20 * TRUE_TURN_RATE_RAD_S,
            -TRUE_CURRENT_MPS[0],
            -TRUE_CURRENT_MPS[1],
        ]
    )
    initial_covariance = np.diag(
        [
            2000.0**2,
            2000.0**2,
            np.deg2rad(25.0) ** 2,
            0.70**2,
            (0.50 * TRUE_TURN_RATE_RAD_S) ** 2,
            0.40**2,
            0.40**2,
        ]
    )
    return Gaussian(initial_mean, initial_covariance)


def _make_filter():
    transition_model = StateTransitionModel(
        propagate_boat_state,
        process_noise_covariance(1.0),
    )
    return BayesianFilter(transition_model, _initial_distribution())


def _range_observation(record, predicted_state):
    landmark = record["landmark"]
    range_func = make_landmark_range_func(landmark)
    predicted_range = float(range_func(predicted_state.mean())[0])
    range_std = RANGE_STD_FLOOR_M + RANGE_STD_FRACTION * predicted_range
    return Observation(
        np.array([record["range_measurement"]]),
        np.array([[range_std**2]]),
        range_func,
    )


def _apply_event(bayes_filter, predicted_state, event, use_sun):
    time_s, event_type, data = event
    if event_type == "speed":
        observation = Observation(
            np.array([data["measurement"]]),
            np.array([[SPEED_STD_MPS**2]]),
            observe_speed,
        )
        return bayes_filter.update(observation, predicted_state, use_jacobian=False)

    if event_type == "sun":
        if not use_sun:
            return predicted_state
        observation = Observation(
            data["measurement"],
            _unit_vector_covariance(SUN_STD_RAD),
            make_sun_observation_func(time_s),
        )
        return bayes_filter.update(observation, predicted_state, use_jacobian=False)

    if event_type == "landmark":
        bearing_observation = Observation(
            data["bearing_measurement"],
            _unit_vector_covariance(LANDMARK_BEARING_STD_RAD),
            make_landmark_bearing_func(data["landmark"]),
        )
        state = bayes_filter.update(
            bearing_observation,
            predicted_state,
            use_jacobian=False,
        )
        return bayes_filter.update(
            _range_observation(data, state),
            state,
            use_jacobian=False,
        )

    raise ValueError(f"Unknown event type: {event_type}")


def _run_filter(events, use_sun):
    bayes_filter = _make_filter()
    state = bayes_filter.state
    previous_time = 0.0
    filtered_states = []
    filter_times = []

    for event in events:
        time_s = float(event[0])
        delta_t_s = time_s - previous_time
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_s)
        )
        predicted_state = bayes_filter.predict(
            state,
            delta_t_s,
            use_jacobian=False,
        )
        state = _apply_event(bayes_filter, predicted_state, event, use_sun)
        bayes_filter.set_state(state)
        filtered_states.append(state)
        filter_times.append(time_s)
        previous_time = time_s

    return bayes_filter, filtered_states, np.asarray(filter_times)


def _rts_smooth(bayes_filter, filtered_states, filter_times):
    """RTS equations with Q scaled to each irregular event interval."""
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


def _metrics(states, truth):
    position_error = states[:, POSITION] - truth[:, POSITION]
    heading_error = angle_difference(states[:, HEADING], truth[:, HEADING])
    current_error = states[:, CURRENT] - truth[:, CURRENT]
    return {
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        ),
        "heading_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(heading_error**2)))
        ),
        "current_rmse_mps": float(
            np.sqrt(np.mean(np.sum(current_error**2, axis=1)))
        ),
    }


def run_boat_navigation(seed=1201):
    """Run Sun-aided and no-Sun filters and RTS smoothers on one island voyage."""
    rng = np.random.default_rng(seed)
    sensor_data = _make_sensor_events(rng)
    events = sensor_data["events"]

    sun_filter, sun_filtered_states, filter_times = _run_filter(events, use_sun=True)
    no_sun_filter, no_sun_filtered_states, no_sun_times = _run_filter(
        events,
        use_sun=False,
    )
    if not np.allclose(filter_times, no_sun_times):
        raise RuntimeError("Sun and no-Sun runs must use the same prediction times")

    sun_smoothed_states = _rts_smooth(
        sun_filter,
        sun_filtered_states,
        filter_times,
    )
    no_sun_smoothed_states = _rts_smooth(
        no_sun_filter,
        no_sun_filtered_states,
        filter_times,
    )

    truth = np.asarray([truth_state_at_time(time_s) for time_s in filter_times])
    filtered = _state_array(sun_filtered_states)
    smoothed = _state_array(sun_smoothed_states)
    no_sun_filtered = _state_array(no_sun_filtered_states)
    no_sun_smoothed = _state_array(no_sun_smoothed_states)

    return {
        "times": filter_times,
        "truth": truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "no_sun_filtered": no_sun_filtered,
        "no_sun_smoothed": no_sun_smoothed,
        "filtered_covariances": _covariance_array(sun_filtered_states),
        "smoothed_covariances": _covariance_array(sun_smoothed_states),
        "no_sun_filtered_covariances": _covariance_array(no_sun_filtered_states),
        "no_sun_smoothed_covariances": _covariance_array(no_sun_smoothed_states),
        "metrics": {
            "sun_filter": _metrics(filtered, truth),
            "sun_rts": _metrics(smoothed, truth),
            "no_sun_filter": _metrics(no_sun_filtered, truth),
            "no_sun_rts": _metrics(no_sun_smoothed, truth),
        },
        **sensor_data,
    }


def _print_summary(result):
    print("Six-hour island passage")
    print(
        f"  speed log: ~10 min, sigma={SPEED_STD_MPS:.1f} m/s; "
        f"headland: ~30 min, sigma={np.rad2deg(LANDMARK_BEARING_STD_RAD):.0f} deg; "
        f"Sun: ~1 min, sigma={np.rad2deg(SUN_STD_RAD):.0f} deg"
    )
    print("  method          position RMSE    heading RMSE    current RMSE")
    for key, label in [
        ("no_sun_filter", "No Sun filter"),
        ("no_sun_rts", "No Sun RTS"),
        ("sun_filter", "Sun filter"),
        ("sun_rts", "Sun RTS"),
    ]:
        metrics = result["metrics"][key]
        print(
            f"  {label:<14} {metrics['position_rmse_m']:>9.0f} m"
            f" {metrics['heading_rmse_deg']:>12.2f} deg"
            f" {metrics['current_rmse_mps']:>12.3f} m/s"
        )
    print(
        "  final Sun RTS current: "
        f"[{result['smoothed'][-1, 5]:.3f}, {result['smoothed'][-1, 6]:.3f}] m/s; "
        f"truth [{TRUE_CURRENT_MPS[0]:.3f}, {TRUE_CURRENT_MPS[1]:.3f}] m/s"
    )


if __name__ == "__main__":
    from examples.boat_plotting import plot_boat_navigation

    result = run_boat_navigation()
    _print_summary(result)
    plot_boat_navigation(result, save_path="examples/figures/boat-navigation.png")

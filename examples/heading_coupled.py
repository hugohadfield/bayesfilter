"""Planar tracking with heading coupled to signed speed and curvature."""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


def propagate_heading_coupled_state(state, delta_t_s):
    """Propagate [x, y, heading, signed speed, curvature]."""
    x_position, y_position, heading, speed, curvature = np.asarray(state)
    heading_increment = delta_t_s * speed * curvature
    midpoint_heading = heading + 0.5 * heading_increment
    return np.array(
        [
            x_position + delta_t_s * speed * np.cos(midpoint_heading),
            y_position + delta_t_s * speed * np.sin(midpoint_heading),
            heading + heading_increment,
            speed,
            curvature,
        ]
    )


def heading_coupled_jacobian(state, delta_t_s):
    """Analytic Jacobian of :func:`propagate_heading_coupled_state`."""
    _, _, heading, speed, curvature = np.asarray(state)
    midpoint_heading = heading + 0.5 * delta_t_s * speed * curvature
    sine = np.sin(midpoint_heading)
    cosine = np.cos(midpoint_heading)
    jacobian = np.eye(5)
    jacobian[0, 2] = -delta_t_s * speed * sine
    jacobian[0, 3] = delta_t_s * cosine - 0.5 * delta_t_s**2 * speed * curvature * sine
    jacobian[0, 4] = -0.5 * delta_t_s**2 * speed**2 * sine
    jacobian[1, 2] = delta_t_s * speed * cosine
    jacobian[1, 3] = delta_t_s * sine + 0.5 * delta_t_s**2 * speed * curvature * cosine
    jacobian[1, 4] = 0.5 * delta_t_s**2 * speed**2 * cosine
    jacobian[2, 3] = delta_t_s * curvature
    jacobian[2, 4] = delta_t_s * speed
    return jacobian


def observe_global_position(state):
    return np.asarray(state)[:2]


def global_position_jacobian(_state):
    jacobian = np.zeros((2, 5))
    jacobian[:, :2] = np.eye(2)
    return jacobian


def observe_local_angular_rate(state):
    state = np.asarray(state)
    return np.array([state[3] * state[4]])


def local_angular_rate_jacobian(state):
    state = np.asarray(state)
    jacobian = np.zeros((1, 5))
    jacobian[0, 3] = state[4]
    jacobian[0, 4] = state[3]
    return jacobian


def signed_speed_profile(time_s):
    """Forward, stop, and reverse speed profile used by the simulation."""
    time_s = np.asarray(time_s)
    speed = np.empty_like(time_s, dtype=float)

    forward = time_s < 6.0
    slowing = (time_s >= 6.0) & (time_s < 8.0)
    stopped = (time_s >= 8.0) & (time_s < 10.0)
    reversing = (time_s >= 10.0) & (time_s < 12.0)
    backward = time_s >= 12.0

    speed[forward] = 1.2
    slowing_phase = (time_s[slowing] - 6.0) / 2.0
    slowing_weight = slowing_phase**2 * (3.0 - 2.0 * slowing_phase)
    speed[slowing] = 1.2 * (1.0 - slowing_weight)
    speed[stopped] = 0.0
    reversing_phase = (time_s[reversing] - 10.0) / 2.0
    reversing_weight = reversing_phase**2 * (3.0 - 2.0 * reversing_phase)
    speed[reversing] = -0.9 * reversing_weight
    speed[backward] = -0.9
    return speed


def curvature_profile(time_s):
    """Slowly varying signed curvature used by the simulation."""
    time_s = np.asarray(time_s)
    return 0.18 + 0.09 * np.sin(0.35 * time_s)


def simulate_heading_coupled_truth(times):
    """Simulate a vehicle that cannot translate sideways or turn in place."""
    times = np.asarray(times)
    truth = np.empty((len(times), 5))
    truth[0] = np.array(
        [0.0, 0.0, 0.35, signed_speed_profile(times[0]), curvature_profile(times[0])]
    )
    for index, delta_t_s in enumerate(np.diff(times)):
        midpoint_time = 0.5 * (times[index] + times[index + 1])
        midpoint_state = truth[index].copy()
        midpoint_state[3] = signed_speed_profile(midpoint_time)
        midpoint_state[4] = curvature_profile(midpoint_time)
        truth[index + 1] = propagate_heading_coupled_state(
            midpoint_state,
            delta_t_s,
        )
        truth[index + 1, 3] = signed_speed_profile(times[index + 1])
        truth[index + 1, 4] = curvature_profile(times[index + 1])
    return truth


def angle_difference(first, second):
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def _make_sensor_observations(
    truth,
    times,
    rng,
    position_interval_steps,
    position_std,
    angular_rate_std,
):
    position_times = times[position_interval_steps::position_interval_steps]
    position_truth = truth[position_interval_steps::position_interval_steps, :2]
    position_measurements = position_truth + rng.normal(
        0.0,
        position_std,
        position_truth.shape,
    )
    angular_rate_times = times[1:]
    angular_rate_truth = truth[1:, 3] * truth[1:, 4]
    angular_rate_measurements = angular_rate_truth + rng.normal(
        0.0,
        angular_rate_std,
        angular_rate_truth.shape,
    )

    events = []
    for time_s, measurement in zip(angular_rate_times, angular_rate_measurements):
        events.append(
            (
                time_s,
                Observation(
                    np.array([measurement]),
                    np.array([[angular_rate_std**2]]),
                    observe_local_angular_rate,
                    local_angular_rate_jacobian,
                ),
            )
        )
    for time_s, measurement in zip(position_times, position_measurements):
        events.append(
            (
                time_s,
                Observation(
                    measurement,
                    position_std**2 * np.eye(2),
                    observe_global_position,
                    global_position_jacobian,
                ),
            )
        )
    events.sort(key=lambda event: event[0])
    event_times = np.array([event[0] for event in events])
    observations = [event[1] for event in events]
    return {
        "observations": observations,
        "event_times": event_times,
        "position_times": position_times,
        "position_measurements": position_measurements,
        "angular_rate_times": angular_rate_times,
        "angular_rate_measurements": angular_rate_measurements,
    }


def run_heading_coupled_tracking(
    seed=401,
    duration_s=20.0,
    filter_rate_hz=20.0,
    position_rate_hz=5.0,
    use_jacobian=False,
    speed_process_std=0.06,
    curvature_process_std=0.009,
):
    """Fuse asynchronous global position and local angular-rate measurements."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    times = np.arange(round(duration_s * filter_rate_hz) + 1) * delta_t_s
    truth = simulate_heading_coupled_truth(times)
    position_interval_steps = round(filter_rate_hz / position_rate_hz)
    if not np.isclose(position_interval_steps * position_rate_hz, filter_rate_hz):
        raise ValueError("position_rate_hz must divide filter_rate_hz")

    position_std = 0.35
    angular_rate_std = 0.035
    sensor_data = _make_sensor_observations(
        truth,
        times,
        rng,
        position_interval_steps,
        position_std,
        angular_rate_std,
    )
    transition_model = StateTransitionModel(
        propagate_heading_coupled_state,
        np.diag([1e-5, 1e-5, 1e-5, speed_process_std**2, curvature_process_std**2]),
        heading_coupled_jacobian,
    )
    initial_mean = truth[0] + np.array([0.45, -0.35, 0.22, -0.25, 0.07])
    initial_covariance = np.diag([0.65**2, 0.65**2, 0.30**2, 0.35**2, 0.14**2])
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states, filter_times = bayes_filter.run(
        sensor_data["observations"],
        sensor_data["event_times"],
        rate_hz=filter_rate_hz,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )
    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])
    aligned_truth = truth[1 : len(filter_times) + 1]

    def position_rmse(states):
        return np.sqrt(
            np.mean(np.sum((states[:, :2] - aligned_truth[:, :2]) ** 2, axis=1))
        )

    def heading_rmse(states):
        errors = angle_difference(states[:, 2], aligned_truth[:, 2])
        return np.sqrt(np.mean(errors**2))

    return {
        "times": np.asarray(filter_times),
        "truth": aligned_truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "event_times": sensor_data["event_times"],
        "position_times": sensor_data["position_times"],
        "position_measurements": sensor_data["position_measurements"],
        "angular_rate_times": sensor_data["angular_rate_times"],
        "angular_rate_measurements": sensor_data["angular_rate_measurements"],
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "filtered_heading_rmse": heading_rmse(filtered),
        "smoothed_heading_rmse": heading_rmse(smoothed),
        "use_jacobian": use_jacobian,
        "position_rate_hz": position_rate_hz,
        "angular_rate_hz": filter_rate_hz,
    }

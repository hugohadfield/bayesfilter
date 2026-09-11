"""Known-correspondence landmark SLAM with range and bearing observations."""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


LANDMARK_POSITIONS = np.array(
    [
        [4.0, 4.5],
        [10.5, 1.0],
        [14.0, 7.5],
        [9.0, 13.0],
        [1.5, 11.5],
        [-4.5, 7.0],
        [-3.5, 0.0],
        [5.0, -3.5],
    ]
)

INITIAL_LANDMARK_OFFSETS = np.array(
    [
        [1.0, -0.8],
        [-1.1, 0.6],
        [0.8, 1.1],
        [-0.7, -1.2],
        [1.2, 0.7],
        [-0.9, 1.0],
        [0.6, -1.1],
        [-1.0, -0.7],
    ]
)


def landmark_slice(landmark_index):
    """Return the state slice for a landmark's two coordinates."""
    start = 5 + 2 * landmark_index
    return slice(start, start + 2)


def propagate_slam_state(state, delta_t_s):
    """Propagate robot pose, speed, yaw rate, and static landmarks."""
    state = np.asarray(state)
    propagated = state.copy()
    heading, speed, yaw_rate = state[2:5]
    midpoint_heading = heading + 0.5 * delta_t_s * yaw_rate
    propagated[0] += delta_t_s * speed * np.cos(midpoint_heading)
    propagated[1] += delta_t_s * speed * np.sin(midpoint_heading)
    propagated[2] += delta_t_s * yaw_rate
    return propagated


def slam_transition_jacobian(state, delta_t_s):
    """Return the analytic Jacobian of the coordinated-motion transition."""
    state = np.asarray(state)
    heading, speed, yaw_rate = state[2:5]
    midpoint_heading = heading + 0.5 * delta_t_s * yaw_rate
    sine = np.sin(midpoint_heading)
    cosine = np.cos(midpoint_heading)
    jacobian = np.eye(len(state))
    jacobian[0, 2] = -delta_t_s * speed * sine
    jacobian[0, 3] = delta_t_s * cosine
    jacobian[0, 4] = -0.5 * delta_t_s**2 * speed * sine
    jacobian[1, 2] = delta_t_s * speed * cosine
    jacobian[1, 3] = delta_t_s * sine
    jacobian[1, 4] = 0.5 * delta_t_s**2 * speed * cosine
    jacobian[2, 4] = delta_t_s
    return jacobian


def observe_slam_state(state, landmark_indices):
    """Observe odometry plus range and local bearing vectors to landmarks."""
    state = np.asarray(state)
    cosine = np.cos(state[2])
    sine = np.sin(state[2])
    world_to_local = np.array([[cosine, sine], [-sine, cosine]])
    observation = [state[3], state[4]]
    for landmark_index in landmark_indices:
        displacement = state[landmark_slice(landmark_index)] - state[:2]
        distance = np.linalg.norm(displacement)
        local_direction = world_to_local @ (displacement / distance)
        observation.extend([distance, local_direction[0], local_direction[1]])
    return np.asarray(observation)


def slam_observation_jacobian(state, landmark_indices):
    """Return the range/bearing-vector observation Jacobian."""
    state = np.asarray(state)
    cosine = np.cos(state[2])
    sine = np.sin(state[2])
    world_to_local = np.array([[cosine, sine], [-sine, cosine]])
    jacobian = np.zeros((2 + 3 * len(landmark_indices), len(state)))
    jacobian[0, 3] = 1.0
    jacobian[1, 4] = 1.0
    for visible_index, landmark_index in enumerate(landmark_indices):
        row = 2 + 3 * visible_index
        coordinates = landmark_slice(landmark_index)
        displacement = state[coordinates] - state[:2]
        distance = np.linalg.norm(displacement)
        world_direction = displacement / distance
        local_direction = world_to_local @ world_direction
        direction_derivative = world_to_local @ (
            np.eye(2) - np.outer(world_direction, world_direction)
        ) / distance

        jacobian[row, :2] = -world_direction
        jacobian[row, coordinates] = world_direction
        jacobian[row + 1 : row + 3, :2] = -direction_derivative
        jacobian[row + 1 : row + 3, coordinates] = direction_derivative
        jacobian[row + 1 : row + 3, 2] = np.array(
            [local_direction[1], -local_direction[0]]
        )
    return jacobian


def speed_profile(times_s):
    """Return the robot's forward speed around the loop."""
    times_s = np.asarray(times_s)
    return np.full_like(times_s, 1.02, dtype=float)


def yaw_rate_profile(times_s):
    """Return the turn rate for a nearly closed circular trajectory."""
    times_s = np.asarray(times_s)
    return np.full_like(times_s, 0.12, dtype=float)


def simulate_slam_truth(times_s):
    """Simulate a robot trajectory through the fixed landmark map."""
    times_s = np.asarray(times_s)
    dimension = 5 + 2 * len(LANDMARK_POSITIONS)
    truth = np.empty((len(times_s), dimension))
    truth[0, :5] = np.array(
        [0.0, 0.0, 0.25, speed_profile(times_s[0]), yaw_rate_profile(times_s[0])]
    )
    truth[0, 5:] = LANDMARK_POSITIONS.ravel()
    for index, delta_t_s in enumerate(np.diff(times_s)):
        truth[index + 1] = propagate_slam_state(truth[index], delta_t_s)
        truth[index + 1, 3] = speed_profile(times_s[index + 1])
        truth[index + 1, 4] = yaw_rate_profile(times_s[index + 1])
    return truth


def visible_landmarks(state, maximum_range=9.0, field_of_view_rad=np.deg2rad(250.0)):
    """Return landmarks visible within sensor range and field of view."""
    visible = []
    for landmark_index in range(len(LANDMARK_POSITIONS)):
        displacement = state[landmark_slice(landmark_index)] - state[:2]
        distance = np.linalg.norm(displacement)
        bearing = np.arctan2(displacement[1], displacement[0]) - state[2]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        if distance <= maximum_range and abs(bearing) <= 0.5 * field_of_view_rad:
            visible.append(landmark_index)
    return tuple(visible)


def _measurement_covariance(
    landmark_count,
    speed_std,
    yaw_rate_std,
    range_std,
    bearing_vector_std,
):
    variances = [speed_std**2, yaw_rate_std**2]
    variances.extend(
        [range_std**2, bearing_vector_std**2, bearing_vector_std**2]
        * landmark_count
    )
    return np.diag(variances)


def _noisy_measurement(
    state,
    landmark_indices,
    rng,
    speed_std,
    yaw_rate_std,
    range_std,
    bearing_std_rad,
):
    measurement = observe_slam_state(state, landmark_indices)
    measurement[:2] += rng.normal(0.0, [speed_std, yaw_rate_std])
    for visible_index in range(len(landmark_indices)):
        row = 2 + 3 * visible_index
        measurement[row] += rng.normal(0.0, range_std)
        direction = measurement[row + 1 : row + 3]
        angle = np.arctan2(direction[1], direction[0])
        angle += rng.normal(0.0, bearing_std_rad)
        measurement[row + 1 : row + 3] = [np.cos(angle), np.sin(angle)]
    return measurement


def run_known_correspondence_slam(
    seed=901,
    duration_s=52.0,
    delta_t_s=0.1,
    use_jacobian=True,
):
    """Jointly estimate a robot trajectory and a known-correspondence map."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s / delta_t_s) + 1) * delta_t_s
    truth = simulate_slam_truth(times)

    speed_std = 0.04
    yaw_rate_std = 0.008
    range_std = 0.10
    bearing_std_rad = np.deg2rad(0.8)
    observations = []
    measurements = []
    visible_indices = []
    for truth_state in truth[1:]:
        indices = visible_landmarks(truth_state)

        def observation_func(state, landmark_indices=indices):
            return observe_slam_state(state, landmark_indices)

        def jacobian_func(state, landmark_indices=indices):
            return slam_observation_jacobian(state, landmark_indices)

        measurement = _noisy_measurement(
            truth_state,
            indices,
            rng,
            speed_std,
            yaw_rate_std,
            range_std,
            bearing_std_rad,
        )
        covariance = _measurement_covariance(
            len(indices),
            speed_std,
            yaw_rate_std,
            range_std,
            bearing_std_rad,
        )
        observations.append(
            Observation(measurement, covariance, observation_func, jacobian_func)
        )
        measurements.append(measurement)
        visible_indices.append(indices)

    dimension = truth.shape[1]
    process_variances = np.full(dimension, 1e-10)
    process_variances[:5] = [2e-5, 2e-5, 8e-6, 0.045**2, 0.007**2]
    model = StateTransitionModel(
        propagate_slam_state,
        np.diag(process_variances),
        slam_transition_jacobian,
    )
    initial_mean = truth[0].copy()
    initial_mean[:5] += np.array([0.0, 0.0, 0.0, -0.12, 0.025])
    initial_mean[5:] += INITIAL_LANDMARK_OFFSETS.ravel()
    initial_variances = np.full(dimension, 1.5**2)
    initial_variances[:5] = [0.02**2, 0.02**2, 0.01**2, 0.20**2, 0.04**2]
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, np.diag(initial_variances)),
    )
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

    def position_rmse(states):
        return np.sqrt(np.mean(np.sum((states[:, :2] - truth[:, :2]) ** 2, axis=1)))

    def landmark_rmse(state):
        landmarks = state[5:].reshape(-1, 2)
        return np.sqrt(np.mean(np.sum((landmarks - LANDMARK_POSITIONS) ** 2, axis=1)))

    return {
        "times": times,
        "truth": truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "measurements": measurements,
        "visible_landmark_indices": visible_indices,
        "landmark_positions": LANDMARK_POSITIONS.copy(),
        "initial_landmark_estimates": initial_mean[5:].reshape(-1, 2),
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "initial_landmark_rmse": landmark_rmse(filtered[0]),
        "final_landmark_rmse": landmark_rmse(filtered[-1]),
        "use_jacobian": use_jacobian,
    }


if __name__ == "__main__":
    from examples.slam_plotting import plot_known_correspondence_slam

    result = run_known_correspondence_slam()
    path = Path(__file__).parent / "figures" / "known-correspondence-slam.png"
    plot_known_correspondence_slam(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f} m"
    )
    print(
        f"landmark RMSE: {result['initial_landmark_rmse']:.3f} -> "
        f"{result['final_landmark_rmse']:.3f} m"
    )
    print(f"saved {path}")

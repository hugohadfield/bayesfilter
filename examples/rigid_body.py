"""Rigid-body filtering with an SO(3) rotation-bivector state."""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


CANONICAL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.75, 0.0, 0.0],
        [0.0, 0.75, 0.0],
        [0.0, 0.0, 0.75],
    ]
)


def hat(vector):
    x_value, y_value, z_value = np.asarray(vector)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ]
    )


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def so3_exp(rotation_vector):
    """Map a rotation bivector/vector in R^3 to SO(3)."""
    rotation_vector = np.asarray(rotation_vector)
    angle = np.linalg.norm(rotation_vector)
    skew = hat(rotation_vector)
    if angle < 1e-8:
        return np.eye(3) + skew + 0.5 * skew @ skew
    return (
        np.eye(3)
        + (np.sin(angle) / angle) * skew
        + ((1.0 - np.cos(angle)) / angle**2) * skew @ skew
    )


def so3_log(rotation_matrix):
    """Map SO(3) to its principal rotation-bivector/vector coordinates."""
    rotation_matrix = np.asarray(rotation_matrix)
    cosine = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    antisymmetric = rotation_matrix - rotation_matrix.T
    if angle < 1e-8:
        return 0.5 * vee(antisymmetric)
    if np.pi - angle < 1e-5:
        symmetric = 0.5 * (rotation_matrix + rotation_matrix.T)
        _, eigenvectors = np.linalg.eigh(symmetric)
        axis = eigenvectors[:, -1]
        if np.dot(axis, vee(antisymmetric)) < 0.0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * vee(antisymmetric)


def compose_rotation_vectors(rotation_vector, increment):
    return so3_log(so3_exp(rotation_vector) @ so3_exp(increment))


def rotation_distance(first_rotation_vector, second_rotation_vector):
    relative = so3_exp(first_rotation_vector).T @ so3_exp(second_rotation_vector)
    return np.linalg.norm(so3_log(relative))


def torque_free_angular_acceleration(angular_velocity, inertia_matrix):
    angular_momentum = inertia_matrix @ angular_velocity
    return np.linalg.solve(
        inertia_matrix,
        -np.cross(angular_velocity, angular_momentum),
    )


def integrate_angular_velocity(angular_velocity, delta_t_s, inertia_matrix):
    """Return an RK4 angular-velocity step and its stage-weighted average."""

    def derivative(value):
        return torque_free_angular_acceleration(value, inertia_matrix)

    stage_1 = derivative(angular_velocity)
    midpoint_1 = angular_velocity + 0.5 * delta_t_s * stage_1
    stage_2 = derivative(midpoint_1)
    midpoint_2 = angular_velocity + 0.5 * delta_t_s * stage_2
    stage_3 = derivative(midpoint_2)
    endpoint = angular_velocity + delta_t_s * stage_3
    stage_4 = derivative(endpoint)
    next_angular_velocity = (
        angular_velocity
        + delta_t_s * (stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4) / 6.0
    )
    average_angular_velocity = (
        angular_velocity + 2.0 * midpoint_1 + 2.0 * midpoint_2 + endpoint
    ) / 6.0
    return next_angular_velocity, average_angular_velocity


def propagate_rigid_body_state(state, delta_t_s, inertia_matrix):
    """Propagate [position, velocity, rotation vector, body angular velocity]."""
    state = np.asarray(state)
    position = state[0:3]
    velocity = state[3:6]
    rotation_vector = state[6:9]
    angular_velocity = state[9:12]
    next_angular_velocity, average_angular_velocity = integrate_angular_velocity(
        angular_velocity,
        delta_t_s,
        inertia_matrix,
    )
    next_rotation_vector = compose_rotation_vectors(
        rotation_vector,
        delta_t_s * average_angular_velocity,
    )
    return np.concatenate(
        (
            position + delta_t_s * velocity,
            velocity,
            next_rotation_vector,
            next_angular_velocity,
        )
    )


def transform_canonical_points(state, canonical_points=CANONICAL_POINTS):
    """Observe labeled body-frame canonical points in world coordinates."""
    position = np.asarray(state)[0:3]
    rotation = so3_exp(np.asarray(state)[6:9])
    world_points = position + np.asarray(canonical_points) @ rotation.T
    return world_points.reshape(-1)


def inertia_from_log_ratios(log_ratios):
    ratios = np.exp(np.asarray(log_ratios))
    return np.diag(np.concatenate(([1.0], ratios)))


def rigid_body_energy(angular_velocity, inertia_matrix):
    return 0.5 * angular_velocity @ inertia_matrix @ angular_velocity


def angular_momentum_magnitude(angular_velocity, inertia_matrix):
    return np.linalg.norm(inertia_matrix @ angular_velocity)


def simulate_rigid_body_truth(initial_state, times, inertia_matrix):
    truth = [np.asarray(initial_state)]
    for delta_t_s in np.diff(times):
        truth.append(propagate_rigid_body_state(truth[-1], delta_t_s, inertia_matrix))
    return np.array(truth)


def _rigid_body_observations(truth, measurement_std, rng):
    measurement_dimension = CANONICAL_POINTS.size
    covariance = measurement_std**2 * np.eye(measurement_dimension)
    measurements = np.array(
        [
            transform_canonical_points(state)
            + rng.normal(0.0, measurement_std, measurement_dimension)
            for state in truth[1:]
        ]
    )
    observations = [
        Observation(
            measurement,
            covariance,
            transform_canonical_points,
        )
        for measurement in measurements
    ]
    return measurements, observations


def _collect_rigid_body_result(
    times,
    truth,
    measurements,
    filter_states,
    smoother_states,
    inertia_matrix,
):
    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])
    filtered_rotation_error = np.array(
        [
            rotation_distance(expected[6:9], actual[6:9])
            for expected, actual in zip(truth, filtered)
        ]
    )
    smoothed_rotation_error = np.array(
        [
            rotation_distance(expected[6:9], actual[6:9])
            for expected, actual in zip(truth, smoothed)
        ]
    )
    truth_energy = np.array(
        [rigid_body_energy(state[9:12], inertia_matrix) for state in truth]
    )
    truth_momentum = np.array(
        [angular_momentum_magnitude(state[9:12], inertia_matrix) for state in truth]
    )
    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_rotation_error": filtered_rotation_error,
        "smoothed_rotation_error": smoothed_rotation_error,
        "truth_energy": truth_energy,
        "truth_momentum": truth_momentum,
        "inertia_matrix": inertia_matrix,
    }


def run_known_inertia_rigid_body(seed=301, num_steps=45, delta_t_s=0.06):
    rng = np.random.default_rng(seed)
    times = np.arange(num_steps + 1, dtype=float) * delta_t_s
    inertia_matrix = np.diag([1.0, 1.4, 1.8])
    initial_truth = np.array(
        [
            -0.4,
            0.2,
            0.1,
            0.30,
            -0.08,
            0.12,
            0.18,
            -0.12,
            0.08,
            0.82,
            0.48,
            0.33,
        ]
    )
    truth = simulate_rigid_body_truth(initial_truth, times, inertia_matrix)
    measurements, observations = _rigid_body_observations(
        truth,
        measurement_std=0.035,
        rng=rng,
    )

    def transition(state, step_s):
        return propagate_rigid_body_state(state, step_s, inertia_matrix)

    model = StateTransitionModel(
        transition,
        np.diag(
            [
                *([2e-5] * 3),
                *([2e-5] * 3),
                *([5e-5] * 3),
                *([8e-5] * 3),
            ]
        ),
    )
    initial_offset = np.array(
        [
            0.12,
            -0.08,
            0.07,
            -0.08,
            0.05,
            -0.04,
            0.09,
            -0.06,
            0.07,
            -0.12,
            0.10,
            -0.08,
        ]
    )
    initial_covariance = np.diag(
        [
            *([0.12**2] * 3),
            *([0.10**2] * 3),
            *([0.12**2] * 3),
            *([0.14**2] * 3),
        ]
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_truth + initial_offset, initial_covariance),
    )
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=False,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=False,
    )
    return _collect_rigid_body_result(
        times,
        truth,
        measurements,
        filter_states,
        smoother_states,
        inertia_matrix,
    )


def run_inertia_estimation_rigid_body(seed=302, num_steps=100, delta_t_s=0.04):
    rng = np.random.default_rng(seed)
    times = np.arange(num_steps + 1, dtype=float) * delta_t_s
    inertia_matrix = np.diag([1.0, 1.4, 1.8])
    true_log_ratios = np.log(np.diag(inertia_matrix)[1:])
    initial_angular_velocity = np.array([0.82, 0.48, 0.33])
    # This buys enough chart range for a longer run without crossing the pi cut.
    initial_rotation_vector = (
        -1.2 * initial_angular_velocity / np.linalg.norm(initial_angular_velocity)
    )
    initial_truth = np.concatenate(
        (
            np.array([-0.4, 0.2, 0.1, 0.30, -0.08, 0.12]),
            initial_rotation_vector,
            initial_angular_velocity,
        )
    )
    truth = simulate_rigid_body_truth(initial_truth, times, inertia_matrix)
    measurements, _ = _rigid_body_observations(
        truth,
        measurement_std=0.025,
        rng=rng,
    )

    def transition(augmented_state, step_s):
        estimated_inertia = inertia_from_log_ratios(augmented_state[12:14])
        rigid_state = propagate_rigid_body_state(
            augmented_state[:12],
            step_s,
            estimated_inertia,
        )
        return np.concatenate((rigid_state, augmented_state[12:14]))

    def observe_augmented_state(augmented_state):
        return transform_canonical_points(augmented_state[:12])

    parameter_measurements = [
        Observation(
            measurement,
            0.025**2 * np.eye(CANONICAL_POINTS.size),
            observe_augmented_state,
        )
        for measurement in measurements
    ]
    model = StateTransitionModel(
        transition,
        np.diag(
            [
                *([1e-5] * 3),
                *([1e-5] * 3),
                *([3e-5] * 3),
                *([5e-5] * 3),
                2e-7,
                2e-7,
            ]
        ),
    )
    initial_rigid_offset = np.array(
        [
            0.08,
            -0.06,
            0.05,
            -0.05,
            0.04,
            -0.03,
            0.06,
            -0.04,
            0.05,
            -0.08,
            0.07,
            -0.06,
        ]
    )
    initial_log_ratios = np.log([1.18, 2.05])
    initial_mean = np.concatenate(
        (initial_truth + initial_rigid_offset, initial_log_ratios)
    )
    initial_covariance = np.diag(
        [
            *([0.10**2] * 3),
            *([0.08**2] * 3),
            *([0.09**2] * 3),
            *([0.11**2] * 3),
            0.16**2,
            0.16**2,
        ]
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states = bayes_filter.run_synchronous(
        parameter_measurements,
        times,
        use_jacobian=False,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=False,
    )
    result = _collect_rigid_body_result(
        times,
        truth,
        measurements,
        filter_states,
        smoother_states,
        inertia_matrix,
    )
    result.update(
        {
            "true_inertia_ratios": np.exp(true_log_ratios),
            "filtered_inertia_ratios": np.exp(result["filtered"][:, 12:14]),
            "smoothed_inertia_ratios": np.exp(result["smoothed"][:, 12:14]),
        }
    )
    return result

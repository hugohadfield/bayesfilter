import numpy as np

from examples.rigid_body import (
    CANONICAL_POINTS,
    angular_momentum_magnitude,
    compose_rotation_vectors,
    inertia_from_log_ratios,
    rigid_body_energy,
    rotation_distance,
    run_inertia_estimation_rigid_body,
    run_known_inertia_rigid_body,
    simulate_rigid_body_truth,
    so3_exp,
    so3_log,
    transform_canonical_points,
)


def test_so3_exp_and_log_round_trip_rotation_bivectors():
    rotation_vectors = [
        np.zeros(3),
        np.array([1e-9, -2e-9, 3e-9]),
        np.array([0.4, -0.2, 0.7]),
        np.array([-1.1, 0.6, 0.3]),
    ]
    for rotation_vector in rotation_vectors:
        rotation = so3_exp(rotation_vector)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            so3_log(rotation),
            rotation_vector,
            atol=1e-9,
        )


def test_so3_log_is_stable_near_pi_branch_boundary():
    axis = np.array([0.3, -0.4, 0.5])
    axis /= np.linalg.norm(axis)
    for distance_from_pi in (1e-3, 1e-6, 1e-8, 0.0):
        rotation = so3_exp(axis * (np.pi - distance_from_pi))
        reconstructed = so3_exp(so3_log(rotation))
        np.testing.assert_allclose(reconstructed, rotation, atol=5e-8)


def test_rotation_vector_composition_matches_matrix_composition():
    first = np.array([0.3, -0.2, 0.1])
    increment = np.array([-0.1, 0.25, 0.05])
    composed = compose_rotation_vectors(first, increment)
    np.testing.assert_allclose(
        so3_exp(composed),
        so3_exp(first) @ so3_exp(increment),
        atol=1e-12,
    )


def test_canonical_point_observation_contains_position_and_rotation_columns():
    position = np.array([1.2, -0.4, 0.8])
    rotation_vector = np.array([0.35, -0.12, 0.28])
    state = np.concatenate((position, np.zeros(3), rotation_vector, np.zeros(3)))
    world_points = transform_canonical_points(state).reshape(-1, 3)
    rotation = so3_exp(rotation_vector)

    np.testing.assert_allclose(world_points[0], position)
    np.testing.assert_allclose(
        (world_points[1:] - position).T,
        rotation @ CANONICAL_POINTS[1:].T,
    )


def test_torque_free_integrator_conserves_energy_and_momentum():
    inertia = np.diag([1.0, 1.4, 1.8])
    initial_state = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.2,
            -0.1,
            0.05,
            0.1,
            -0.1,
            0.05,
            0.82,
            0.48,
            0.33,
        ]
    )
    times = np.linspace(0.0, 3.0, 61)
    truth = simulate_rigid_body_truth(initial_state, times, inertia)
    energy = np.array([rigid_body_energy(state[9:12], inertia) for state in truth])
    momentum = np.array(
        [angular_momentum_magnitude(state[9:12], inertia) for state in truth]
    )

    assert np.ptp(energy) / energy[0] < 1e-8
    assert np.ptp(momentum) / momentum[0] < 1e-8
    np.testing.assert_allclose(
        truth[:, 0:3],
        initial_state[0:3] + times[:, None] * initial_state[3:6],
    )


def test_known_inertia_example_tracks_full_pose_and_stays_in_log_chart():
    result = run_known_inertia_rigid_body()

    assert result["truth"].shape == (46, 12)
    assert result["measurements"].shape == (45, 12)
    assert np.mean(result["smoothed_rotation_error"]) < np.mean(
        result["filtered_rotation_error"]
    )
    assert np.max(np.linalg.norm(result["truth"][:, 6:9], axis=1)) < 2.8
    for covariance in result["smoothed_covariances"]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8


def test_inertia_estimation_converges_and_contracts_uncertainty():
    result = run_inertia_estimation_rigid_body()
    true_ratios = result["true_inertia_ratios"]
    initial_ratios = result["filtered_inertia_ratios"][0]
    final_ratios = result["filtered_inertia_ratios"][-1]
    initial_error = np.linalg.norm(initial_ratios - true_ratios)
    final_error = np.linalg.norm(final_ratios - true_ratios)
    initial_parameter_std = np.sqrt(np.diag(result["filtered_covariances"][0])[12:14])
    final_parameter_std = np.sqrt(np.diag(result["filtered_covariances"][-1])[12:14])

    np.testing.assert_allclose(
        np.diag(inertia_from_log_ratios(np.log(true_ratios))),
        [1.0, 1.4, 1.8],
    )
    assert final_error < 0.35 * initial_error
    assert np.all(final_parameter_std < 0.75 * initial_parameter_std)
    assert np.mean(result["smoothed_rotation_error"]) < np.mean(
        result["filtered_rotation_error"]
    )
    assert np.max(np.linalg.norm(result["truth"][:, 6:9], axis=1)) < 2.6
    assert (
        rotation_distance(
            result["truth"][-1, 6:9],
            result["smoothed"][-1, 6:9],
        )
        < 0.05
    )

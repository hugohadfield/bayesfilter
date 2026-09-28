import numpy as np

from examples.gravity_vector_from_imu import (
    GRAVITY_M_S2,
    gravity_angle_error_deg,
    observe_raw_imu,
    propagate_gravity_state,
    raw_imu_observation_jacobian,
    run_gravity_vector_from_imu,
)


def test_gravity_rotates_opposite_body_rotation():
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0])

    propagated = propagate_gravity_state(state, 1.0)

    np.testing.assert_allclose(
        propagated[:3],
        np.array([0.0, -1.0, 0.0]),
        atol=1e-12,
    )


def test_raw_imu_observation_is_gyro_and_negative_gravity():
    state = np.array(
        [1.0, -2.0, 9.4, 0.1, -0.2, 0.3]
    )

    observation = observe_raw_imu(state)

    np.testing.assert_allclose(
        observation,
        np.array([0.1, -0.2, 0.3, -1.0, 2.0, -9.4]),
    )

    expected_jacobian = np.block(
        [
            [np.zeros((3, 3)), np.eye(3)],
            [-np.eye(3), np.zeros((3, 3))],
        ]
    )
    np.testing.assert_allclose(
        raw_imu_observation_jacobian(state),
        expected_jacobian,
    )


def test_gravity_angle_error_is_scale_invariant():
    truth = np.array([[0.0, 0.0, GRAVITY_M_S2]])
    estimate = np.array([[0.0, 0.0, 2.0 * GRAVITY_M_S2]])

    np.testing.assert_allclose(
        gravity_angle_error_deg(estimate, truth),
        np.array([0.0]),
        atol=1e-10,
    )


def test_gravity_filter_rejects_linear_acceleration_better_than_naive_accel():
    result = run_gravity_vector_from_imu()

    assert result["filtered_angle_rmse_deg"] < 2.0
    assert result["naive_angle_rmse_deg"] > 2.5
    assert (
        result["filtered_angle_rmse_deg"]
        < 0.6 * result["naive_angle_rmse_deg"]
    )

    assert result["filtered_acceleration_burst_rmse_deg"] < 2.5
    assert result["naive_acceleration_burst_rmse_deg"] > 4.0
    assert (
        result["filtered_acceleration_burst_rmse_deg"]
        < 0.5 * result["naive_acceleration_burst_rmse_deg"]
    )

    assert result["final_angle_error_deg"] < 2.0
    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["filtered_covariances"]).all()

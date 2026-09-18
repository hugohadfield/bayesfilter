import numpy as np
import pytest

from examples.imu_lever_arm_calibration import (
    LEVER_Z,
    RICH_EXCITATION_START_S,
    TRUE_ACCEL_BIAS_M_S2,
    TRUE_LEVER_ARM_M,
    acceleration_design_matrix,
    angular_acceleration_profile,
    angular_velocity_profile,
    estimate_angular_kinematics,
    make_accelerometer_model,
    rotational_acceleration,
    run_imu_lever_arm_calibration,
    skew,
)


def test_skew_matches_cross_product():
    a = np.array([0.3, -0.8, 1.2])
    b = np.array([-0.4, 0.7, 0.1])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_rotational_acceleration_matches_design_matrix():
    omega = np.array([0.6, -0.3, 1.1])
    alpha = np.array([0.2, 0.4, -0.1])
    lever = np.array([0.18, -0.11, 0.27])
    np.testing.assert_allclose(
        rotational_acceleration(omega, alpha, lever),
        acceleration_design_matrix(omega, alpha) @ lever,
    )


def test_pure_z_rotation_cannot_observe_z_lever_arm():
    times = np.linspace(0.5, RICH_EXCITATION_START_S - 0.5, 20)
    matrices = [
        acceleration_design_matrix(
            angular_velocity_profile(time_s),
            angular_acceleration_profile(time_s),
        )
        for time_s in times
    ]
    stacked = np.vstack(matrices)

    np.testing.assert_allclose(stacked[:, LEVER_Z], 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(stacked[:, :2]) == 2


def test_rich_excitation_observes_all_three_lever_dimensions():
    times = np.linspace(11.0, 18.0, 30)
    stacked = np.vstack(
        [
            acceleration_design_matrix(
                angular_velocity_profile(time_s),
                angular_acceleration_profile(time_s),
            )
            for time_s in times
        ]
    )
    assert np.linalg.matrix_rank(stacked) == 3


def test_causal_gyro_fit_recovers_angular_kinematics():
    gyro_rate_hz = 200.0
    gyro_times = np.arange(0.0, 15.0, 1.0 / gyro_rate_hz)
    gyro = angular_velocity_profile(gyro_times)
    query_times = np.arange(0.2, 15.0, 0.02)

    omega, alpha = estimate_angular_kinematics(
        gyro_times,
        gyro,
        query_times,
    )
    omega_truth = angular_velocity_profile(query_times)
    alpha_truth = angular_acceleration_profile(query_times)

    assert np.sqrt(np.mean((omega - omega_truth) ** 2)) < 2e-3
    assert np.sqrt(np.mean((alpha - alpha_truth) ** 2)) < 1e-2


def test_accelerometer_observation_model_includes_bias_and_exact_jacobian():
    omega = np.array([0.4, -0.7, 1.0])
    alpha = np.array([0.3, 0.2, -0.15])
    observe, jacobian, matrix = make_accelerometer_model(omega, alpha)
    state = np.r_[TRUE_LEVER_ARM_M, TRUE_ACCEL_BIAS_M_S2]

    np.testing.assert_allclose(observe(state), matrix @ state)
    np.testing.assert_allclose(jacobian(state), matrix)
    np.testing.assert_allclose(matrix[:, 3:], np.eye(3))


@pytest.mark.parametrize(
    "use_jacobian",
    [False, True],
    ids=["unscented", "jacobian"],
)
def test_example_recovers_imu_lever_arm_and_shows_observability(use_jacobian):
    result = run_imu_lever_arm_calibration(use_jacobian=use_jacobian)

    assert result["gyro_rate_hz"] == 200.0
    assert result["accel_rate_hz"] == 50.0

    assert result["pre_excitation_z_std_m"] > 0.25
    assert result["pre_excitation_z_error_m"] > 0.15
    assert result["full_excitation_z_std_m"] < 0.02

    assert result["final_lever_error_m"] < 0.01
    assert result["final_accel_bias_error_m_s2"] < 0.01

    assert result["smoothed_lever_rmse_m"] < result["filtered_lever_rmse_m"]
    assert result["smoothed_lever_rmse_m"] < 0.01
    assert result["initial_smoothed_lever_error_m"] < 0.01

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"][::100]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

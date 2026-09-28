import numpy as np

from examples.camera_gyro_calibration import (
    GYRO_BIAS,
    TIME_OFFSET,
    TRUE_GYRO_BIAS_RAD_S,
    TRUE_TIME_OFFSET_S,
    WEAK_EXCITATION_END_S,
    rig_angular_velocity_gyro_frame,
    rotation_vector_to_matrix,
    run_camera_gyro_calibration,
    simulate_camera_gyro_logs,
)


def test_rotation_vector_conversion_is_a_rotation_matrix():
    matrix = rotation_vector_to_matrix(np.array([0.2, -0.1, 0.3]))
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12)


def test_initial_motion_is_single_axis_then_becomes_three_axis():
    weak = rig_angular_velocity_gyro_frame(2.0)
    rich = rig_angular_velocity_gyro_frame(WEAK_EXCITATION_END_S + 1.0)
    assert np.allclose(weak[:2], 0.0)
    assert abs(weak[2]) > 0.05
    assert np.count_nonzero(np.abs(rich) > 0.05) == 3


def test_camera_and_gyro_logs_have_independent_rates():
    data = simulate_camera_gyro_logs()
    gyro_dt = np.median(np.diff(data["gyro_times_s"]))
    camera_dt = np.median(np.diff(data["camera_times_s"]))
    assert gyro_dt < camera_dt / 5.0
    assert data["gyro_measurements_rad_s"].shape[1] == 3
    assert data["visual_angular_velocity_rad_s"].shape[1] == 3


def test_filter_recovers_spatiotemporal_calibration_and_bias():
    result = run_camera_gyro_calibration()
    assert result["final_rotation_error_deg"] < 1.0
    assert abs(result["final_time_offset_s"] - TRUE_TIME_OFFSET_S) < 0.005
    assert np.max(
        np.abs(result["final_gyro_bias_rad_s"] - TRUE_GYRO_BIAS_RAD_S)
    ) < 0.003


def test_calibration_strongly_improves_cross_sensor_alignment():
    result = run_camera_gyro_calibration()
    assert (
        result["calibrated_angular_rate_rmse_rad_s"]
        < 0.25 * result["naive_angular_rate_rmse_rad_s"]
    )


def test_rich_three_axis_motion_resolves_weak_rotation_direction():
    result = run_camera_gyro_calibration()
    weak_index = result["weak_excitation_index"]
    weak_rotation_std = np.sqrt(
        np.diag(result["filtered_covariances"][weak_index])[:3]
    )
    final_rotation_std = np.sqrt(
        np.diag(result["filtered_covariances"][-1])[:3]
    )
    assert np.max(weak_rotation_std) > 5.0 * np.max(final_rotation_std)
    assert (
        np.sqrt(
            result["filtered_covariances"][-1, TIME_OFFSET, TIME_OFFSET]
        )
        < 0.001
    )
    assert np.max(
        np.sqrt(
            np.diag(result["filtered_covariances"][-1])[
                GYRO_BIAS.start : GYRO_BIAS.stop
            ]
        )
    ) < 0.002

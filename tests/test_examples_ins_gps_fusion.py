import numpy as np

from examples.ins_gps_fusion import (
    DEFAULT_INS_RATE_HZ,
    GPS_OUTAGE_S,
    drift_process_covariance,
    drift_transition,
    drift_transition_jacobian,
    run_ins_gps_fusion,
    simulate_ins_gps,
    yaw_rotation,
)


def test_drift_transition_matches_jacobian_for_linear_model():
    state = np.array(
        [
            0.4,
            -0.2,
            0.1,
            0.15,
            0.03,
            -0.02,
            0.01,
            0.005,
        ]
    )
    delta_t_s = 0.05

    expected = drift_transition_jacobian(
        state,
        delta_t_s,
    ) @ state
    actual = drift_transition(state, delta_t_s)

    np.testing.assert_allclose(actual, expected)


def test_process_covariance_is_symmetric_positive_semidefinite():
    covariance = drift_process_covariance(
        1.0 / DEFAULT_INS_RATE_HZ
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-14,
    )
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-14


def test_simulation_has_full_pose_ins_and_long_gps_outage():
    data = simulate_ins_gps()

    assert data["ins_local_positions_m"].shape[1] == 3
    assert data["ins_local_rotation_vectors"].shape[1] == 3

    outage_start, outage_end = GPS_OUTAGE_S
    assert not np.any(
        (data["gps_times"] >= outage_start)
        & (data["gps_times"] <= outage_end)
    )


def test_yaw_rotation_is_a_rotation_matrix():
    rotation = yaw_rotation(0.7)

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-12,
    )
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_gps_fusion_corrects_ins_frame_wander():
    result = run_ins_gps_fusion()

    assert result["dead_reckoning_position_rmse_m"] > 10.0
    assert result["filtered_position_rmse_m"] < 1.0
    assert result["smoothed_position_rmse_m"] < 0.5

    assert (
        result["filtered_position_rmse_m"]
        < 0.10 * result["dead_reckoning_position_rmse_m"]
    )


def test_rts_reconstructs_the_gps_outage():
    result = run_ins_gps_fusion()

    assert result["filtered_outage_position_rmse_m"] < 1.6
    assert result["smoothed_outage_position_rmse_m"] < 0.7
    assert (
        result["smoothed_outage_position_rmse_m"]
        < 0.60 * result["filtered_outage_position_rmse_m"]
    )


def test_fused_orientation_is_finite():
    result = run_ins_gps_fusion()

    assert np.isfinite(result["filtered_rotation_error_rad"]).all()
    assert np.isfinite(result["smoothed_rotation_error_rad"]).all()
    assert np.rad2deg(
        np.mean(result["smoothed_rotation_error_rad"])
    ) < 1.0

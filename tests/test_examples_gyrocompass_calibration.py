import numpy as np

from examples.gyrocompass_calibration import (
    ACCEL_BIAS_STABILITY_UG,
    EARTH_RATE_DEG_HR,
    GYRO_BIAS_STABILITY_DEG_HR,
    POSE_ROTATIONS,
    TRUE_GYRO_BIAS_DEG_HR,
    TRUE_HEADING_DEG,
    TRUE_LATITUDE_DEG,
    accel_dwell_white_std_g,
    calibration_observation_matrix,
    derive_latitude_heading,
    gyro_dwell_white_std_deg_hr,
    make_calibration_observation_model,
    run_gyrocompass_calibration,
    stacked_observation_rank,
    true_inertial_vectors,
)


def test_true_inertial_vectors_encode_latitude_and_heading():
    gravity, earth_rate = true_inertial_vectors()
    state = np.zeros(12)
    state[:3] = gravity
    state[3:6] = earth_rate

    latitude_deg, heading_deg = derive_latitude_heading(state)

    np.testing.assert_allclose(np.linalg.norm(gravity), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(earth_rate),
        EARTH_RATE_DEG_HR,
        atol=1e-10,
    )
    np.testing.assert_allclose(latitude_deg, TRUE_LATITUDE_DEG, atol=1e-10)
    np.testing.assert_allclose(heading_deg, TRUE_HEADING_DEG, atol=1e-10)


def test_dwell_averaging_uses_realistic_high_grade_mems_noise():
    # ADIS16495-class white noise averaged for the default 90 second dwell.
    np.testing.assert_allclose(
        gyro_dwell_white_std_deg_hr(),
        0.5366563146,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        1e6 * accel_dwell_white_std_g(),
        1.192569588,
        rtol=1e-8,
    )

    assert GYRO_BIAS_STABILITY_DEG_HR == 0.8
    assert ACCEL_BIAS_STABILITY_UG == 3.2

    # The unknown turn-on bias is deliberately harder than Earth rate itself.
    assert np.linalg.norm(TRUE_GYRO_BIAS_DEG_HR) > 2.0 * EARTH_RATE_DEG_HR


def test_same_axis_placements_are_rank_deficient_until_axis_changes():
    assert stacked_observation_rank(1) == 6
    assert stacked_observation_rank(2) == 10
    assert stacked_observation_rank(3) == 10
    assert stacked_observation_rank(4) == 12


def test_observation_closure_matches_exact_linear_matrix():
    pose_rotation = POSE_ROTATIONS[3]
    matrix = calibration_observation_matrix(pose_rotation)
    observe, jacobian = make_calibration_observation_model(pose_rotation)

    state = np.array(
        [
            0.15,
            -0.12,
            0.98,
            4.0,
            -9.0,
            -11.0,
            25.0,
            -18.0,
            12.0,
            4e-4,
            -3e-4,
            8e-4,
        ]
    )

    np.testing.assert_allclose(observe(state), matrix @ state)
    np.testing.assert_allclose(jacobian(state), matrix)


def test_gyrocompass_calibrates_bias_and_finds_latitude_and_north():
    result = run_gyrocompass_calibration(use_jacobian=True)

    # Three poses about one fixture axis still leave two state directions
    # unobservable, despite 4.5 minutes of stationary data.
    assert result["same_axis_rank"] == 10
    assert abs(result["same_axis_latitude_error_deg"]) > 15.0
    assert abs(result["same_axis_heading_error_deg"]) > 20.0

    # One placement about a different axis makes the linear calibration full
    # rank and causes the navigation estimate to snap into place.
    assert result["cross_axis_rank"] == 12
    assert abs(result["cross_axis_latitude_error_deg"]) < 2.0
    assert abs(result["cross_axis_heading_error_deg"]) < 4.0

    # After the full multi-position session, a realistic tactical-grade MEMS
    # noise level is sufficient for useful gyrocompassing.
    assert abs(result["final_latitude_error_deg"]) < 2.0
    assert abs(result["final_heading_error_deg"]) < 3.0
    assert result["final_gyro_bias_error_deg_hr"] < 1.5
    assert result["final_accel_bias_error_ug"] < 10.0

    assert result["filtered_navigation_std_deg"][-1, 0] < 2.0
    assert result["filtered_navigation_std_deg"][-1, 1] < 3.0

    # RTS can use all later placements to reconstruct the inertial vectors at
    # the beginning of the calibration session.
    assert abs(result["initial_smoothed_latitude_error_deg"]) < 2.0
    assert abs(result["initial_smoothed_heading_error_deg"]) < 3.0

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

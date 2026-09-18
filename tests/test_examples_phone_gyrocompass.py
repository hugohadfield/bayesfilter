import numpy as np

from examples.phone_gyrocompass import (
    EARTH_RATE_DEG_HR,
    TRUE_DECLINATION_DEG,
    TRUE_LATITUDE_DEG,
    derive_latitude_declination,
    estimate_pose_from_gravity_and_magnetometer,
    human_pose_rotations,
    magnetic_frame_to_phone_rotation,
    phone_bias_instability_deg_hr,
    phone_gyro_dwell_std_deg_hr,
    run_phone_gyrocompass,
    stacked_observation_rank,
    static_pose_rotations,
    true_earth_rate_magnetic_frame,
)


def test_true_earth_rate_encodes_latitude_and_declination():
    earth_rate = true_earth_rate_magnetic_frame()
    latitude, declination = derive_latitude_declination(earth_rate)

    np.testing.assert_allclose(
        np.linalg.norm(earth_rate),
        EARTH_RATE_DEG_HR,
        atol=1e-12,
    )
    np.testing.assert_allclose(latitude, TRUE_LATITUDE_DEG, atol=1e-12)
    np.testing.assert_allclose(
        declination,
        TRUE_DECLINATION_DEG,
        atol=1e-12,
    )


def test_pixel_class_phone_noise_numbers_are_in_expected_regime():
    np.testing.assert_allclose(
        phone_gyro_dwell_std_deg_hr(60.0),
        3.580959038,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        phone_bias_instability_deg_hr(),
        1.973335401,
        rtol=1e-8,
    )
    assert phone_gyro_dwell_std_deg_hr(60.0) < EARTH_RATE_DEG_HR


def test_gravity_and_magnetometer_recover_pose_without_known_hand_rotation():
    rotation = magnetic_frame_to_phone_rotation(
        np.deg2rad(37.0),
        np.deg2rad(-24.0),
        np.deg2rad(113.0),
    )
    inclination = np.deg2rad(65.0)
    gravity = rotation @ np.array([0.0, 0.0, 1.0])
    magnetic_field = rotation @ np.array(
        [np.cos(inclination), 0.0, np.sin(inclination)]
    )

    estimated = estimate_pose_from_gravity_and_magnetometer(
        gravity,
        magnetic_field,
    )
    np.testing.assert_allclose(estimated, rotation, atol=1e-12)


def test_static_phone_is_rank_deficient_but_varied_hand_poses_are_full_rank():
    static = static_pose_rotations(24)
    human = human_pose_rotations(seed=11, num_poses=24)

    assert stacked_observation_rank(static) == 3
    assert stacked_observation_rank(human) == 6


def test_phone_gyrocompass_recovers_latitude_declination_and_bias():
    result = run_phone_gyrocompass(seed=11, use_jacobian=True)
    human = result["human"]
    static = result["static"]

    # Repeated observations in one pose cannot distinguish Earth rate from a
    # fixed phone-frame gyro bias.
    assert static["geometry_rank"] == 3
    assert abs(static["final_latitude_error_deg"]) > 20.0
    assert static["filtered_navigation_std_deg"][-1, 0] > 20.0

    # Arbitrary human reorientation creates the missing geometry. No known
    # rotation rate or prescribed stop angle is used.
    assert human["geometry_rank"] == 6
    assert abs(human["final_latitude_error_deg"]) < 5.0
    assert abs(human["final_declination_error_deg"]) < 5.0
    assert human["final_gyro_bias_error_deg_hr"] < 3.0

    assert human["filtered_navigation_std_deg"][-1, 0] < 6.0
    assert human["filtered_navigation_std_deg"][-1, 1] < 7.0

    assert np.isfinite(human["filtered"]).all()
    assert np.isfinite(human["smoothed"]).all()
    for covariance in human["smoothed_covariances"][::6]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

import numpy as np

from examples.radar_camera_fusion import (
    CAMERA_DROPOUT_S,
    CAMERA_POSITION_M,
    RADAR_DROPOUT_S,
    RADAR_POSITION_M,
    camera_bearing_jacobian,
    observe_camera_bearing,
    observe_radar_range_rate,
    radar_range_rate_jacobian,
    run_radar_camera_fusion,
    simulate_radar_camera_truth,
)


def test_camera_bearing_jacobian_matches_finite_difference():
    state = np.array([4.0, 3.0, 1.2, -0.4])
    analytic = camera_bearing_jacobian(state)

    epsilon = 1e-6
    numerical = np.empty_like(analytic)
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = epsilon
        numerical[:, index] = (
            observe_camera_bearing(state + offset)
            - observe_camera_bearing(state - offset)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic,
        numerical,
        atol=2e-7,
        rtol=2e-6,
    )


def test_radar_range_rate_jacobian_matches_finite_difference():
    state = np.array([5.0, 7.0, 1.4, 0.8])
    analytic = radar_range_rate_jacobian(state)

    epsilon = 1e-6
    numerical = np.empty_like(analytic)
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = epsilon
        numerical[:, index] = (
            observe_radar_range_rate(state + offset)
            - observe_radar_range_rate(state - offset)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic,
        numerical,
        atol=3e-7,
        rtol=3e-6,
    )


def test_sensor_geometry_is_complementary():
    state = np.array([8.0, 6.0, 1.5, 0.7])

    camera_jacobian = camera_bearing_jacobian(state)
    radar_jacobian = radar_range_rate_jacobian(state)

    # Camera bearing has no instantaneous velocity sensitivity.
    np.testing.assert_allclose(
        camera_jacobian[:, 2:4],
        np.zeros((2, 2)),
    )

    # Radar directly measures radial velocity.
    assert np.linalg.norm(radar_jacobian[1, 2:4]) > 0.9

    # Neither sensor is colocated with the other, so their geometry differs.
    assert not np.allclose(CAMERA_POSITION_M, RADAR_POSITION_M)


def test_truth_contains_real_maneuvering():
    times = np.linspace(0.0, 20.0, 401)
    truth = simulate_radar_camera_truth(times)

    velocity_change = np.linalg.norm(
        truth[-1, 2:4] - truth[0, 2:4]
    )
    assert velocity_change > 1.0


def test_radar_camera_fusion_beats_single_sensor_baselines():
    result = run_radar_camera_fusion()

    assert result["filtered_position_rmse_m"] < 0.8
    assert result["smoothed_position_rmse_m"] < 0.4
    assert (
        result["smoothed_position_rmse_m"]
        < result["filtered_position_rmse_m"]
    )

    assert result["filtered_velocity_rmse_m_s"] < 0.6
    assert result["smoothed_velocity_rmse_m_s"] < 0.4

    assert (
        result["filtered_position_rmse_m"]
        < 0.25 * result["camera_only_position_rmse_m"]
    )
    assert (
        result["filtered_position_rmse_m"]
        < 0.25 * result["radar_only_position_rmse_m"]
    )

    assert result["camera_only_position_rmse_m"] > 2.0
    assert result["radar_only_position_rmse_m"] > 2.0

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.all(result["fused_position_std_m"] >= 0.0)


def test_both_sensor_dropout_intervals_are_represented():
    result = run_radar_camera_fusion()

    camera_times = result["camera_times"]
    radar_times = result["radar_times"]

    assert not np.any(
        (camera_times >= CAMERA_DROPOUT_S[0])
        & (camera_times <= CAMERA_DROPOUT_S[1])
    )
    assert not np.any(
        (radar_times >= RADAR_DROPOUT_S[0])
        & (radar_times <= RADAR_DROPOUT_S[1])
    )

    assert np.any(camera_times < CAMERA_DROPOUT_S[0])
    assert np.any(camera_times > CAMERA_DROPOUT_S[1])
    assert np.any(radar_times < RADAR_DROPOUT_S[0])
    assert np.any(radar_times > RADAR_DROPOUT_S[1])

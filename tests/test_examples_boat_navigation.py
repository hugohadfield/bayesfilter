import numpy as np

from examples.boat_navigation import (
    CLOUD_END_S,
    CLOUD_START_S,
    LANDMARK_BEARING_STD_RAD,
    LANDMARK_INTERVAL_S,
    MISSED_LANDMARK_SIGHTINGS,
    RANGE_STD_FLOOR_M,
    RANGE_STD_FRACTION,
    SPEED_INTERVAL_S,
    SPEED_STD_MPS,
    SUN_INTERVAL_S,
    SUN_STD_RAD,
    TIMING_JITTER,
    TRUE_CURRENT_MPS,
    run_boat_navigation,
)


def test_boat_navigation_sensor_schedule_is_sparse_and_jittered():
    result = run_boat_navigation(seed=1201)

    speed_intervals = np.diff(np.r_[0.0, result["speed_times"]])
    assert np.all(speed_intervals >= SPEED_INTERVAL_S * (1.0 - TIMING_JITTER))
    assert np.all(speed_intervals <= SPEED_INTERVAL_S * (1.0 + TIMING_JITTER))

    sun_intervals = np.diff(np.r_[0.0, result["sun_times_all"]])
    assert np.all(sun_intervals >= SUN_INTERVAL_S * (1.0 - TIMING_JITTER))
    assert np.all(sun_intervals <= SUN_INTERVAL_S * (1.0 + TIMING_JITTER))
    assert np.all(
        (result["sun_times"] < CLOUD_START_S)
        | (result["sun_times"] > CLOUD_END_S)
    )

    landmark_intervals = np.diff(np.r_[0.0, result["landmark_schedule"]])
    assert np.all(landmark_intervals >= LANDMARK_INTERVAL_S * (1.0 - TIMING_JITTER))
    assert np.all(landmark_intervals <= LANDMARK_INTERVAL_S * (1.0 + TIMING_JITTER))
    assert len(result["landmark_records"]) == (
        len(result["landmark_schedule"]) - len(MISSED_LANDMARK_SIGHTINGS)
    )


def test_boat_navigation_uses_deliberately_rough_human_measurements():
    assert SPEED_STD_MPS == 0.60
    assert np.isclose(np.rad2deg(LANDMARK_BEARING_STD_RAD), 20.0)
    assert np.isclose(np.rad2deg(SUN_STD_RAD), 25.0)
    assert RANGE_STD_FLOOR_M == 500.0
    assert RANGE_STD_FRACTION == 0.40


def test_sun_measurements_improve_global_orientation_and_position():
    result = run_boat_navigation(seed=1201)
    metrics = result["metrics"]

    assert metrics["sun_rts"]["heading_rmse_deg"] < metrics["no_sun_rts"]["heading_rmse_deg"]
    assert metrics["sun_rts"]["position_rmse_m"] < metrics["no_sun_rts"]["position_rmse_m"]


def test_round_island_voyage_makes_current_estimable():
    result = run_boat_navigation(seed=1201)
    final_current = result["smoothed"][-1, 5:7]

    assert np.linalg.norm(final_current - TRUE_CURRENT_MPS) < 0.12
    assert result["metrics"]["sun_rts"]["current_rmse_mps"] < 0.12


def test_no_sun_rts_is_returned_on_same_timeline():
    result = run_boat_navigation(seed=1201)

    assert result["filtered"].shape == result["smoothed"].shape
    assert result["filtered"].shape == result["no_sun_filtered"].shape
    assert result["filtered"].shape == result["no_sun_smoothed"].shape
    assert len(result["times"]) == len(result["filtered"])

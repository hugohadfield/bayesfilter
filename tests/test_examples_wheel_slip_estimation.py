import numpy as np

from examples.wheel_slip_estimation import (
    KNOWN_WHEEL_RADIUS_M,
    SLIP_LOGIT,
    SPEED,
    TERRAIN_SEGMENTS,
    logit,
    observe_wheel_rate,
    propagate_slip_state,
    run_wheel_slip_estimation,
    sigmoid,
)


def test_slip_state_transition_matches_constant_acceleration():
    state = np.array([10.0, 12.0, 0.4, logit(0.10)])
    propagated = propagate_slip_state(state, 2.5)
    assert np.isclose(propagated[0], 10.0 + 12.0 * 2.5 + 0.5 * 0.4 * 2.5**2)
    assert np.isclose(propagated[1], 13.0)
    assert np.isclose(propagated[2], 0.4)
    assert np.isclose(propagated[3], state[3])


def test_wheel_observation_matches_slip_definition():
    state = np.array([0.0, 15.0, 0.0, logit(0.20)])
    expected = 15.0 / (KNOWN_WHEEL_RADIUS_M * 0.80)
    assert np.isclose(observe_wheel_rate(state)[0], expected)
    assert np.isclose(sigmoid(state[SLIP_LOGIT]), 0.20)


def test_filter_recovers_surface_dependent_slip():
    result = run_wheel_slip_estimation()
    summaries = result["terrain_summary"]
    by_name = {}
    for summary in summaries:
        by_name.setdefault(summary["terrain"], []).append(summary)

    assert by_name["asphalt"][0]["estimated_mean_slip"] < 0.035
    assert 0.045 < by_name["gravel"][0]["estimated_mean_slip"] < 0.11
    assert 0.18 < by_name["sand"][0]["estimated_mean_slip"] < 0.30
    assert 0.09 < by_name["wet grass"][0]["estimated_mean_slip"] < 0.19
    assert by_name["asphalt"][-1]["estimated_mean_slip"] < 0.045


def test_rts_improves_slip_reconstruction():
    result = run_wheel_slip_estimation()
    assert result["smoothed_slip_rmse"] < result["filtered_slip_rmse"]
    assert result["filtered_slip_rmse"] < 0.025
    assert result["smoothed_slip_rmse"] < 0.015


def test_sensor_schedule_is_sparse_gps_and_dense_wheel_rate():
    result = run_wheel_slip_estimation()
    assert len(result["wheel_times"]) > 10 * len(result["gps_times"])
    assert np.all(np.diff(result["wheel_times"]) > 0.0)
    assert np.all(np.diff(result["gps_times"]) > 0.0)


def test_drive_covers_all_configured_terrain_segments():
    result = run_wheel_slip_estimation()
    names = [summary["terrain"] for summary in result["terrain_summary"]]
    configured_names = [segment[2] for segment in TERRAIN_SEGMENTS]
    assert names == configured_names

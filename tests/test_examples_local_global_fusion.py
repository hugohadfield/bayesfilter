import numpy as np
import pytest

from examples.local_global_fusion import (
    DEFAULT_FILTER_RATE_HZ,
    DEFAULT_GLOBAL_RATE_HZ,
    DEFAULT_SPEED_RATE_HZ,
    GLOBAL_OUTAGE_END_S,
    GLOBAL_OUTAGE_START_S,
    GYRO_BIAS,
    TRUE_GYRO_BIAS_RAD_S,
    global_position_jacobian,
    gyro_jacobian,
    local_global_jacobian,
    observe_global_position,
    observe_gyro,
    observe_speed,
    propagate_local_global_state,
    run_local_global_fusion,
    speed_jacobian,
)


def _finite_difference_jacobian(function, state, step=1e-6):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append(
            (function(state + offset) - function(state - offset)) / (2 * step)
        )
    return np.column_stack(columns)


@pytest.fixture(scope="module")
def result():
    return run_local_global_fusion()


def test_transition_jacobian_matches_finite_difference():
    state = np.array([2.0, -1.0, 0.7, 4.2, -0.08, 0.012])
    delta_t_s = 0.02
    numerical = _finite_difference_jacobian(
        lambda value: propagate_local_global_state(value, delta_t_s),
        state,
    )
    np.testing.assert_allclose(
        local_global_jacobian(state, delta_t_s),
        numerical,
        atol=1e-8,
    )


def test_observation_jacobians_match_finite_difference():
    state = np.array([2.0, -1.0, 0.7, 4.2, -0.08, 0.012])
    for observation, jacobian in (
        (observe_speed, speed_jacobian),
        (observe_gyro, gyro_jacobian),
        (observe_global_position, global_position_jacobian),
    ):
        np.testing.assert_allclose(
            jacobian(state),
            _finite_difference_jacobian(observation, state),
            atol=1e-10,
        )


def test_sensor_rates_and_global_outage(result):
    np.testing.assert_allclose(
        np.diff(result["gyro_times"]),
        1.0 / DEFAULT_FILTER_RATE_HZ,
    )
    np.testing.assert_allclose(
        np.diff(result["speed_times"]),
        1.0 / DEFAULT_SPEED_RATE_HZ,
    )
    global_times = result["global_times"]
    assert not np.any(
        (global_times >= GLOBAL_OUTAGE_START_S)
        & (global_times <= GLOBAL_OUTAGE_END_S)
    )
    before = global_times[global_times < GLOBAL_OUTAGE_START_S]
    after = global_times[global_times > GLOBAL_OUTAGE_END_S]
    np.testing.assert_allclose(np.diff(before), 1.0 / DEFAULT_GLOBAL_RATE_HZ)
    np.testing.assert_allclose(np.diff(after), 1.0 / DEFAULT_GLOBAL_RATE_HZ)


def test_fusion_beats_raw_global_positions_and_local_dead_reckoning(result):
    assert result["filtered_position_rmse"] < result["global_position_rmse"]
    assert result["filtered_position_rmse"] < 0.05 * result[
        "dead_reckoning_position_rmse"
    ]
    assert result["filtered_position_rmse"] < 3.0
    assert result["outage_max_position_error_m"] < 5.0


def test_rts_improves_position_and_heading(result):
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_heading_rmse"] < result["filtered_heading_rmse"]
    assert result["smoothed_position_rmse"] < 1.5
    assert result["smoothed_heading_rmse"] < np.deg2rad(1.0)


def test_global_fixes_identify_gyro_bias_and_reanchor_uncertainty(result):
    assert abs(result["final_gyro_bias_rad_s"] - TRUE_GYRO_BIAS_RAD_S) < 0.002

    before_index = np.argmin(
        np.abs(result["times"] - (GLOBAL_OUTAGE_START_S - 1.0))
    )
    end_index = np.argmin(np.abs(result["times"] - GLOBAL_OUTAGE_END_S))
    recovered_index = np.argmin(
        np.abs(result["times"] - (GLOBAL_OUTAGE_END_S + 5.0))
    )
    trace = result["position_covariance_trace"]
    assert trace[end_index] > 10.0 * trace[before_index]
    assert trace[recovered_index] < 0.15 * trace[end_index]
    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.isfinite(result["filtered"][:, GYRO_BIAS]).all()

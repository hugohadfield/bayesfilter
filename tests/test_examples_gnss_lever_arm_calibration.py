import numpy as np
import pytest

from examples.gnss_lever_arm_calibration import (
    FIRST_TURN_START_S,
    HEADING,
    LEVER_X,
    LEVER_Y,
    SPEED,
    TRUE_LEVER_ARM_M,
    YAW_RATE,
    gnss_antenna_position_jacobian,
    observe_gnss_antenna_position,
    propagate_vehicle_state,
    run_gnss_lever_arm_calibration,
    simulate_vehicle_truth,
    vehicle_transition_jacobian,
)


def _finite_difference_jacobian(function, state, step=1e-6):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append(
            (function(state + offset) - function(state - offset)) / (2.0 * step)
        )
    return np.column_stack(columns)


def test_gnss_observes_rotated_antenna_offset():
    state = np.array([10.0, -3.0, np.pi / 2.0, 4.0, 0.0, 2.0, 0.5])
    np.testing.assert_allclose(
        observe_gnss_antenna_position(state),
        np.array([9.5, -1.0]),
        atol=1e-12,
    )


def test_transition_jacobian_matches_finite_difference():
    state = np.array([3.0, -2.0, 0.6, 8.4, -0.12, 2.2, 0.75])
    delta_t_s = 0.02
    numerical = _finite_difference_jacobian(
        lambda value: propagate_vehicle_state(value, delta_t_s),
        state,
    )
    np.testing.assert_allclose(
        vehicle_transition_jacobian(state, delta_t_s),
        numerical,
        atol=2e-7,
        rtol=2e-7,
    )


def test_gnss_jacobian_matches_finite_difference():
    state = np.array([3.0, -2.0, 0.6, 8.4, -0.12, 2.2, 0.75])
    numerical = _finite_difference_jacobian(observe_gnss_antenna_position, state)
    np.testing.assert_allclose(
        gnss_antenna_position_jacobian(state),
        numerical,
        atol=2e-7,
        rtol=2e-7,
    )


def test_truth_starts_straight_then_excites_lever_arm_with_turns():
    times = np.arange(0.0, 90.0 + 0.02, 0.02)
    truth = simulate_vehicle_truth(times)
    straight = times < FIRST_TURN_START_S
    turning = times > FIRST_TURN_START_S + 5.0

    np.testing.assert_allclose(truth[straight, YAW_RATE], 0.0, atol=1e-12)
    assert np.max(np.abs(truth[turning, YAW_RATE])) > 0.1
    assert np.ptp(truth[:, HEADING]) > np.deg2rad(90.0)
    assert np.ptp(truth[:, SPEED]) > 1.0
    np.testing.assert_allclose(
        truth[:, LEVER_X : LEVER_Y + 1],
        np.tile(TRUE_LEVER_ARM_M, (len(times), 1)),
    )


@pytest.mark.parametrize(
    "use_jacobian",
    [False, True],
    ids=["unscented", "jacobian"],
)
def test_motion_calibrates_lever_arm_and_recovers_body_position(use_jacobian):
    result = run_gnss_lever_arm_calibration(use_jacobian=use_jacobian)

    assert result["filter_rate_hz"] == 50.0
    assert result["speed_rate_hz"] == 10.0
    assert result["gnss_rate_hz"] == 2.0
    assert len(result["times"]) == 4500
    assert len(result["gyro_times"]) == 4500
    assert len(result["speed_times"]) == 900
    assert len(result["gnss_times"]) == 180
    np.testing.assert_allclose(np.diff(result["times"]), 0.02, atol=1e-12)
    np.testing.assert_allclose(np.diff(result["gnss_times"]), 0.5, atol=1e-12)

    assert result["final_lever_arm_error_m"] < 0.12
    np.testing.assert_allclose(
        result["final_lever_arm_m"],
        TRUE_LEVER_ARM_M,
        atol=0.12,
    )
    assert (
        result["post_turn_lever_uncertainty_m"]
        < 0.25 * result["pre_turn_lever_uncertainty_m"]
    )
    assert result["naive_gnss_as_body_rmse_m"] > 2.0
    assert result["smoothed_body_position_rmse_m"] < 0.25
    assert (
        result["smoothed_body_position_rmse_m"]
        < result["filtered_body_position_rmse_m"]
    )

    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"][::250]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

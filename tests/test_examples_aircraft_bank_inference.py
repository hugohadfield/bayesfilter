import numpy as np
import pytest

from examples.aircraft_bank_inference import (
    BANK,
    COURSE,
    FLIGHT_PATH_ANGLE,
    SPEED,
    aircraft_transition_jacobian,
    bank_profile,
    coordinated_turn_rate,
    observe_position,
    position_jacobian,
    propagate_aircraft_state,
    run_aircraft_bank_inference,
    simulate_aircraft_truth,
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


def test_coordinated_turn_rate_has_expected_sign_and_magnitude():
    speed = 100.0
    bank = np.deg2rad(30.0)
    rate = coordinated_turn_rate(speed, 0.0, bank)

    assert rate > 0.0
    np.testing.assert_allclose(rate, 9.80665 * np.tan(bank) / speed)
    np.testing.assert_allclose(
        coordinated_turn_rate(speed, 0.0, -bank),
        -rate,
    )


def test_transition_jacobian_matches_finite_difference():
    state = np.array(
        [120.0, -70.0, 1500.0, 96.0, 0.8, np.deg2rad(4.0), np.deg2rad(23.0)]
    )
    delta_t_s = 0.8
    numerical = _finite_difference_jacobian(
        lambda value: propagate_aircraft_state(value, delta_t_s),
        state,
    )
    np.testing.assert_allclose(
        aircraft_transition_jacobian(state, delta_t_s),
        numerical,
        atol=2e-6,
        rtol=2e-6,
    )


def test_position_observation_contains_no_attitude_measurement():
    state = np.array([1.0, 2.0, 3.0, 90.0, 0.5, 0.1, -0.2])
    np.testing.assert_allclose(observe_position(state), state[:3])
    jacobian = position_jacobian(state)
    np.testing.assert_allclose(jacobian[:, :3], np.eye(3))
    np.testing.assert_allclose(jacobian[:, 3:], 0.0)


def test_truth_contains_left_right_turns_and_climb_descent():
    times = np.arange(181, dtype=float)
    truth = simulate_aircraft_truth(times)

    assert np.rad2deg(truth[:, BANK]).max() > 24.0
    assert np.rad2deg(truth[:, BANK]).min() < -29.0
    assert np.rad2deg(truth[:, FLIGHT_PATH_ANGLE]).max() > 4.9
    assert np.rad2deg(truth[:, FLIGHT_PATH_ANGLE]).min() < -3.9
    assert np.ptp(truth[:, COURSE]) > np.deg2rad(40.0)
    assert np.ptp(truth[:, SPEED]) > 10.0
    np.testing.assert_allclose(truth[:, BANK], bank_profile(times))


@pytest.mark.parametrize(
    "use_jacobian",
    [False, True],
    ids=["unscented", "jacobian"],
)
def test_position_only_example_recovers_bank_and_flight_path(use_jacobian):
    result = run_aircraft_bank_inference(use_jacobian=use_jacobian)

    assert result["truth"].shape == (1791, 7)
    assert result["measurements"].shape == (180, 3)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert result["filter_rate_hz"] == 10.0
    assert result["position_rate_hz"] == 1.0
    np.testing.assert_allclose(np.diff(result["times"]), 0.1, atol=1e-12)
    np.testing.assert_allclose(np.diff(result["measurement_times"]), 1.0)
    assert len(result["times"]) > 9 * len(result["measurement_times"])
    assert result["raw_measurement_position_rmse_m"] > 45.0
    assert result["smoothed_position_rmse_m"] < result["filtered_position_rmse_m"]
    assert result["smoothed_position_rmse_m"] < 20.0
    assert result["smoothed_bank_rmse_deg"] < result["filtered_bank_rmse_deg"]
    assert result["smoothed_bank_rmse_deg"] < 4.0
    assert result["turning_smoothed_bank_rmse_deg"] < 3.5
    assert result["smoothed_flight_path_angle_rmse_deg"] < 1.0
    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"][::20]:
        np.testing.assert_allclose(covariance, covariance.T, atol=3e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

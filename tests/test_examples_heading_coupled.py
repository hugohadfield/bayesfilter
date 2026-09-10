import numpy as np
import pytest

from examples.heading_coupled import (
    global_position_jacobian,
    heading_coupled_jacobian,
    local_angular_rate_jacobian,
    observe_global_position,
    observe_local_angular_rate,
    propagate_heading_coupled_state,
    run_heading_coupled_tracking,
    simulate_heading_coupled_truth,
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


def test_stationary_vehicle_cannot_translate_or_turn():
    state = np.array([1.2, -0.4, 0.7, 0.0, 0.35])
    propagated = propagate_heading_coupled_state(state, 0.5)

    np.testing.assert_allclose(propagated[:3], state[:3])
    np.testing.assert_allclose(observe_local_angular_rate(state), [0.0])


def test_reverse_motion_reverses_displacement_and_heading_rate():
    forward = np.array([0.0, 0.0, 0.4, 1.2, 0.25])
    reverse = forward.copy()
    reverse[3] *= -1.0
    forward_next = propagate_heading_coupled_state(forward, 0.1)
    reverse_next = propagate_heading_coupled_state(reverse, 0.1)
    heading_direction = np.array([np.cos(forward[2]), np.sin(forward[2])])

    assert np.dot(forward_next[:2], heading_direction) > 0.0
    assert np.dot(reverse_next[:2], heading_direction) < 0.0
    assert forward_next[2] - forward[2] > 0.0
    assert reverse_next[2] - reverse[2] < 0.0


@pytest.mark.parametrize(
    "state",
    [
        np.array([0.3, -0.2, 0.6, 1.1, 0.24]),
        np.array([0.3, -0.2, 0.6, -0.8, 0.24]),
        np.array([0.3, -0.2, 0.6, 0.0, 0.24]),
        np.array([0.3, -0.2, 0.6, 1.1, 0.0]),
    ],
)
def test_transition_jacobian_matches_finite_difference(state):
    delta_t_s = 0.05
    numerical = _finite_difference_jacobian(
        lambda value: propagate_heading_coupled_state(value, delta_t_s),
        state,
    )
    np.testing.assert_allclose(
        heading_coupled_jacobian(state, delta_t_s),
        numerical,
        atol=1e-8,
    )


def test_observation_jacobians_match_finite_difference():
    state = np.array([0.3, -0.2, 0.6, -0.8, 0.24])
    np.testing.assert_allclose(
        global_position_jacobian(state),
        _finite_difference_jacobian(observe_global_position, state),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        local_angular_rate_jacobian(state),
        _finite_difference_jacobian(observe_local_angular_rate, state),
        atol=1e-10,
    )


def test_truth_heading_is_fixed_during_stationary_interval():
    times = np.arange(401) * 0.05
    truth = simulate_heading_coupled_truth(times)
    stopped = (times >= 8.0) & (times <= 10.0)

    np.testing.assert_allclose(truth[stopped, 3], 0.0)
    assert np.ptp(truth[stopped, 2]) < 1e-12


@pytest.mark.parametrize(
    "use_jacobian",
    [False, True],
    ids=["unscented", "jacobian"],
)
def test_heading_coupled_example_tracks_and_smooths_both_modes(use_jacobian):
    result = run_heading_coupled_tracking(use_jacobian=use_jacobian)

    assert result["truth"].shape == (400, 5)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert result["position_measurements"].shape == (100, 2)
    assert result["angular_rate_measurements"].shape == (400,)
    assert result["event_times"].shape == (500,)
    assert np.all(np.diff(result["event_times"]) >= 0.0)
    np.testing.assert_allclose(np.diff(result["position_times"]), 0.2)
    np.testing.assert_allclose(np.diff(result["angular_rate_times"]), 0.05)
    np.testing.assert_allclose(result["times"], result["angular_rate_times"])
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_heading_rmse"] < result["filtered_heading_rmse"]
    assert result["smoothed_position_rmse"] < 0.20
    assert result["smoothed_heading_rmse"] < 0.06
    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"][::20]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

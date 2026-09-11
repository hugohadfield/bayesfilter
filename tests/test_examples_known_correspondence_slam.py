import numpy as np
import pytest

from examples.known_correspondence_slam import (
    LANDMARK_POSITIONS,
    observe_slam_state,
    propagate_slam_state,
    run_known_correspondence_slam,
    slam_observation_jacobian,
    slam_transition_jacobian,
    visible_landmarks,
)


def _finite_difference_jacobian(function, state, step=1e-6):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append((function(state + offset) - function(state - offset)) / (2 * step))
    return np.column_stack(columns)


def _example_state():
    return np.concatenate(
        (np.array([1.2, -0.4, 0.35, 1.1, 0.08]), LANDMARK_POSITIONS.ravel())
    )


def test_slam_transition_preserves_static_landmarks():
    state = _example_state()
    propagated = propagate_slam_state(state, 0.2)

    np.testing.assert_allclose(propagated[5:], state[5:])
    assert not np.allclose(propagated[:3], state[:3])


def test_slam_transition_jacobian_matches_finite_difference():
    state = _example_state()
    delta_t_s = 0.15

    np.testing.assert_allclose(
        slam_transition_jacobian(state, delta_t_s),
        _finite_difference_jacobian(
            lambda candidate: propagate_slam_state(candidate, delta_t_s),
            state,
        ),
        atol=2e-9,
    )


def test_slam_observation_jacobian_matches_finite_difference():
    state = _example_state()
    landmark_indices = (0, 3, 6)

    np.testing.assert_allclose(
        slam_observation_jacobian(state, landmark_indices),
        _finite_difference_jacobian(
            lambda candidate: observe_slam_state(candidate, landmark_indices),
            state,
        ),
        atol=2e-9,
    )


def test_visibility_respects_range_and_field_of_view():
    state = _example_state()
    state[:3] = [0.0, 0.0, 0.0]

    visible = visible_landmarks(state, maximum_range=7.0, field_of_view_rad=np.pi)

    assert 0 in visible
    assert 6 not in visible
    assert 2 not in visible


@pytest.mark.parametrize(
    "use_jacobian,maximum_position_rmse",
    [(True, 0.16), (False, 0.22)],
    ids=["jacobian", "unscented"],
)
def test_slam_example_recovers_trajectory_and_map(
    use_jacobian,
    maximum_position_rmse,
):
    result = run_known_correspondence_slam(use_jacobian=use_jacobian)

    assert result["truth"].shape == (521, 21)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert len(result["visible_landmark_indices"]) == 520
    visible_counts = [len(indices) for indices in result["visible_landmark_indices"]]
    assert max(visible_counts) >= 4
    assert np.mean(np.asarray(visible_counts) > 0) > 0.9
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_position_rmse"] < maximum_position_rmse
    assert result["final_landmark_rmse"] < 0.12
    assert result["final_landmark_rmse"] < 0.12 * result["initial_landmark_rmse"]
    assert np.isfinite(result["smoothed"]).all()

    for covariance in result["smoothed_covariances"][::40]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

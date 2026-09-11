import numpy as np

from examples.ballistic_tracking import (
    GRAVITY_KM_S2,
    atmospheric_density_ratio,
    ballistic_derivative,
    observe_radar,
    radar_jacobian,
    run_ballistic_tracking,
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


def test_exponential_atmosphere_thins_with_altitude():
    density = atmospheric_density_ratio(np.array([0.0, 7.5, 15.0, 30.0]))

    np.testing.assert_allclose(density[0], 1.0)
    assert np.all(np.diff(density) < 0.0)
    np.testing.assert_allclose(density[1], np.exp(-1.0))


def test_ballistic_drag_opposes_velocity():
    state = np.array([0.0, 12.0, 1.2, -0.4, np.log(0.04)])
    acceleration = ballistic_derivative(state)[2:4]
    drag_acceleration = acceleration - np.array([0.0, -GRAVITY_KM_S2])

    assert drag_acceleration @ state[2:4] < 0.0


def test_radar_jacobian_matches_finite_difference():
    state = np.array([-9.0, 14.0, 1.1, -0.2, np.log(0.04)])

    np.testing.assert_allclose(
        radar_jacobian(state),
        _finite_difference_jacobian(observe_radar, state),
        atol=2e-9,
    )


def test_ballistic_example_tracks_and_infers_drag():
    result = run_ballistic_tracking()

    assert result["truth"].shape == (451, 5)
    assert result["measurements"].shape == (450, 2)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_position_rmse"] < 0.025

    true_drag = result["true_drag_coefficient"]
    initial_drag_error = abs(result["filtered_drag_coefficient"][0] - true_drag)
    final_drag_error = abs(result["filtered_drag_coefficient"][-1] - true_drag)
    assert final_drag_error < 0.05 * true_drag
    assert final_drag_error < 0.05 * initial_drag_error
    assert np.isfinite(result["smoothed"]).all()

    for covariance in result["smoothed_covariances"][::30]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

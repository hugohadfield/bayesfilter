import numpy as np

from examples.orbit_determination import (
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    ground_station_states,
    observe_range_and_range_rate,
    range_and_range_rate_jacobian,
    run_orbit_determination,
    simulate_orbit_truth,
    visible_ground_stations,
)


def _finite_difference_jacobian(function, state, step=1e-5):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append(
            (function(state + offset) - function(state - offset)) / (2 * step)
        )
    return np.column_stack(columns)


def test_rotating_ground_stations_stay_on_earth_and_move_tangentially():
    positions, velocities = ground_station_states(1234.0)

    np.testing.assert_allclose(
        np.linalg.norm(positions, axis=1),
        EARTH_RADIUS_KM,
    )
    np.testing.assert_allclose(np.sum(positions * velocities, axis=1), 0.0, atol=1e-10)


def test_visibility_rejects_station_on_far_side_of_earth():
    station_positions = np.array(
        [[EARTH_RADIUS_KM, 0.0], [-EARTH_RADIUS_KM, 0.0]]
    )
    target_position = np.array([7050.0, 0.0])

    np.testing.assert_array_equal(
        visible_ground_stations(target_position, station_positions),
        [0],
    )


def test_range_and_range_rate_jacobian_matches_finite_difference():
    state = np.array([6800.0, 1900.0, -2.1, 7.1])
    station_position, station_velocity = ground_station_states(800.0)
    position = station_position[0]
    velocity = station_velocity[0]

    np.testing.assert_allclose(
        range_and_range_rate_jacobian(state, position, velocity),
        _finite_difference_jacobian(
            lambda value: observe_range_and_range_rate(value, position, velocity),
            state,
        ),
        atol=2e-8,
    )


def test_circular_two_body_truth_conserves_radius_and_energy():
    radius_km = 7050.0
    initial_state = np.array(
        [radius_km, 0.0, 0.0, np.sqrt(EARTH_MU_KM3_S2 / radius_km)]
    )
    times = np.arange(181) * 30.0
    truth = simulate_orbit_truth(initial_state, times)
    radii = np.linalg.norm(truth[:, :2], axis=1)
    energy = np.sum(truth[:, 2:4] ** 2, axis=1) / 2.0 - EARTH_MU_KM3_S2 / radii

    assert np.ptp(radii) < 3e-4
    assert np.ptp(energy) < 2e-7


def test_orbit_example_recovers_state_from_ground_stations():
    result = run_orbit_determination()

    assert result["truth"].shape == (181, 4)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert len(result["measurements"]) == 180
    visible_counts = np.array(
        [len(indices) for indices in result["visible_station_indices"]]
    )
    assert visible_counts.min() == 1
    assert visible_counts.max() >= 2
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_position_rmse"] < 0.15
    assert np.isfinite(result["smoothed"]).all()

    for covariance in result["smoothed_covariances"][::20]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

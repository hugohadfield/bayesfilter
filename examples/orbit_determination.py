"""Orbit determination from rotating ground-station measurements."""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_S = 7.2921159e-5
GROUND_STATION_LONGITUDES_RAD = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)


def orbital_derivative(state):
    """Differentiate planar inertial state ``[x, y, vx, vy]``."""
    state = np.asarray(state)
    position = state[:2]
    radius = np.linalg.norm(position)
    acceleration = -EARTH_MU_KM3_S2 * position / radius**3
    return np.concatenate((state[2:4], acceleration))


def propagate_orbit_state(state, delta_t_s):
    """Advance the two-body state with a fourth-order Runge-Kutta step."""
    state = np.asarray(state)
    stage_1 = orbital_derivative(state)
    stage_2 = orbital_derivative(state + 0.5 * delta_t_s * stage_1)
    stage_3 = orbital_derivative(state + 0.5 * delta_t_s * stage_2)
    stage_4 = orbital_derivative(state + delta_t_s * stage_3)
    return state + delta_t_s * (
        stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
    ) / 6.0


def ground_station_states(time_s, longitudes=GROUND_STATION_LONGITUDES_RAD):
    """Return inertial positions and velocities of equatorial ground stations."""
    angles = np.asarray(longitudes) + EARTH_ROTATION_RAD_S * time_s
    positions = EARTH_RADIUS_KM * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    velocities = EARTH_ROTATION_RAD_S * np.column_stack(
        (-positions[:, 1], positions[:, 0])
    )
    return positions, velocities


def visible_ground_stations(target_position, station_positions):
    """Return station indices whose target line of sight is above the horizon."""
    target_position = np.asarray(target_position)
    line_of_sight = target_position - station_positions
    return np.flatnonzero(np.sum(line_of_sight * station_positions, axis=1) > 0.0)


def observe_range_and_range_rate(state, station_position, station_velocity):
    """Observe slant range and range rate from one ground station."""
    relative_position = np.asarray(state)[:2] - np.asarray(station_position)
    relative_velocity = np.asarray(state)[2:4] - np.asarray(station_velocity)
    range_km = np.linalg.norm(relative_position)
    range_rate_km_s = relative_position @ relative_velocity / range_km
    return np.array([range_km, range_rate_km_s])


def range_and_range_rate_jacobian(state, station_position, station_velocity):
    """Return the range/range-rate observation Jacobian."""
    relative_position = np.asarray(state)[:2] - np.asarray(station_position)
    relative_velocity = np.asarray(state)[2:4] - np.asarray(station_velocity)
    range_km = np.linalg.norm(relative_position)
    direction = relative_position / range_km
    range_rate = direction @ relative_velocity
    jacobian = np.zeros((2, 4))
    jacobian[0, :2] = direction
    jacobian[1, :2] = (relative_velocity - range_rate * direction) / range_km
    jacobian[1, 2:4] = direction
    return jacobian


def observe_station_network(state, station_positions, station_velocities):
    """Stack range/range-rate observations from the selected stations."""
    return np.concatenate(
        [
            observe_range_and_range_rate(state, position, velocity)
            for position, velocity in zip(station_positions, station_velocities)
        ]
    )


def station_network_jacobian(state, station_positions, station_velocities):
    """Stack the selected station observation Jacobians."""
    return np.vstack(
        [
            range_and_range_rate_jacobian(state, position, velocity)
            for position, velocity in zip(station_positions, station_velocities)
        ]
    )


def simulate_orbit_truth(initial_state, times):
    """Simulate a deterministic planar two-body orbit."""
    truth = [np.asarray(initial_state)]
    for delta_t_s in np.diff(times):
        truth.append(propagate_orbit_state(truth[-1], delta_t_s))
    return np.array(truth)


def run_orbit_determination(seed=701, duration_s=5400.0, delta_t_s=30.0):
    """Determine a low-Earth orbit from visibility-limited ground observations."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s / delta_t_s) + 1) * delta_t_s
    initial_radius_km = 7050.0
    initial_truth = np.array(
        [0.0, initial_radius_km, -np.sqrt(EARTH_MU_KM3_S2 / initial_radius_km), 0.0]
    )
    truth = simulate_orbit_truth(initial_truth, times)

    range_std_km = 0.15
    range_rate_std_km_s = 0.002
    observations = []
    measurements = []
    visible_station_indices = []
    station_positions_by_time = []
    for time_s, truth_state in zip(times[1:], truth[1:]):
        all_positions, all_velocities = ground_station_states(time_s)
        visible = visible_ground_stations(truth_state[:2], all_positions)
        selected_positions = all_positions[visible]
        selected_velocities = all_velocities[visible]
        station_positions_by_time.append(all_positions)
        visible_station_indices.append(visible)

        def observation_func(
            state,
            positions=selected_positions,
            velocities=selected_velocities,
        ):
            return observe_station_network(state, positions, velocities)

        def jacobian_func(
            state,
            positions=selected_positions,
            velocities=selected_velocities,
        ):
            return station_network_jacobian(state, positions, velocities)

        exact_measurement = observation_func(truth_state)
        measurement_std = np.tile(
            [range_std_km, range_rate_std_km_s], len(visible)
        )
        measurement = exact_measurement + rng.normal(0.0, measurement_std)
        covariance = np.diag(measurement_std**2)
        measurements.append(measurement)
        observations.append(
            Observation(measurement, covariance, observation_func, jacobian_func)
        )

    model = StateTransitionModel(
        propagate_orbit_state,
        np.diag([2e-3, 2e-3, 2e-7, 2e-7]),
    )
    initial_mean = initial_truth + np.array([8.0, -6.0, 0.012, -0.008])
    initial_covariance = np.diag([15.0**2, 15.0**2, 0.03**2, 0.03**2])
    bayes_filter = BayesianFilter(model, Gaussian(initial_mean, initial_covariance))
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=False,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=False,
    )
    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.array(
        [state.covariance() for state in smoother_states]
    )

    def position_rmse(states):
        return np.sqrt(np.mean(np.sum((states[:, :2] - truth[:, :2]) ** 2, axis=1)))

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "station_positions": np.array(station_positions_by_time),
        "visible_station_indices": visible_station_indices,
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "range_std_km": range_std_km,
        "range_rate_std_km_s": range_rate_std_km_s,
    }


if __name__ == "__main__":
    from examples.plotting import plot_orbit_determination

    result = run_orbit_determination()
    path = Path(__file__).parent / "figures" / "orbit-determination.png"
    plot_orbit_determination(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f} km"
    )
    print(f"saved {path}")

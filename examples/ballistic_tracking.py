"""Radar tracking of a ballistic object with drag-coefficient inference."""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_KM_S2 = 9.80665e-3
ATMOSPHERE_SCALE_HEIGHT_KM = 7.5
RADAR_POSITION_KM = np.array([0.0, 0.0])


def atmospheric_density_ratio(altitude_km):
    """Return density relative to sea level for an exponential atmosphere."""
    altitude_km = np.clip(altitude_km, 0.0, 100.0)
    return np.exp(-altitude_km / ATMOSPHERE_SCALE_HEIGHT_KM)


def ballistic_derivative(state):
    """Differentiate ``[x, h, vx, vh, log(c_drag)]`` in km and seconds."""
    state = np.asarray(state)
    velocity = state[2:4]
    speed = np.linalg.norm(velocity)
    drag_coefficient = np.exp(np.clip(state[4], -12.0, 2.0))
    drag_scale = (
        drag_coefficient
        * atmospheric_density_ratio(state[1])
        * speed
    )
    acceleration = -drag_scale * velocity
    acceleration[1] -= GRAVITY_KM_S2
    return np.array(
        [velocity[0], velocity[1], acceleration[0], acceleration[1], 0.0]
    )


def propagate_ballistic_state(state, delta_t_s):
    """Advance the ballistic state with a fourth-order Runge-Kutta step."""
    state = np.asarray(state)
    stage_1 = ballistic_derivative(state)
    stage_2 = ballistic_derivative(state + 0.5 * delta_t_s * stage_1)
    stage_3 = ballistic_derivative(state + 0.5 * delta_t_s * stage_2)
    stage_4 = ballistic_derivative(state + delta_t_s * stage_3)
    return state + delta_t_s * (
        stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
    ) / 6.0


def observe_radar(state, radar_position=RADAR_POSITION_KM):
    """Observe slant range in km and elevation angle in radians."""
    displacement = np.asarray(state)[:2] - np.asarray(radar_position)
    return np.array(
        [
            np.linalg.norm(displacement),
            np.arctan2(displacement[1], displacement[0]),
        ]
    )


def radar_jacobian(state, radar_position=RADAR_POSITION_KM):
    """Return the range/elevation observation Jacobian."""
    displacement = np.asarray(state)[:2] - np.asarray(radar_position)
    range_squared = displacement @ displacement
    range_km = np.sqrt(range_squared)
    jacobian = np.zeros((2, 5))
    jacobian[0, :2] = displacement / range_km
    jacobian[1, :2] = np.array(
        [-displacement[1], displacement[0]]
    ) / range_squared
    return jacobian


def simulate_ballistic_truth(initial_state, times):
    """Simulate a deterministic ballistic trajectory."""
    truth = [np.asarray(initial_state)]
    for delta_t_s in np.diff(times):
        truth.append(propagate_ballistic_state(truth[-1], delta_t_s))
    return np.array(truth)


def run_ballistic_tracking(seed=601, duration_s=45.0, delta_t_s=0.1):
    """Infer trajectory, velocity, and drag from noisy range/elevation radar."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s / delta_t_s) + 1) * delta_t_s
    true_drag_coefficient = 0.035
    initial_truth = np.array(
        [-22.0, 18.0, 1.45, 0.15, np.log(true_drag_coefficient)]
    )
    truth = simulate_ballistic_truth(initial_truth, times)

    range_std_km = 0.08
    elevation_std_rad = np.deg2rad(0.15)
    measurement_covariance = np.diag(
        [range_std_km**2, elevation_std_rad**2]
    )
    measurements = np.array(
        [
            observe_radar(state)
            + rng.normal(0.0, [range_std_km, elevation_std_rad])
            for state in truth[1:]
        ]
    )
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_radar,
            radar_jacobian,
        )
        for measurement in measurements
    ]

    model = StateTransitionModel(
        propagate_ballistic_state,
        np.diag([2e-5, 2e-5, 3e-6, 3e-6, 2e-6]),
    )
    initial_mean = initial_truth + np.array(
        [0.45, -0.35, 0.06, -0.04, np.log(0.060 / true_drag_coefficient)]
    )
    initial_covariance = np.diag(
        [0.8**2, 0.8**2, 0.12**2, 0.12**2, 0.5**2]
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
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
        "true_drag_coefficient": true_drag_coefficient,
        "filtered_drag_coefficient": np.exp(filtered[:, 4]),
        "smoothed_drag_coefficient": np.exp(smoothed[:, 4]),
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "radar_position": RADAR_POSITION_KM.copy(),
        "range_std_km": range_std_km,
        "elevation_std_rad": elevation_std_rad,
    }


if __name__ == "__main__":
    from examples.plotting import plot_ballistic_tracking

    result = run_ballistic_tracking()
    path = Path(__file__).parent / "figures" / "ballistic-drag-estimation.png"
    plot_ballistic_tracking(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f} km"
    )
    print(
        f"drag coefficient: {result['filtered_drag_coefficient'][0]:.4f} -> "
        f"{result['filtered_drag_coefficient'][-1]:.4f} "
        f"(true {result['true_drag_coefficient']:.4f})"
    )
    print(f"saved {path}")

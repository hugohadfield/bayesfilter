"""Weigh a planet from the gravitational bending of a spacecraft flyby.

A synthetic spacecraft performs a close hyperbolic flyby of a Jupiter-like
planet. The navigation system supplies a noisy 2D spacecraft position fix
once every four hours. The filter does not know the planet's gravitational
parameter: it estimates it jointly with spacecraft position and velocity.

The hidden state is

``[x, y, vx, vy, log(mu)]``

where ``mu = G M`` is the planet's gravitational parameter in km^3/s^2.
Representing ``mu`` in log space keeps the inferred mass positive. A second
filter holds ``mu`` fixed at the deliberately wrong initial value, making the
value of online system identification visible in the trajectory itself.

The measurements are intentionally simplified navigation position fixes rather
than raw radiometric tracking data. That keeps the example focused on the
central idea: an unknown physical parameter can be included in the filter state
when it affects the dynamics that generate the observations.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


POSITION = slice(0, 2)
VELOCITY = slice(2, 4)
LOG_MU = 4

GRAVITATIONAL_CONSTANT_KM3_KG_S2 = 6.67430e-20
JUPITER_MU_KM3_S2 = 1.26686534e8
JUPITER_RADIUS_KM = 71492.0
TRUE_PLANET_MASS_KG = JUPITER_MU_KM3_S2 / GRAVITATIONAL_CONSTANT_KM3_KG_S2

INCOMING_SPEED_KM_S = 18.0
TARGET_PERIAPSIS_KM = 4.0 * JUPITER_RADIUS_KM
IMPACT_PARAMETER_KM = TARGET_PERIAPSIS_KM * np.sqrt(
    1.0
    + 2.0
    * JUPITER_MU_KM3_S2
    / (TARGET_PERIAPSIS_KM * INCOMING_SPEED_KM_S**2)
)
INITIAL_X_KM = -4.0e6
DURATION_S = 6.0 * 24.0 * 60.0 * 60.0
OBSERVATION_INTERVAL_S = 4.0 * 60.0 * 60.0
POSITION_STD_KM = 3000.0
MAX_INTEGRATION_STEP_S = 900.0
INITIAL_MU_FRACTION = 0.60


def gravitational_acceleration(position_km, mu_km3_s2):
    """Return two-body gravitational acceleration toward the planet."""
    position_km = np.asarray(position_km, dtype=float)
    radius_km = np.linalg.norm(position_km)
    return -mu_km3_s2 * position_km / radius_km**3


def _cartesian_derivative(state, mu_km3_s2):
    state = np.asarray(state)
    return np.concatenate(
        [
            state[VELOCITY],
            gravitational_acceleration(state[POSITION], mu_km3_s2),
        ]
    )


def propagate_cartesian_state(state, delta_t_s, mu_km3_s2):
    """Propagate ``[x, y, vx, vy]`` using fourth-order Runge-Kutta substeps."""
    state = np.asarray(state, dtype=float).copy()
    substeps = max(
        1,
        int(np.ceil(abs(delta_t_s) / MAX_INTEGRATION_STEP_S)),
    )
    step_s = delta_t_s / substeps

    for _ in range(substeps):
        stage_1 = _cartesian_derivative(state, mu_km3_s2)
        stage_2 = _cartesian_derivative(
            state + 0.5 * step_s * stage_1,
            mu_km3_s2,
        )
        stage_3 = _cartesian_derivative(
            state + 0.5 * step_s * stage_2,
            mu_km3_s2,
        )
        stage_4 = _cartesian_derivative(
            state + step_s * stage_3,
            mu_km3_s2,
        )
        state = state + step_s * (
            stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
        ) / 6.0

    return state


def _augmented_derivative(state):
    state = np.asarray(state)
    mu_km3_s2 = np.exp(state[LOG_MU])
    return np.concatenate(
        [
            state[VELOCITY],
            gravitational_acceleration(state[POSITION], mu_km3_s2),
            [0.0],
        ]
    )


def propagate_flyby_state(state, delta_t_s):
    """Propagate ``[x, y, vx, vy, log(mu)]`` through the two-body flyby."""
    state = np.asarray(state, dtype=float).copy()
    substeps = max(
        1,
        int(np.ceil(abs(delta_t_s) / MAX_INTEGRATION_STEP_S)),
    )
    step_s = delta_t_s / substeps

    for _ in range(substeps):
        stage_1 = _augmented_derivative(state)
        stage_2 = _augmented_derivative(state + 0.5 * step_s * stage_1)
        stage_3 = _augmented_derivative(state + 0.5 * step_s * stage_2)
        stage_4 = _augmented_derivative(state + step_s * stage_3)
        state = state + step_s * (
            stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
        ) / 6.0

    return state


def observe_position(state):
    """Observe the spacecraft's planar position."""
    return np.asarray(state)[POSITION]


def initial_truth_state():
    """Return the incoming spacecraft state far upstream of the planet."""
    return np.array(
        [
            INITIAL_X_KM,
            IMPACT_PARAMETER_KM,
            INCOMING_SPEED_KM_S,
            0.0,
        ]
    )


def simulate_flyby_truth():
    """Generate the deterministic Jupiter-like flyby at observation times."""
    times = np.arange(
        round(DURATION_S / OBSERVATION_INTERVAL_S) + 1,
        dtype=float,
    ) * OBSERVATION_INTERVAL_S
    truth = [initial_truth_state()]
    for delta_t_s in np.diff(times):
        truth.append(
            propagate_cartesian_state(
                truth[-1],
                delta_t_s,
                JUPITER_MU_KM3_S2,
            )
        )
    return times, np.asarray(truth)


def make_position_measurements(seed=42):
    """Generate one noisy navigation position fix every four hours."""
    times, truth = simulate_flyby_truth()
    rng = np.random.default_rng(seed)
    measurements = truth[1:, POSITION] + rng.normal(
        0.0,
        POSITION_STD_KM,
        size=(len(times) - 1, 2),
    )
    return times, truth, measurements


def process_noise_covariance():
    """Return small model-error noise for one four-hour filtering interval."""
    return np.diag(
        [
            2.0**2,
            2.0**2,
            2e-4**2,
            2e-4**2,
            2e-5**2,
        ]
    )


def initial_distribution():
    """Start with poor navigation and a planet that is 40% too light."""
    truth = initial_truth_state()
    mean = np.array(
        [
            truth[0] + 40000.0,
            truth[1] - 30000.0,
            truth[2] + 0.20,
            truth[3] - 0.15,
            np.log(INITIAL_MU_FRACTION * JUPITER_MU_KM3_S2),
        ]
    )
    covariance = np.diag(
        [
            70000.0**2,
            70000.0**2,
            0.40**2,
            0.40**2,
            0.60**2,
        ]
    )
    return Gaussian(mean, covariance)


def _make_observations(measurements):
    covariance = np.eye(2) * POSITION_STD_KM**2
    return [
        Observation(measurement, covariance, observe_position)
        for measurement in measurements
    ]


def _run_estimating_filter(times, measurements):
    model = StateTransitionModel(
        propagate_flyby_state,
        process_noise_covariance(),
    )
    bayes_filter = BayesianFilter(model, initial_distribution())
    filtered_states = bayes_filter.run_synchronous(
        _make_observations(measurements),
        times,
        use_jacobian=False,
    )
    smoothed_states = RTS(bayes_filter).apply(
        filtered_states,
        times,
        use_jacobian=False,
    )
    return filtered_states, smoothed_states


def _run_fixed_wrong_mass_filter(times, measurements):
    fixed_mu = INITIAL_MU_FRACTION * JUPITER_MU_KM3_S2

    def fixed_mass_transition(state, delta_t_s):
        return propagate_cartesian_state(state, delta_t_s, fixed_mu)

    truth = initial_truth_state()
    initial_state = Gaussian(
        np.array(
            [
                truth[0] + 40000.0,
                truth[1] - 30000.0,
                truth[2] + 0.20,
                truth[3] - 0.15,
            ]
        ),
        np.diag([70000.0**2, 70000.0**2, 0.40**2, 0.40**2]),
    )
    model = StateTransitionModel(
        fixed_mass_transition,
        process_noise_covariance()[:4, :4],
    )
    bayes_filter = BayesianFilter(model, initial_state)
    return bayes_filter.run_synchronous(
        _make_observations(measurements),
        times,
        use_jacobian=False,
    )


def _state_array(states):
    return np.asarray([state.mean() for state in states])


def _covariance_array(states):
    return np.asarray([state.covariance() for state in states])


def _position_rmse(states, truth):
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (states[:, POSITION] - truth[:, POSITION]) ** 2,
                    axis=1,
                )
            )
        )
    )


def _velocity_deflection_deg(truth):
    incoming = truth[0, VELOCITY]
    outgoing = truth[-1, VELOCITY]
    cosine = float(
        incoming @ outgoing
        / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    )
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def run_planet_flyby_mass_estimation(seed=42):
    """Estimate a Jupiter-like planet's mass from a noisy spacecraft flyby."""
    times, truth, measurements = make_position_measurements(seed)
    filtered_states, smoothed_states = _run_estimating_filter(times, measurements)
    fixed_states = _run_fixed_wrong_mass_filter(times, measurements)

    filtered = _state_array(filtered_states)
    smoothed = _state_array(smoothed_states)
    fixed = _state_array(fixed_states)
    filtered_covariances = _covariance_array(filtered_states)
    smoothed_covariances = _covariance_array(smoothed_states)

    filtered_mu = np.exp(filtered[:, LOG_MU])
    smoothed_mu = np.exp(smoothed[:, LOG_MU])
    radius_km = np.linalg.norm(truth[:, POSITION], axis=1)
    closest_approach_index = int(np.argmin(radius_km))

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "fixed_wrong_mass": fixed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_mu_km3_s2": filtered_mu,
        "smoothed_mu_km3_s2": smoothed_mu,
        "final_mu_km3_s2": float(filtered_mu[-1]),
        "final_mass_kg": float(
            filtered_mu[-1] / GRAVITATIONAL_CONSTANT_KM3_KG_S2
        ),
        "true_mass_kg": TRUE_PLANET_MASS_KG,
        "filtered_position_rmse_km": _position_rmse(filtered, truth),
        "smoothed_position_rmse_km": _position_rmse(smoothed, truth),
        "fixed_wrong_mass_position_rmse_km": _position_rmse(fixed, truth),
        "closest_approach_index": closest_approach_index,
        "closest_approach_time_s": float(times[closest_approach_index]),
        "closest_approach_radius_km": float(radius_km[closest_approach_index]),
        "flyby_deflection_deg": _velocity_deflection_deg(truth),
        "final_log_mu_std": float(
            np.sqrt(filtered_covariances[-1, LOG_MU, LOG_MU])
        ),
    }


def plot_planet_flyby_mass_estimation(result, output_path=None):
    """Plot trajectory bending and convergence of the inferred planet mass."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        result["truth"][:, 0],
        result["truth"][:, 1],
        "--",
        label="truth",
    )
    axes[0].plot(
        result["filtered"][:, 0],
        result["filtered"][:, 1],
        label="estimated mass",
    )
    axes[0].plot(
        result["fixed_wrong_mass"][:, 0],
        result["fixed_wrong_mass"][:, 1],
        label="mass fixed 40% low",
    )
    axes[0].add_patch(
        Circle(
            (0.0, 0.0),
            JUPITER_RADIUS_KM,
            alpha=0.25,
            label="planet",
        )
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [km]")
    axes[0].set_ylabel("y [km]")
    axes[0].set_title("Spacecraft flyby")
    axes[0].legend()

    time_h = result["times"] / 3600.0
    filtered_ratio = result["filtered_mu_km3_s2"] / JUPITER_MU_KM3_S2
    smoothed_ratio = result["smoothed_mu_km3_s2"] / JUPITER_MU_KM3_S2
    log_mu_std = np.sqrt(result["filtered_covariances"][:, LOG_MU, LOG_MU])
    lower_ratio = (
        np.exp(result["filtered"][:, LOG_MU] - log_mu_std)
        / JUPITER_MU_KM3_S2
    )
    upper_ratio = (
        np.exp(result["filtered"][:, LOG_MU] + log_mu_std)
        / JUPITER_MU_KM3_S2
    )

    axes[1].fill_between(
        time_h,
        lower_ratio,
        upper_ratio,
        alpha=0.15,
        label="filter ±1σ",
    )
    axes[1].plot(time_h, filtered_ratio, label="filtered mass")
    axes[1].plot(time_h, smoothed_ratio, label="RTS-smoothed mass")
    axes[1].axhline(1.0, linestyle="--", label="true mass")
    axes[1].axvline(
        result["closest_approach_time_s"] / 3600.0,
        linestyle=":",
        label="closest approach",
    )
    axes[1].set_xlabel("time [h]")
    axes[1].set_ylabel("estimated mass / true mass")
    axes[1].set_title("The flyby weighs the planet")
    axes[1].legend()

    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    return figure


def _print_summary(result):
    mass_error_percent = 100.0 * (
        result["final_mass_kg"] / result["true_mass_kg"] - 1.0
    )
    print("Planet mass estimation from a spacecraft flyby")
    print(
        f"  closest approach: "
        f"{result['closest_approach_radius_km'] / JUPITER_RADIUS_KM:.2f} "
        "planet radii"
    )
    print(f"  trajectory deflection: {result['flyby_deflection_deg']:.1f} deg")
    print(
        f"  mu: {INITIAL_MU_FRACTION:.2f} initial -> "
        f"{result['final_mu_km3_s2'] / JUPITER_MU_KM3_S2:.4f} of truth"
    )
    print(f"  final mass error: {mass_error_percent:+.3f}%")
    print(
        f"  position RMSE: {result['filtered_position_rmse_km']:.0f} km filtered, "
        f"{result['smoothed_position_rmse_km']:.0f} km RTS, "
        f"{result['fixed_wrong_mass_position_rmse_km']:.0f} km with mass fixed wrong"
    )


if __name__ == "__main__":
    result = run_planet_flyby_mass_estimation()
    path = Path(__file__).parent / "figures" / "planet-flyby-mass.png"
    plot_planet_flyby_mass_estimation(result, path)
    _print_summary(result)
    print(f"saved {path}")

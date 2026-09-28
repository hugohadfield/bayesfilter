"""Recover a heliocentric asteroid orbit from noisy telescope angles only.

The observer is Earth, moving on a known circular orbit around the Sun.  The
asteroid follows planar two-body dynamics, but the telescope never observes
range: every measurement is only a noisy line-of-sight direction from Earth.
As Earth moves and the asteroid's trajectory curves under solar gravity, those
angles become sufficient to recover the asteroid's heliocentric position and
velocity.

The hidden state is ``[x, y, vx, vy]`` in AU and AU/day.  The observation is a
2D unit line-of-sight vector, which avoids angular wrap-around inside the
generic BayesFilter update.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


# Gaussian gravitational constant squared, in AU^3 / day^2.
SUN_MU_AU3_DAY2 = 0.0002959122082855911
DURATION_DAYS = 180.0
OBSERVATION_INTERVAL_DAYS = 2.0
TIMING_JITTER = 0.20
ANGLE_STD_ARCSEC = 5.0
ANGLE_STD_RAD = np.deg2rad(ANGLE_STD_ARCSEC / 3600.0)
OBSERVATION_GAP_START_DAY = 70.0
OBSERVATION_GAP_END_DAY = 85.0

TRUE_SEMIMAJOR_AXIS_AU = 1.35
TRUE_ECCENTRICITY = 0.28
TRUE_ANOMALY_RAD = np.deg2rad(50.0)

POSITION = slice(0, 2)
VELOCITY = slice(2, 4)


def state_from_orbital_elements(semimajor_axis_au, eccentricity, true_anomaly_rad):
    """Create a planar heliocentric Cartesian state from Keplerian elements."""
    semilatus_rectum = semimajor_axis_au * (1.0 - eccentricity**2)
    radius = semilatus_rectum / (
        1.0 + eccentricity * np.cos(true_anomaly_rad)
    )
    position = radius * np.array(
        [np.cos(true_anomaly_rad), np.sin(true_anomaly_rad)]
    )
    velocity_scale = np.sqrt(SUN_MU_AU3_DAY2 / semilatus_rectum)
    velocity = velocity_scale * np.array(
        [
            -np.sin(true_anomaly_rad),
            eccentricity + np.cos(true_anomaly_rad),
        ]
    )
    return np.concatenate([position, velocity])


def orbital_elements(state):
    """Return planar semi-major axis and eccentricity from a Cartesian state."""
    state = np.asarray(state)
    position = state[POSITION]
    velocity = state[VELOCITY]
    radius = np.linalg.norm(position)
    speed_squared = float(velocity @ velocity)
    specific_energy = 0.5 * speed_squared - SUN_MU_AU3_DAY2 / radius
    semimajor_axis = -SUN_MU_AU3_DAY2 / (2.0 * specific_energy)
    angular_momentum = position[0] * velocity[1] - position[1] * velocity[0]
    eccentricity_squared = 1.0 - (
        angular_momentum**2 / (SUN_MU_AU3_DAY2 * semimajor_axis)
    )
    eccentricity = np.sqrt(max(0.0, eccentricity_squared))
    return float(semimajor_axis), float(eccentricity)


def initial_truth_state():
    return state_from_orbital_elements(
        TRUE_SEMIMAJOR_AXIS_AU,
        TRUE_ECCENTRICITY,
        TRUE_ANOMALY_RAD,
    )


def _state_derivative(state):
    position = np.asarray(state)[POSITION]
    velocity = np.asarray(state)[VELOCITY]
    radius = np.linalg.norm(position)
    acceleration = -SUN_MU_AU3_DAY2 * position / radius**3
    return np.concatenate([velocity, acceleration])


def propagate_orbit(state, delta_t_days):
    """Propagate the two-body orbit with RK4 substeps of at most 0.25 day."""
    state = np.asarray(state, dtype=float).copy()
    substeps = max(1, int(np.ceil(abs(delta_t_days) / 0.25)))
    step = delta_t_days / substeps

    for _ in range(substeps):
        k1 = _state_derivative(state)
        k2 = _state_derivative(state + 0.5 * step * k1)
        k3 = _state_derivative(state + 0.5 * step * k2)
        k4 = _state_derivative(state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return state


def process_noise_covariance(delta_t_days):
    """Tiny model-error random walk, scaled by the irregular time interval."""
    diffusion_std_per_sqrt_day = np.array([1e-7, 1e-7, 2e-8, 2e-8])
    return np.diag(
        diffusion_std_per_sqrt_day**2 * max(delta_t_days, 1e-12)
    )


def earth_position(time_days):
    """Known circular 1 AU Earth orbit in the same heliocentric plane."""
    mean_motion = np.sqrt(SUN_MU_AU3_DAY2)
    angle = mean_motion * time_days
    return np.array([np.cos(angle), np.sin(angle)])


def make_line_of_sight_observation_func(time_days):
    observer = earth_position(time_days)

    def observe_line_of_sight(state):
        delta = np.asarray(state)[POSITION] - observer
        return delta / np.linalg.norm(delta)

    return observe_line_of_sight


def _rotate_unit_vector(vector, angle_rad):
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ]
    )


def _unit_vector_covariance(angle_std_rad):
    vector_std = np.sin(angle_std_rad)
    return vector_std**2 * np.eye(2)


def truth_state_at_time(time_days):
    return propagate_orbit(initial_truth_state(), float(time_days))


def _jittered_observation_times(rng):
    times = []
    current_time = 0.0
    while True:
        current_time += OBSERVATION_INTERVAL_DAYS * rng.uniform(
            1.0 - TIMING_JITTER,
            1.0 + TIMING_JITTER,
        )
        if current_time >= DURATION_DAYS:
            break
        if OBSERVATION_GAP_START_DAY < current_time < OBSERVATION_GAP_END_DAY:
            continue
        times.append(current_time)
    return np.asarray(times)


def make_observations(seed=42):
    """Generate sparse telescope line-of-sight measurements over 180 days."""
    rng = np.random.default_rng(seed)
    times = _jittered_observation_times(rng)
    measurements = []

    for time_days in times:
        truth = truth_state_at_time(time_days)
        observation_func = make_line_of_sight_observation_func(time_days)
        true_direction = observation_func(truth)
        noisy_direction = _rotate_unit_vector(
            true_direction,
            rng.normal(0.0, ANGLE_STD_RAD),
        )
        measurements.append(noisy_direction)

    return times, np.asarray(measurements)


def initial_distribution():
    """Deliberately poor initial orbit guess with broad uncertainty."""
    truth = initial_truth_state()
    initial_mean = truth + np.array([0.05, -0.04, 0.0015, -0.0012])
    initial_covariance = np.diag(
        [0.08**2, 0.08**2, 0.003**2, 0.003**2]
    )
    return Gaussian(initial_mean, initial_covariance)


def _make_filter():
    transition_model = StateTransitionModel(
        propagate_orbit,
        process_noise_covariance(1.0),
    )
    return BayesianFilter(transition_model, initial_distribution())


def _run_filter(observation_times, measurements):
    bayes_filter = _make_filter()
    state = bayes_filter.state
    previous_time = 0.0
    filtered_states = []

    for time_days, measurement in zip(observation_times, measurements):
        delta_t_days = float(time_days) - previous_time
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_days)
        )
        predicted_state = bayes_filter.predict(
            state,
            delta_t_days,
            use_jacobian=False,
        )
        observation = Observation(
            measurement,
            _unit_vector_covariance(ANGLE_STD_RAD),
            make_line_of_sight_observation_func(time_days),
        )
        state = bayes_filter.update(
            observation,
            predicted_state,
            use_jacobian=False,
        )
        bayes_filter.set_state(state)
        filtered_states.append(state)
        previous_time = float(time_days)

    return bayes_filter, filtered_states


def _rts_smooth(bayes_filter, filtered_states, observation_times):
    """RTS smoother with process noise matched to each observation interval."""
    smoothed_states = list(filtered_states)
    current_smoothed = filtered_states[-1]

    for index in range(len(filtered_states) - 2, -1, -1):
        delta_t_days = observation_times[index + 1] - observation_times[index]
        bayes_filter.transition_model.transition_noise_covariance = (
            process_noise_covariance(delta_t_days)
        )
        predicted_state, cross_covariance = (
            bayes_filter.transition_model.predict_with_cross_covariance(
                filtered_states[index],
                delta_t_days,
                use_jacobian=False,
            )
        )
        smoother_gain = cross_covariance @ np.linalg.inv(
            predicted_state.covariance()
        )
        smoothed_mean = filtered_states[index].mean() + smoother_gain @ (
            current_smoothed.mean() - predicted_state.mean()
        )
        smoothed_covariance = filtered_states[index].covariance() + smoother_gain @ (
            current_smoothed.covariance() - predicted_state.covariance()
        ) @ smoother_gain.T
        current_smoothed = Gaussian(smoothed_mean, smoothed_covariance)
        smoothed_states[index] = current_smoothed

    return smoothed_states


def _state_array(states):
    return np.asarray([state.mean() for state in states])


def _covariance_array(states):
    return np.asarray([state.covariance() for state in states])


def run_asteroid_orbit_estimation(seed=42):
    """Recover an asteroid orbit from Earth-based angle measurements only."""
    observation_times, measurements = make_observations(seed)
    bayes_filter, filtered_states = _run_filter(
        observation_times,
        measurements,
    )
    smoothed_states = _rts_smooth(
        bayes_filter,
        filtered_states,
        observation_times,
    )

    filtered = _state_array(filtered_states)
    smoothed = _state_array(smoothed_states)
    truth = np.asarray(
        [truth_state_at_time(time_days) for time_days in observation_times]
    )

    position_error = np.linalg.norm(filtered[:, POSITION] - truth[:, POSITION], axis=1)
    smoothed_position_error = np.linalg.norm(
        smoothed[:, POSITION] - truth[:, POSITION],
        axis=1,
    )

    initial_elements = orbital_elements(initial_distribution().mean())
    final_elements = orbital_elements(filtered[-1])
    smoothed_final_elements = orbital_elements(smoothed[-1])
    truth_elements = orbital_elements(truth[-1])

    return {
        "times": observation_times,
        "measurements": measurements,
        "truth": truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": _covariance_array(filtered_states),
        "smoothed_covariances": _covariance_array(smoothed_states),
        "position_rmse_au": float(np.sqrt(np.mean(position_error**2))),
        "smoothed_position_rmse_au": float(
            np.sqrt(np.mean(smoothed_position_error**2))
        ),
        "final_position_error_au": float(position_error[-1]),
        "initial_elements": initial_elements,
        "final_elements": final_elements,
        "smoothed_final_elements": smoothed_final_elements,
        "truth_elements": truth_elements,
    }


def plot_asteroid_orbit(result, output_path=None):
    """Plot the recovered heliocentric orbit and orbital-element convergence."""
    import matplotlib.pyplot as plt

    dense_times = np.linspace(0.0, DURATION_DAYS, 800)
    asteroid_truth = np.asarray([truth_state_at_time(t) for t in dense_times])
    earth_truth = np.asarray([earth_position(t) for t in dense_times])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(earth_truth[:, 0], earth_truth[:, 1], label="Earth orbit")
    axes[0].plot(
        asteroid_truth[:, 0],
        asteroid_truth[:, 1],
        "--",
        label="Asteroid truth",
    )
    axes[0].plot(
        result["filtered"][:, 0],
        result["filtered"][:, 1],
        label="Filtered asteroid",
    )
    axes[0].scatter([0.0], [0.0], marker="*", s=140, label="Sun")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [AU]")
    axes[0].set_ylabel("y [AU]")
    axes[0].set_title("Angles-only heliocentric orbit recovery")
    axes[0].legend()

    semimajor_axes = []
    eccentricities = []
    for state in result["filtered"]:
        semimajor_axis, eccentricity = orbital_elements(state)
        semimajor_axes.append(semimajor_axis)
        eccentricities.append(eccentricity)

    axes[1].plot(result["times"], semimajor_axes, label="estimated a [AU]")
    axes[1].axhline(TRUE_SEMIMAJOR_AXIS_AU, linestyle="--", label="true a")
    axes[1].plot(result["times"], eccentricities, label="estimated e")
    axes[1].axhline(TRUE_ECCENTRICITY, linestyle=":", label="true e")
    axes[1].axvspan(
        OBSERVATION_GAP_START_DAY,
        OBSERVATION_GAP_END_DAY,
        alpha=0.12,
        label="no observations",
    )
    axes[1].set_xlabel("Time [days]")
    axes[1].set_title("Recovered orbital elements")
    axes[1].legend()

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure


def _print_summary(result):
    initial_a, initial_e = result["initial_elements"]
    final_a, final_e = result["final_elements"]
    truth_a, truth_e = result["truth_elements"]
    print("Angles-only asteroid orbit estimation")
    print(f"  observations: {len(result['times'])} over {DURATION_DAYS:.0f} days")
    print(
        f"  semi-major axis: {initial_a:.3f} -> {final_a:.4f} AU "
        f"(truth {truth_a:.3f})"
    )
    print(
        f"  eccentricity:    {initial_e:.3f} -> {final_e:.4f} "
        f"(truth {truth_e:.3f})"
    )
    print(
        f"  final position error: {result['final_position_error_au']:.5f} AU"
    )


if __name__ == "__main__":
    result = run_asteroid_orbit_estimation()
    _print_summary(result)
    plot_asteroid_orbit(result, "examples/figures/asteroid-orbit.png")

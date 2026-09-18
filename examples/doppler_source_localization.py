"""Localize a stationary acoustic source from Doppler frequency alone.

A receiver moves along a known path while measuring only the apparent frequency
of a stationary source. There are no range, bearing, time-of-arrival, or
amplitude measurements.

For source position `s`, receiver position `r`, receiver velocity `v`, and
unknown emitted frequency `f_0`, the moving-observer Doppler model is

`f_obs = f_0 * (1 + dot(v, (s-r)/||s-r||) / c)`.

The hidden state is simply

`[source_x, source_y, emitted_frequency]`.

The first 25 seconds are deliberately a straight constant-velocity drive. A
source reflected across that road produces exactly the same entire frequency
history, so the source location is globally ambiguous even with noiseless
measurements. Once the receiver turns, that mirror symmetry is broken and the
source location rapidly becomes observable.

The default example uses acoustic propagation because car-scale speeds produce
Doppler shifts of tens of hertz and make the effect easy to visualize. The same
geometry applies to RF Doppler with the appropriate propagation speed.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


SOURCE_X = 0
SOURCE_Y = 1
EMITTED_FREQUENCY = 2
STATE_SIZE = 3

SPEED_OF_SOUND_M_S = 343.0
TRUE_SOURCE_POSITION_M = np.array([350.0, 260.0])
TRUE_EMITTED_FREQUENCY_HZ = 725.0
TRUE_SOURCE_STATE = np.array(
    [
        TRUE_SOURCE_POSITION_M[0],
        TRUE_SOURCE_POSITION_M[1],
        TRUE_EMITTED_FREQUENCY_HZ,
    ]
)

DEFAULT_DURATION_S = 90.0
DEFAULT_OBSERVATION_RATE_HZ = 2.0
RECEIVER_SPEED_M_S = 22.0
INITIAL_RECEIVER_POSITION_M = np.array([-900.0, -420.0])
STRAIGHT_SEGMENT_END_S = 25.0
FIRST_TURN_END_S = 40.0
SECOND_TURN_START_S = 58.0
SECOND_TURN_END_S = 76.0
FREQUENCY_STD_HZ = 0.25


def _smoothstep01(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def receiver_heading_profile(time_s):
    """Known receiver heading: straight, turn north, then bend southeast."""
    time_s = np.asarray(time_s, dtype=float)
    first_turn = np.deg2rad(80.0) * _smoothstep01(
        (time_s - STRAIGHT_SEGMENT_END_S)
        / (FIRST_TURN_END_S - STRAIGHT_SEGMENT_END_S)
    )
    second_turn = np.deg2rad(-115.0) * _smoothstep01(
        (time_s - SECOND_TURN_START_S)
        / (SECOND_TURN_END_S - SECOND_TURN_START_S)
    )
    return first_turn + second_turn


def simulate_receiver_path(times):
    """Integrate the known receiver trajectory and return position and velocity."""
    times = np.asarray(times, dtype=float)
    heading = receiver_heading_profile(times)
    velocity = RECEIVER_SPEED_M_S * np.column_stack(
        [np.cos(heading), np.sin(heading)]
    )

    position = np.empty_like(velocity)
    position[0] = INITIAL_RECEIVER_POSITION_M
    for index, delta_t_s in enumerate(np.diff(times)):
        position[index + 1] = position[index] + 0.5 * delta_t_s * (
            velocity[index] + velocity[index + 1]
        )
    return position, velocity


def observe_doppler(
    state,
    receiver_position,
    receiver_velocity,
    propagation_speed=SPEED_OF_SOUND_M_S,
):
    """Predict the scalar frequency measured by the moving receiver."""
    state = np.asarray(state)
    source_position = state[:2]
    emitted_frequency_hz = state[EMITTED_FREQUENCY]

    displacement = source_position - np.asarray(receiver_position)
    distance = np.linalg.norm(displacement)
    direction_to_source = displacement / distance
    speed_toward_source = np.dot(np.asarray(receiver_velocity), direction_to_source)

    return np.array(
        [
            emitted_frequency_hz
            * (1.0 + speed_toward_source / propagation_speed)
        ]
    )


def doppler_jacobian(
    state,
    receiver_position,
    receiver_velocity,
    propagation_speed=SPEED_OF_SOUND_M_S,
):
    """Analytic Jacobian of :func:`observe_doppler`."""
    state = np.asarray(state)
    source_position = state[:2]
    emitted_frequency_hz = state[EMITTED_FREQUENCY]
    receiver_position = np.asarray(receiver_position)
    receiver_velocity = np.asarray(receiver_velocity)

    displacement = source_position - receiver_position
    distance = np.linalg.norm(displacement)
    direction_to_source = displacement / distance
    direction_jacobian = (
        np.eye(2) - np.outer(direction_to_source, direction_to_source)
    ) / distance

    speed_toward_source = np.dot(receiver_velocity, direction_to_source)
    speed_gradient = direction_jacobian @ receiver_velocity

    jacobian = np.zeros((1, STATE_SIZE))
    jacobian[0, :2] = (
        emitted_frequency_hz / propagation_speed * speed_gradient
    )
    jacobian[0, EMITTED_FREQUENCY] = (
        1.0 + speed_toward_source / propagation_speed
    )
    return jacobian


def make_doppler_model(receiver_position, receiver_velocity):
    """Close the known receiver state into one observation model."""

    def observe(state):
        return observe_doppler(
            state,
            receiver_position,
            receiver_velocity,
        )

    def jacobian(state):
        return doppler_jacobian(
            state,
            receiver_position,
            receiver_velocity,
        )

    return observe, jacobian


def propagate_source_state(state, _delta_t_s):
    """Stationary source with constant transmitted frequency."""
    return np.asarray(state).copy()


def source_transition_jacobian(_state, _delta_t_s):
    return np.eye(STATE_SIZE)


def mirror_source_across_initial_road(source_state=TRUE_SOURCE_STATE):
    """Reflect a source across the straight initial receiver path."""
    source_state = np.asarray(source_state).copy()
    road_y = INITIAL_RECEIVER_POSITION_M[1]
    source_state[SOURCE_Y] = 2.0 * road_y - source_state[SOURCE_Y]
    return source_state


def simulate_doppler_measurements(
    seed=123,
    duration_s=DEFAULT_DURATION_S,
    observation_rate_hz=DEFAULT_OBSERVATION_RATE_HZ,
):
    """Generate a known receiver path and noisy scalar Doppler measurements."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / observation_rate_hz
    times = np.arange(round(duration_s * observation_rate_hz) + 1) * delta_t_s
    receiver_positions, receiver_velocities = simulate_receiver_path(times)

    expected_frequency_hz = np.asarray(
        [
            observe_doppler(
                TRUE_SOURCE_STATE,
                receiver_position,
                receiver_velocity,
            )[0]
            for receiver_position, receiver_velocity in zip(
                receiver_positions,
                receiver_velocities,
            )
        ]
    )
    measurements_hz = expected_frequency_hz + rng.normal(
        0.0,
        FREQUENCY_STD_HZ,
        len(times),
    )

    mirror_state = mirror_source_across_initial_road()
    mirror_frequency_hz = np.asarray(
        [
            observe_doppler(
                mirror_state,
                receiver_position,
                receiver_velocity,
            )[0]
            for receiver_position, receiver_velocity in zip(
                receiver_positions,
                receiver_velocities,
            )
        ]
    )

    return {
        "times": times,
        "receiver_positions": receiver_positions,
        "receiver_velocities": receiver_velocities,
        "expected_frequency_hz": expected_frequency_hz,
        "measurements_hz": measurements_hz,
        "mirror_source_state": mirror_state,
        "mirror_frequency_hz": mirror_frequency_hz,
    }


def run_doppler_source_localization(
    seed=123,
    duration_s=DEFAULT_DURATION_S,
    observation_rate_hz=DEFAULT_OBSERVATION_RATE_HZ,
    use_jacobian=False,
):
    """Infer source position and emitted frequency from frequency measurements."""
    data = simulate_doppler_measurements(
        seed=seed,
        duration_s=duration_s,
        observation_rate_hz=observation_rate_hz,
    )
    times = data["times"]

    observations = []
    for measurement_hz, receiver_position, receiver_velocity in zip(
        data["measurements_hz"][1:],
        data["receiver_positions"][1:],
        data["receiver_velocities"][1:],
    ):
        observation_func, jacobian_func = make_doppler_model(
            receiver_position,
            receiver_velocity,
        )
        observations.append(
            Observation(
                np.array([measurement_hz]),
                np.array([[FREQUENCY_STD_HZ**2]]),
                observation_func,
                jacobian_func,
            )
        )

    transition_model = StateTransitionModel(
        propagate_source_state,
        np.diag([1e-10, 1e-10, 1e-12]),
        source_transition_jacobian,
    )
    initial_mean = np.array([100.0, 50.0, 690.0])
    initial_std = np.array([250.0, 250.0, 40.0])
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
    )

    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=use_jacobian,
    )

    filtered = np.asarray([state.mean() for state in filter_states])
    smoothed = np.asarray([state.mean() for state in smoother_states])
    filtered_covariances = np.asarray(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.asarray(
        [state.covariance() for state in smoother_states]
    )

    filtered_position_error_m = np.linalg.norm(
        filtered[:, :2] - TRUE_SOURCE_POSITION_M,
        axis=1,
    )
    smoothed_position_error_m = np.linalg.norm(
        smoothed[:, :2] - TRUE_SOURCE_POSITION_M,
        axis=1,
    )
    position_std_m = np.sqrt(
        filtered_covariances[:, SOURCE_X, SOURCE_X]
        + filtered_covariances[:, SOURCE_Y, SOURCE_Y]
    )

    straight_end_index = int(
        np.argmin(np.abs(times - STRAIGHT_SEGMENT_END_S))
    )
    post_turn_index = int(np.argmin(np.abs(times - 60.0)))

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_position_error_m": filtered_position_error_m,
        "smoothed_position_error_m": smoothed_position_error_m,
        "position_std_m": position_std_m,
        "final_source_position_m": filtered[-1, :2].copy(),
        "final_position_error_m": float(filtered_position_error_m[-1]),
        "final_emitted_frequency_hz": float(
            filtered[-1, EMITTED_FREQUENCY]
        ),
        "final_frequency_error_hz": float(
            filtered[-1, EMITTED_FREQUENCY] - TRUE_EMITTED_FREQUENCY_HZ
        ),
        "filtered_position_rmse_m": float(
            np.sqrt(np.mean(filtered_position_error_m**2))
        ),
        "smoothed_position_rmse_m": float(
            np.sqrt(np.mean(smoothed_position_error_m**2))
        ),
        "straight_end_position_error_m": float(
            filtered_position_error_m[straight_end_index]
        ),
        "post_turn_position_error_m": float(
            filtered_position_error_m[post_turn_index]
        ),
        "straight_end_position_std_m": float(
            position_std_m[straight_end_index]
        ),
        "post_turn_position_std_m": float(position_std_m[post_turn_index]),
        "initial_smoothed_position_error_m": float(
            smoothed_position_error_m[0]
        ),
        "observation_rate_hz": observation_rate_hz,
        "use_jacobian": use_jacobian,
    }


def plot_doppler_source_localization(result, output_path=None):
    """Plot path geometry, frequency-only data, and source convergence."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    times = result["times"]

    axes[0, 0].plot(
        result["receiver_positions"][:, 0],
        result["receiver_positions"][:, 1],
        label="known receiver path",
    )
    axes[0, 0].scatter(
        TRUE_SOURCE_POSITION_M[0],
        TRUE_SOURCE_POSITION_M[1],
        marker="*",
        s=150,
        label="true source",
    )
    axes[0, 0].scatter(
        result["mirror_source_state"][SOURCE_X],
        result["mirror_source_state"][SOURCE_Y],
        marker="x",
        s=90,
        label="straight-road mirror source",
    )
    axes[0, 0].plot(
        result["filtered"][:, SOURCE_X],
        result["filtered"][:, SOURCE_Y],
        alpha=0.7,
        label="online source estimate",
    )
    axes[0, 0].scatter(
        result["smoothed"][0, SOURCE_X],
        result["smoothed"][0, SOURCE_Y],
        marker="o",
        s=55,
        label="RTS source estimate",
    )
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("Only receiver motion and scalar frequency are observed")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(
        times,
        result["measurements_hz"],
        s=8,
        alpha=0.45,
        label="measured frequency",
    )
    axes[0, 1].plot(
        times,
        result["expected_frequency_hz"],
        label="true-source Doppler",
    )
    axes[0, 1].plot(
        times,
        result["mirror_frequency_hz"],
        linestyle="--",
        label="mirror-source Doppler",
    )
    axes[0, 1].axvline(
        STRAIGHT_SEGMENT_END_S,
        linestyle=":",
        label="receiver starts turning",
    )
    axes[0, 1].set_ylabel("observed frequency [Hz]")
    axes[0, 1].set_title("The turn breaks an exact straight-road ambiguity")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        times,
        result["filtered_position_error_m"],
        label="filtered source error",
    )
    axes[1, 0].plot(
        times,
        result["smoothed_position_error_m"],
        label="RTS source error",
    )
    axes[1, 0].plot(
        times,
        result["position_std_m"],
        alpha=0.8,
        label="sqrt(trace(P_position))",
    )
    axes[1, 0].axvline(STRAIGHT_SEGMENT_END_S, linestyle=":")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("metres")
    axes[1, 0].set_title("Source position becomes observable after receiver maneuver")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        result["filtered"][:, EMITTED_FREQUENCY],
        label="filtered emitted frequency",
    )
    axes[1, 1].plot(
        times,
        result["smoothed"][:, EMITTED_FREQUENCY],
        label="RTS emitted frequency",
    )
    axes[1, 1].axhline(
        TRUE_EMITTED_FREQUENCY_HZ,
        linestyle="--",
        label="true emitted frequency",
    )
    axes[1, 1].axvline(STRAIGHT_SEGMENT_END_S, linestyle=":")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("frequency [Hz]")
    axes[1, 1].set_title("The unknown carrier frequency is inferred too")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    return figure


def _print_summary(result):
    print("Doppler-only stationary source localization")
    print(
        f"  source: {result['final_source_position_m']} m "
        f"(truth {TRUE_SOURCE_POSITION_M} m)"
    )
    print(
        f"  final source position error: "
        f"{result['final_position_error_m']:.2f} m"
    )
    print(
        f"  emitted frequency: {result['final_emitted_frequency_hz']:.3f} Hz "
        f"(truth {TRUE_EMITTED_FREQUENCY_HZ:.3f} Hz)"
    )
    print(
        f"  online position error: "
        f"{result['straight_end_position_error_m']:.1f} m at end of straight "
        f"-> {result['post_turn_position_error_m']:.1f} m after maneuver"
    )
    print(
        f"  position RMSE: {result['filtered_position_rmse_m']:.1f} m filtered "
        f"-> {result['smoothed_position_rmse_m']:.2f} m RTS"
    )


if __name__ == "__main__":
    result = run_doppler_source_localization()
    path = Path(__file__).parent / "figures" / "doppler-source-localization.png"
    plot_doppler_source_localization(result, path)
    _print_summary(result)
    print(f"saved {path}")

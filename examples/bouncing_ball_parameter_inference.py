"""Track a bouncing ball while inferring drag and restitution.

Fast state:
    [height, vertical_velocity]

Slow/stationary physical parameters:
    [quadratic_drag, restitution]

Only noisy height is observed.

Between impacts:
    z_dot = v
    v_dot = -g - c_d v |v|

At the floor:
    v_plus = -e v_minus

The filter state is
    [z, v, log(c_d), logit(e)]

so drag stays positive and restitution stays inside (0, 1).

Drag is learned continuously from the airborne arcs; restitution is learned
almost entirely at impacts. This makes the example a compact hybrid-system
demonstration of fast live-state tracking plus slow parameter inference.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_M_S2 = 9.80665
TRUE_DRAG_PER_M = 0.035
TRUE_RESTITUTION = 0.78
TRUE_INITIAL_HEIGHT_M = 12.0
TRUE_INITIAL_VELOCITY_M_S = 2.0

DEFAULT_DURATION_S = 8.0
DEFAULT_RATE_HZ = 50.0
HEIGHT_NOISE_STD_M = 0.05

HEIGHT_INDEX = 0
VELOCITY_INDEX = 1
LOG_DRAG_INDEX = 2
LOGIT_RESTITUTION_INDEX = 3


def sigmoid(value):
    value = np.asarray(value, dtype=float)
    positive = value >= 0.0
    output = np.empty_like(value, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    if output.ndim == 0:
        return float(output)
    return output


def logit(probability):
    probability = np.asarray(probability, dtype=float)
    return np.log(probability / (1.0 - probability))


def bouncing_ball_derivative(state):
    """Differentiate [height, velocity, log(drag), logit(restitution)]."""
    _height, velocity, log_drag, _logit_restitution = np.asarray(state)
    drag_per_m = np.exp(
        np.clip(log_drag, np.log(1e-4), np.log(1.0))
    )
    acceleration = (
        -GRAVITY_M_S2
        - drag_per_m * velocity * abs(velocity)
    )
    return np.array([velocity, acceleration, 0.0, 0.0])


def _integrate_airborne_state(state, delta_t_s):
    """Advance continuous flight dynamics by one RK4 step."""
    state = np.asarray(state)
    stage_1 = bouncing_ball_derivative(state)
    stage_2 = bouncing_ball_derivative(
        state + 0.5 * delta_t_s * stage_1
    )
    stage_3 = bouncing_ball_derivative(
        state + 0.5 * delta_t_s * stage_2
    )
    stage_4 = bouncing_ball_derivative(
        state + delta_t_s * stage_3
    )
    return state + delta_t_s * (
        stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
    ) / 6.0


def propagate_bouncing_ball(state, delta_t_s):
    """Advance flight and apply restitution on floor impacts."""
    state = np.asarray(state).copy()
    num_substeps = 2
    substep_s = delta_t_s / num_substeps

    for _ in range(num_substeps):
        previous = state.copy()
        candidate = _integrate_airborne_state(previous, substep_s)

        if (
            previous[HEIGHT_INDEX] >= 0.0
            and candidate[HEIGHT_INDEX] < 0.0
            and candidate[VELOCITY_INDEX] < 0.0
        ):
            impact_fraction = (
                previous[HEIGHT_INDEX]
                / (
                    previous[HEIGHT_INDEX]
                    - candidate[HEIGHT_INDEX]
                )
            )
            impact_state = _integrate_airborne_state(
                previous,
                impact_fraction * substep_s,
            )
            impact_state[HEIGHT_INDEX] = 0.0
            restitution = sigmoid(
                impact_state[LOGIT_RESTITUTION_INDEX]
            )
            impact_state[VELOCITY_INDEX] = (
                -restitution * impact_state[VELOCITY_INDEX]
            )
            state = _integrate_airborne_state(
                impact_state,
                (1.0 - impact_fraction) * substep_s,
            )
            state[HEIGHT_INDEX] = max(0.0, state[HEIGHT_INDEX])
        else:
            state = candidate
            if (
                state[HEIGHT_INDEX] < 0.0
                and state[VELOCITY_INDEX] < 0.0
            ):
                state[HEIGHT_INDEX] = -state[HEIGHT_INDEX]
                restitution = sigmoid(
                    state[LOGIT_RESTITUTION_INDEX]
                )
                state[VELOCITY_INDEX] = (
                    -restitution * state[VELOCITY_INDEX]
                )

    return state


def observe_height(state):
    return np.array([np.asarray(state)[HEIGHT_INDEX]])


def simulate_bouncing_ball(
    seed=2718,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Generate bouncing truth and noisy height measurements."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    truth = np.empty((len(times), 4))
    truth[0] = np.array(
        [
            TRUE_INITIAL_HEIGHT_M,
            TRUE_INITIAL_VELOCITY_M_S,
            np.log(TRUE_DRAG_PER_M),
            logit(TRUE_RESTITUTION),
        ]
    )
    for index, step_s in enumerate(np.diff(times)):
        truth[index + 1] = propagate_bouncing_ball(
            truth[index],
            step_s,
        )

    height_measurements_m = (
        truth[:, HEIGHT_INDEX]
        + rng.normal(0.0, HEIGHT_NOISE_STD_M, size=len(times))
    )

    bounce_indices = []
    for index in range(1, len(times)):
        if (
            truth[index - 1, VELOCITY_INDEX] < 0.0
            and truth[index, VELOCITY_INDEX] > 0.0
            and truth[index, HEIGHT_INDEX] < 0.20
        ):
            bounce_indices.append(index)

    return {
        "times": times,
        "truth": truth,
        "height_measurements_m": height_measurements_m,
        "bounce_indices": np.array(bounce_indices, dtype=int),
        "rate_hz": rate_hz,
    }


def run_bouncing_ball_parameter_inference(
    seed=2718,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Track height/velocity while inferring drag and restitution."""
    data = simulate_bouncing_ball(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    observations = [
        Observation(
            np.array([measurement]),
            np.array([[HEIGHT_NOISE_STD_M**2]]),
            observe_height,
        )
        for measurement in data["height_measurements_m"][1:]
    ]

    model = StateTransitionModel(
        propagate_bouncing_ball,
        np.diag(
            [
                1.0e-5,
                2.0e-3,
                2.0e-7,
                2.0e-6,
            ]
        ),
    )

    initial_restitution_guess = 0.55
    initial_mean = np.array(
        [
            data["height_measurements_m"][0] + 0.10,
            0.0,
            np.log(0.08),
            logit(initial_restitution_guess),
        ]
    )
    initial_covariance = np.diag(
        [
            0.20**2,
            1.50**2,
            0.70**2,
            0.80**2,
        ]
    )

    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states = bayes_filter.run_synchronous(
        observations,
        data["times"],
        use_jacobian=False,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        data["times"],
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

    filtered_drag_per_m = np.exp(filtered[:, LOG_DRAG_INDEX])
    smoothed_drag_per_m = np.exp(smoothed[:, LOG_DRAG_INDEX])
    filtered_restitution = sigmoid(
        filtered[:, LOGIT_RESTITUTION_INDEX]
    )
    smoothed_restitution = sigmoid(
        smoothed[:, LOGIT_RESTITUTION_INDEX]
    )

    filtered_drag_std_per_m = (
        filtered_drag_per_m
        * np.sqrt(
            np.maximum(
                filtered_covariances[
                    :, LOG_DRAG_INDEX, LOG_DRAG_INDEX
                ],
                0.0,
            )
        )
    )
    filtered_restitution_std = (
        filtered_restitution
        * (1.0 - filtered_restitution)
        * np.sqrt(
            np.maximum(
                filtered_covariances[
                    :,
                    LOGIT_RESTITUTION_INDEX,
                    LOGIT_RESTITUTION_INDEX,
                ],
                0.0,
            )
        )
    )

    truth = data["truth"]

    def rmse(index, states):
        return float(
            np.sqrt(
                np.mean(
                    (
                        states[:, index]
                        - truth[:, index]
                    )
                    ** 2
                )
            )
        )

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_drag_per_m": filtered_drag_per_m,
        "smoothed_drag_per_m": smoothed_drag_per_m,
        "filtered_restitution": filtered_restitution,
        "smoothed_restitution": smoothed_restitution,
        "filtered_drag_std_per_m": filtered_drag_std_per_m,
        "filtered_restitution_std": filtered_restitution_std,
        "filtered_height_rmse_m": rmse(HEIGHT_INDEX, filtered),
        "smoothed_height_rmse_m": rmse(HEIGHT_INDEX, smoothed),
        "filtered_velocity_rmse_m_s": rmse(VELOCITY_INDEX, filtered),
        "smoothed_velocity_rmse_m_s": rmse(VELOCITY_INDEX, smoothed),
        "final_filtered_drag_per_m": float(
            filtered_drag_per_m[-1]
        ),
        "final_filtered_restitution": float(
            filtered_restitution[-1]
        ),
        "initial_drag_guess_per_m": 0.08,
        "initial_restitution_guess": initial_restitution_guess,
    }


def plot_bouncing_ball_parameter_inference(
    result,
    output_path=None,
):
    """Plot fast trajectory tracking and slow physical parameters."""
    import matplotlib.pyplot as plt

    times = result["times"]
    bounce_times = times[result["bounce_indices"]]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].scatter(
        times,
        result["height_measurements_m"],
        s=7,
        alpha=0.30,
        label="noisy height",
    )
    axes[0, 0].plot(
        times,
        result["truth"][:, HEIGHT_INDEX],
        linestyle="--",
        label="true height",
    )
    axes[0, 0].plot(
        times,
        result["filtered"][:, HEIGHT_INDEX],
        label="filtered height",
    )
    axes[0, 0].plot(
        times,
        result["smoothed"][:, HEIGHT_INDEX],
        label="RTS height",
    )
    for bounce_time in bounce_times:
        axes[0, 0].axvline(bounce_time, linestyle=":", alpha=0.22)
    axes[0, 0].set_ylabel("height [m]")
    axes[0, 0].set_title("Fast state: repeated ballistic arcs")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        result["truth"][:, VELOCITY_INDEX],
        linestyle="--",
        label="true velocity",
    )
    axes[0, 1].plot(
        times,
        result["filtered"][:, VELOCITY_INDEX],
        label="filtered velocity",
    )
    axes[0, 1].plot(
        times,
        result["smoothed"][:, VELOCITY_INDEX],
        label="RTS velocity",
    )
    for bounce_time in bounce_times:
        axes[0, 1].axvline(bounce_time, linestyle=":", alpha=0.22)
    axes[0, 1].set_ylabel("vertical velocity [m/s]")
    axes[0, 1].set_title("Velocity is never directly observed")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(
        TRUE_DRAG_PER_M,
        linestyle="--",
        label="true drag",
    )
    axes[1, 0].plot(
        times,
        result["filtered_drag_per_m"],
        label="filtered drag",
    )
    axes[1, 0].plot(
        times,
        result["smoothed_drag_per_m"],
        label="RTS drag",
    )
    axes[1, 0].fill_between(
        times,
        result["filtered_drag_per_m"]
        - 2.0 * result["filtered_drag_std_per_m"],
        result["filtered_drag_per_m"]
        + 2.0 * result["filtered_drag_std_per_m"],
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("quadratic drag [1/m]")
    axes[1, 0].set_title(
        "Slow parameter: flight curvature reveals drag"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(
        TRUE_RESTITUTION,
        linestyle="--",
        label="true restitution",
    )
    axes[1, 1].plot(
        times,
        result["filtered_restitution"],
        label="filtered restitution",
    )
    axes[1, 1].plot(
        times,
        result["smoothed_restitution"],
        label="RTS restitution",
    )
    axes[1, 1].fill_between(
        times,
        result["filtered_restitution"]
        - 2.0 * result["filtered_restitution_std"],
        result["filtered_restitution"]
        + 2.0 * result["filtered_restitution_std"],
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    for bounce_time in bounce_times:
        axes[1, 1].axvline(bounce_time, linestyle=":", alpha=0.22)
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("coefficient of restitution")
    axes[1, 1].set_ylim(0.35, 0.95)
    axes[1, 1].set_title(
        "Slow parameter: impacts reveal bounciness"
    )
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
    print("Bouncing ball motion + parameter inference")
    print(
        f"  height RMSE: "
        f"{100.0 * result['filtered_height_rmse_m']:.2f} cm filtered -> "
        f"{100.0 * result['smoothed_height_rmse_m']:.2f} cm RTS"
    )
    print(
        f"  velocity RMSE: "
        f"{result['filtered_velocity_rmse_m_s']:.3f} m/s filtered -> "
        f"{result['smoothed_velocity_rmse_m_s']:.3f} m/s RTS"
    )
    print(
        f"  drag: "
        f"{result['initial_drag_guess_per_m']:.3f} initial -> "
        f"{result['final_filtered_drag_per_m']:.4f} 1/m "
        f"(true {TRUE_DRAG_PER_M:.4f})"
    )
    print(
        f"  restitution: "
        f"{result['initial_restitution_guess']:.2f} initial -> "
        f"{result['final_filtered_restitution']:.4f} "
        f"(true {TRUE_RESTITUTION:.4f})"
    )


if __name__ == "__main__":
    result = run_bouncing_ball_parameter_inference()
    path = (
        Path(__file__).parent
        / "figures"
        / "bouncing-ball-parameter-inference.png"
    )
    plot_bouncing_ball_parameter_inference(result, path)
    _print_summary(result)
    print(f"saved {path}")

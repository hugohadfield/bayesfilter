"""Track pendulum motion while inferring its length and damping.

This is a deliberately simple two-timescale inference problem.

The fast state is the pendulum motion

    [theta, theta_dot]

while the slow/stationary physical parameters are

    [length, damping].

Only noisy angle is observed. Angular velocity, pendulum length, and damping
are all inferred indirectly through the nonlinear dynamics

    theta_ddot = -(g / L) sin(theta) - c theta_dot.

The filter state is

    [theta, theta_dot, log(length), log(damping)]

so the physical parameters remain positive. Length and damping receive only
tiny process noise relative to the rapidly changing motion states.

The example starts with deliberately poor parameter guesses. The first few
oscillations rapidly identify the pendulum length from its period; damping is
learned more gradually from the decay envelope. RTS smoothing then uses the
whole trajectory to improve the early motion and parameter history.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_M_S2 = 9.80665

TRUE_LENGTH_M = 0.75
TRUE_DAMPING_PER_S = 0.09
TRUE_INITIAL_ANGLE_RAD = np.deg2rad(55.0)
TRUE_INITIAL_ANGULAR_RATE_RAD_S = 0.0

DEFAULT_DURATION_S = 25.0
DEFAULT_RATE_HZ = 20.0
ANGLE_NOISE_STD_RAD = np.deg2rad(1.0)

THETA_INDEX = 0
THETA_DOT_INDEX = 1
LOG_LENGTH_INDEX = 2
LOG_DAMPING_INDEX = 3


def pendulum_derivative(state):
    """Differentiate [theta, theta_dot, log(length), log(damping)]."""
    theta, theta_dot, log_length, log_damping = np.asarray(state)

    # Clipping only protects the numerical model from pathological sigma
    # points far outside the deliberately broad initial parameter prior.
    length_m = np.exp(
        np.clip(log_length, np.log(0.20), np.log(3.0))
    )
    damping_per_s = np.exp(
        np.clip(log_damping, np.log(0.005), np.log(2.0))
    )

    theta_ddot = (
        -(GRAVITY_M_S2 / length_m) * np.sin(theta)
        - damping_per_s * theta_dot
    )
    return np.array(
        [
            theta_dot,
            theta_ddot,
            0.0,
            0.0,
        ]
    )


def propagate_pendulum_state(state, delta_t_s):
    """Advance pendulum motion and stationary parameters with RK4."""
    state = np.asarray(state)
    stage_1 = pendulum_derivative(state)
    stage_2 = pendulum_derivative(
        state + 0.5 * delta_t_s * stage_1
    )
    stage_3 = pendulum_derivative(
        state + 0.5 * delta_t_s * stage_2
    )
    stage_4 = pendulum_derivative(
        state + delta_t_s * stage_3
    )
    return state + delta_t_s * (
        stage_1
        + 2.0 * stage_2
        + 2.0 * stage_3
        + stage_4
    ) / 6.0


def observe_angle(state):
    """Observe pendulum angle only."""
    return np.array([np.asarray(state)[THETA_INDEX]])


def simulate_pendulum(
    seed=314,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Generate a deterministic damped-pendulum trajectory and noisy angles."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    truth = np.empty((len(times), 4))
    truth[0] = np.array(
        [
            TRUE_INITIAL_ANGLE_RAD,
            TRUE_INITIAL_ANGULAR_RATE_RAD_S,
            np.log(TRUE_LENGTH_M),
            np.log(TRUE_DAMPING_PER_S),
        ]
    )
    for index, step_s in enumerate(np.diff(times)):
        truth[index + 1] = propagate_pendulum_state(
            truth[index],
            step_s,
        )

    angle_measurements_rad = (
        truth[:, THETA_INDEX]
        + rng.normal(
            0.0,
            ANGLE_NOISE_STD_RAD,
            size=len(times),
        )
    )

    return {
        "times": times,
        "truth": truth,
        "angle_measurements_rad": angle_measurements_rad,
        "rate_hz": rate_hz,
    }


def run_pendulum_parameter_inference(
    seed=314,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Track motion while jointly inferring pendulum length and damping."""
    data = simulate_pendulum(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    observations = [
        Observation(
            np.array([measurement]),
            np.array([[ANGLE_NOISE_STD_RAD**2]]),
            observe_angle,
        )
        for measurement in data["angle_measurements_rad"][1:]
    ]

    # Fast motion states get appreciable process noise. The log-parameters are
    # almost constant: their process noise is several orders of magnitude
    # smaller, expressing the two-timescale structure directly.
    process_covariance = np.diag(
        [
            2.0e-6,
            8.0e-5,
            1.0e-9,
            2.0e-8,
        ]
    )
    model = StateTransitionModel(
        propagate_pendulum_state,
        process_covariance,
    )

    initial_mean = np.array(
        [
            data["angle_measurements_rad"][0] + np.deg2rad(4.0),
            0.25,
            np.log(1.30),
            np.log(0.25),
        ]
    )
    initial_covariance = np.diag(
        [
            np.deg2rad(6.0) ** 2,
            0.50**2,
            0.45**2,
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

    filtered = np.array(
        [state.mean() for state in filter_states]
    )
    smoothed = np.array(
        [state.mean() for state in smoother_states]
    )
    filtered_covariances = np.array(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.array(
        [state.covariance() for state in smoother_states]
    )

    filtered_length_m = np.exp(
        filtered[:, LOG_LENGTH_INDEX]
    )
    smoothed_length_m = np.exp(
        smoothed[:, LOG_LENGTH_INDEX]
    )
    filtered_damping_per_s = np.exp(
        filtered[:, LOG_DAMPING_INDEX]
    )
    smoothed_damping_per_s = np.exp(
        smoothed[:, LOG_DAMPING_INDEX]
    )

    # First-order conversion from log-parameter covariance to physical units.
    filtered_length_std_m = (
        filtered_length_m
        * np.sqrt(
            np.maximum(
                filtered_covariances[
                    :, LOG_LENGTH_INDEX, LOG_LENGTH_INDEX
                ],
                0.0,
            )
        )
    )
    filtered_damping_std_per_s = (
        filtered_damping_per_s
        * np.sqrt(
            np.maximum(
                filtered_covariances[
                    :, LOG_DAMPING_INDEX, LOG_DAMPING_INDEX
                ],
                0.0,
            )
        )
    )

    true_angle = data["truth"][:, THETA_INDEX]
    true_angular_rate = data["truth"][:, THETA_DOT_INDEX]

    def angle_rmse_deg(states):
        return float(
            np.rad2deg(
                np.sqrt(
                    np.mean(
                        (
                            states[:, THETA_INDEX]
                            - true_angle
                        )
                        ** 2
                    )
                )
            )
        )

    def angular_rate_rmse(states):
        return float(
            np.sqrt(
                np.mean(
                    (
                        states[:, THETA_DOT_INDEX]
                        - true_angular_rate
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
        "filtered_length_m": filtered_length_m,
        "smoothed_length_m": smoothed_length_m,
        "filtered_damping_per_s": filtered_damping_per_s,
        "smoothed_damping_per_s": smoothed_damping_per_s,
        "filtered_length_std_m": filtered_length_std_m,
        "filtered_damping_std_per_s": filtered_damping_std_per_s,
        "filtered_angle_rmse_deg": angle_rmse_deg(filtered),
        "smoothed_angle_rmse_deg": angle_rmse_deg(smoothed),
        "filtered_angular_rate_rmse_rad_s": angular_rate_rmse(
            filtered
        ),
        "smoothed_angular_rate_rmse_rad_s": angular_rate_rmse(
            smoothed
        ),
        "final_filtered_length_m": float(
            filtered_length_m[-1]
        ),
        "final_filtered_damping_per_s": float(
            filtered_damping_per_s[-1]
        ),
        "initial_length_guess_m": float(
            np.exp(initial_mean[LOG_LENGTH_INDEX])
        ),
        "initial_damping_guess_per_s": float(
            np.exp(initial_mean[LOG_DAMPING_INDEX])
        ),
    }


def plot_pendulum_parameter_inference(result, output_path=None):
    """Plot fast state tracking and slow parameter convergence."""
    import matplotlib.pyplot as plt

    times = result["times"]
    true_angle_deg = np.rad2deg(
        result["truth"][:, THETA_INDEX]
    )
    filtered_angle_deg = np.rad2deg(
        result["filtered"][:, THETA_INDEX]
    )
    smoothed_angle_deg = np.rad2deg(
        result["smoothed"][:, THETA_INDEX]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].scatter(
        times,
        np.rad2deg(result["angle_measurements_rad"]),
        s=7,
        alpha=0.35,
        label="noisy angle",
    )
    axes[0, 0].plot(
        times,
        true_angle_deg,
        linestyle="--",
        label="true angle",
    )
    axes[0, 0].plot(
        times,
        filtered_angle_deg,
        label="filtered angle",
    )
    axes[0, 0].plot(
        times,
        smoothed_angle_deg,
        label="RTS angle",
    )
    axes[0, 0].set_ylabel("angle [deg]")
    axes[0, 0].set_title("Fast state: pendulum motion")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        result["truth"][:, THETA_DOT_INDEX],
        linestyle="--",
        label="true angular rate",
    )
    axes[0, 1].plot(
        times,
        result["filtered"][:, THETA_DOT_INDEX],
        label="filtered angular rate",
    )
    axes[0, 1].plot(
        times,
        result["smoothed"][:, THETA_DOT_INDEX],
        label="RTS angular rate",
    )
    axes[0, 1].set_ylabel("angular rate [rad/s]")
    axes[0, 1].set_title(
        "Angular rate is never directly observed"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(
        TRUE_LENGTH_M,
        linestyle="--",
        label="true length",
    )
    axes[1, 0].plot(
        times,
        result["filtered_length_m"],
        label="filtered length",
    )
    axes[1, 0].plot(
        times,
        result["smoothed_length_m"],
        label="RTS length",
    )
    axes[1, 0].fill_between(
        times,
        result["filtered_length_m"]
        - 2.0 * result["filtered_length_std_m"],
        result["filtered_length_m"]
        + 2.0 * result["filtered_length_std_m"],
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("length [m]")
    axes[1, 0].set_title(
        "Slow parameter: oscillation period reveals length"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(
        TRUE_DAMPING_PER_S,
        linestyle="--",
        label="true damping",
    )
    axes[1, 1].plot(
        times,
        result["filtered_damping_per_s"],
        label="filtered damping",
    )
    axes[1, 1].plot(
        times,
        result["smoothed_damping_per_s"],
        label="RTS damping",
    )
    axes[1, 1].fill_between(
        times,
        result["filtered_damping_per_s"]
        - 2.0 * result["filtered_damping_std_per_s"],
        result["filtered_damping_per_s"]
        + 2.0 * result["filtered_damping_std_per_s"],
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("damping [1/s]")
    axes[1, 1].set_title(
        "Slow parameter: decay envelope reveals damping"
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
    print("Pendulum motion + parameter inference")
    print(
        f"  angle RMSE: "
        f"{result['filtered_angle_rmse_deg']:.3f} deg filtered -> "
        f"{result['smoothed_angle_rmse_deg']:.3f} deg RTS"
    )
    print(
        f"  length: "
        f"{result['initial_length_guess_m']:.2f} m initial -> "
        f"{result['final_filtered_length_m']:.4f} m "
        f"(true {TRUE_LENGTH_M:.4f} m)"
    )
    print(
        f"  damping: "
        f"{result['initial_damping_guess_per_s']:.2f} 1/s initial -> "
        f"{result['final_filtered_damping_per_s']:.4f} 1/s "
        f"(true {TRUE_DAMPING_PER_S:.4f} 1/s)"
    )


if __name__ == "__main__":
    result = run_pendulum_parameter_inference()
    path = (
        Path(__file__).parent
        / "figures"
        / "pendulum-parameter-inference.png"
    )
    plot_pendulum_parameter_inference(result, path)
    _print_summary(result)
    print(f"saved {path}")

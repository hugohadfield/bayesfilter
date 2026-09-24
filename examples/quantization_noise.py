"""Why quantization noise is resolution / sqrt(12), not resolution.

A finite-resolution sensor reports the centre of one quantization bin. If the
unknown location of the true value within that bin is approximately uniform,

    e_q ~ Uniform(-Delta / 2, +Delta / 2),

then

    Var(e_q) = Delta^2 / 12
    sigma_q  = Delta / sqrt(12).

A common mistake is to use the sensor resolution itself as one standard
deviation, i.e. R = Delta^2. That overstates the quantization variance by a
factor of 12.

This learning example feeds exactly the same 0.1-degree quantized encoder
measurements to two otherwise identical Kalman filters:

    wrong:   R = Delta^2
    correct: R = Delta^2 / 12.

The state is [angle, angular_rate]. The truth contains a small smooth velocity
variation while the filter uses a constant-velocity model with white angular-
acceleration process noise.

The example compares tracking error, posterior uncertainty, and normalized
innovation squared (NIS). For a well-specified scalar measurement model the
mean NIS should be near one. The deliberately over-conservative R = Delta^2
model produces NIS far below one because it claims the measurements are much
noisier than they actually are.

The uniform-noise model is an approximation: it is most appropriate when the
signal explores quantizer phase rather than remaining stuck in one or two bins.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS

from examples.linear_tracking import (
    constant_velocity_matrix,
    white_acceleration_covariance,
)


ENCODER_RESOLUTION_DEG = 0.1
ENCODER_RESOLUTION_RAD = np.deg2rad(ENCODER_RESOLUTION_DEG)

DEFAULT_DURATION_S = 30.0
DEFAULT_RATE_HZ = 10.0

TRUE_BASE_RATE_DEG_S = 0.37
TRUE_WOBBLE_AMPLITUDE_DEG = 0.08
TRUE_WOBBLE_PERIOD_S = 7.3

PROCESS_ACCEL_SPECTRAL_STD_DEG_S2 = 0.05


def quantize(value, resolution):
    """Round values to the centre of the nearest quantization bin."""
    return resolution * np.round(np.asarray(value) / resolution)


def quantization_standard_deviation(resolution):
    """Standard deviation of uniform quantization error over one bin."""
    return resolution / np.sqrt(12.0)


def simulate_quantized_encoder(
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
    initial_phase_deg=0.013,
):
    """Generate smooth angle/rate truth and a quantized angle measurement."""
    delta_t_s = 1.0 / rate_hz
    times = (
        np.arange(round(duration_s * rate_hz) + 1, dtype=float)
        * delta_t_s
    )

    base_rate = np.deg2rad(TRUE_BASE_RATE_DEG_S)
    amplitude = np.deg2rad(TRUE_WOBBLE_AMPLITUDE_DEG)
    angular_frequency = 2.0 * np.pi / TRUE_WOBBLE_PERIOD_S
    initial_phase = np.deg2rad(initial_phase_deg)

    angle = (
        initial_phase
        + base_rate * times
        + amplitude * np.sin(angular_frequency * times)
    )
    angular_rate = (
        base_rate
        + amplitude
        * angular_frequency
        * np.cos(angular_frequency * times)
    )

    measurements = quantize(
        angle[1:],
        ENCODER_RESOLUTION_RAD,
    )
    quantization_error = measurements - angle[1:]

    return {
        "times": times,
        "truth": np.column_stack((angle, angular_rate)),
        "measurements": measurements,
        "quantization_error": quantization_error,
        "delta_t_s": delta_t_s,
    }


def _run_quantization_filter(data, measurement_variance):
    """Run one linear KF/RTS and record innovation consistency statistics."""
    delta_t_s = data["delta_t_s"]

    def transition(state, step_s):
        return constant_velocity_matrix(1, step_s) @ state

    def transition_jacobian(_state, step_s):
        return constant_velocity_matrix(1, step_s)

    def observe_angle(state):
        return np.array([state[0]])

    def observation_jacobian(_state):
        return np.array([[1.0, 0.0]])

    process_covariance = white_acceleration_covariance(
        1,
        delta_t_s,
        spectral_density=np.deg2rad(
            PROCESS_ACCEL_SPECTRAL_STD_DEG_S2
        )
        ** 2,
    )
    model = StateTransitionModel(
        transition,
        process_covariance,
        transition_jacobian,
    )

    truth = data["truth"]
    initial_mean = truth[0] + np.array(
        [
            np.deg2rad(0.30),
            np.deg2rad(0.12),
        ]
    )
    initial_covariance = np.diag(
        [
            np.deg2rad(0.50) ** 2,
            np.deg2rad(0.20) ** 2,
        ]
    )

    observations = [
        Observation(
            np.array([measurement]),
            np.array([[measurement_variance]]),
            observe_angle,
            observation_jacobian,
        )
        for measurement in data["measurements"]
    ]

    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )

    filter_states = [bayes_filter.state]
    innovations = []
    innovation_variances = []
    nis = []

    for observation, step_s in zip(
        observations,
        np.diff(data["times"]),
    ):
        predicted = bayes_filter.predict(
            bayes_filter.state,
            step_s,
            use_jacobian=True,
        )
        predicted_observation = observation.predict(
            predicted,
            use_jacobian=True,
        )

        innovation = (
            observation.observation
            - observation.observation_func(predicted.mean())
        )
        innovation_variance = (
            predicted_observation.noise_covariance[0, 0]
        )

        innovations.append(float(innovation[0]))
        innovation_variances.append(
            float(innovation_variance)
        )
        nis.append(
            float(innovation[0] ** 2 / innovation_variance)
        )

        bayes_filter.update(
            observation,
            predicted,
            use_jacobian=True,
        )
        filter_states.append(bayes_filter.state)

    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        data["times"],
        use_jacobian=True,
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

    angle_error = filtered[:, 0] - truth[:, 0]
    rate_error = filtered[:, 1] - truth[:, 1]

    return {
        "measurement_variance": measurement_variance,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "innovations": np.asarray(innovations),
        "innovation_variances": np.asarray(
            innovation_variances
        ),
        "nis": np.asarray(nis),
        "mean_nis": float(np.mean(nis)),
        "angle_rmse_rad": float(
            np.sqrt(np.mean(angle_error**2))
        ),
        "rate_rmse_rad_s": float(
            np.sqrt(np.mean(rate_error**2))
        ),
        "mean_angle_sigma_rad": float(
            np.mean(
                np.sqrt(
                    filtered_covariances[:, 0, 0]
                )
            )
        ),
    }


def run_quantization_noise_comparison():
    """Compare R=Delta^2 with the uniform-bin model R=Delta^2/12."""
    data = simulate_quantized_encoder()

    wrong_variance = ENCODER_RESOLUTION_RAD**2
    uniform_variance = (
        ENCODER_RESOLUTION_RAD**2 / 12.0
    )

    wrong = _run_quantization_filter(
        data,
        wrong_variance,
    )
    uniform = _run_quantization_filter(
        data,
        uniform_variance,
    )

    empirical_quantization_std = float(
        np.std(data["quantization_error"])
    )

    return {
        **data,
        "resolution_rad": ENCODER_RESOLUTION_RAD,
        "resolution_deg": ENCODER_RESOLUTION_DEG,
        "uniform_quantization_std_rad":
            quantization_standard_deviation(
                ENCODER_RESOLUTION_RAD
            ),
        "empirical_quantization_std_rad":
            empirical_quantization_std,
        "wrong": wrong,
        "uniform": uniform,
    }


def plot_quantization_noise_comparison(
    result,
    output_path=None,
):
    """Plot quantization error and the consequences of the two R choices."""
    import matplotlib.pyplot as plt

    times = result["times"]
    truth_deg = np.rad2deg(result["truth"][:, 0])
    measurements_deg = np.rad2deg(
        result["measurements"]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    zoom = times <= 5.0
    axes[0, 0].plot(
        times[zoom],
        truth_deg[zoom],
        label="true angle",
    )
    measurement_times = times[1:]
    measurement_zoom = measurement_times <= 5.0
    axes[0, 0].step(
        measurement_times[measurement_zoom],
        measurements_deg[measurement_zoom],
        where="post",
        label="quantized encoder",
    )
    axes[0, 0].set_ylabel("angle [deg]")
    axes[0, 0].set_title(
        "A 0.1-degree encoder reports bin centres"
    )
    axes[0, 0].legend(fontsize=8)

    quantization_error_deg = np.rad2deg(
        result["quantization_error"]
    )
    axes[0, 1].plot(
        measurement_times,
        quantization_error_deg,
        label="quantization error",
    )
    half_bin_deg = 0.5 * result["resolution_deg"]
    axes[0, 1].axhline(
        half_bin_deg,
        linestyle="--",
        label="+/- half bin",
    )
    axes[0, 1].axhline(
        -half_bin_deg,
        linestyle="--",
    )
    axes[0, 1].set_ylabel("error [deg]")
    axes[0, 1].set_title(
        "Quantization error stays within +/- Delta/2"
    )
    axes[0, 1].legend(fontsize=8)

    wrong_error_deg = np.rad2deg(
        result["wrong"]["filtered"][:, 0]
        - result["truth"][:, 0]
    )
    uniform_error_deg = np.rad2deg(
        result["uniform"]["filtered"][:, 0]
        - result["truth"][:, 0]
    )
    axes[1, 0].plot(
        times,
        wrong_error_deg,
        label="R = Delta^2",
    )
    axes[1, 0].plot(
        times,
        uniform_error_deg,
        label="R = Delta^2 / 12",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("angle error [deg]")
    axes[1, 0].set_title(
        "Using the bin width as sigma underweights the sensor"
    )
    axes[1, 0].legend(fontsize=8)

    wrong_running_nis = np.cumsum(
        result["wrong"]["nis"]
    ) / np.arange(1, len(result["wrong"]["nis"]) + 1)
    uniform_running_nis = np.cumsum(
        result["uniform"]["nis"]
    ) / np.arange(
        1,
        len(result["uniform"]["nis"]) + 1,
    )

    axes[1, 1].plot(
        measurement_times,
        wrong_running_nis,
        label="R = Delta^2",
    )
    axes[1, 1].plot(
        measurement_times,
        uniform_running_nis,
        label="R = Delta^2 / 12",
    )
    axes[1, 1].axhline(
        1.0,
        linestyle="--",
        label="expected scalar mean NIS",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("running mean NIS")
    axes[1, 1].set_title(
        "Innovation statistics expose the wrong noise model"
    )
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(output_path, dpi=180)

    return figure


def _print_summary(result):
    expected_std_deg = np.rad2deg(
        result["uniform_quantization_std_rad"]
    )
    empirical_std_deg = np.rad2deg(
        result["empirical_quantization_std_rad"]
    )

    print("Quantization-noise covariance comparison")
    print(
        f"  encoder resolution Delta: "
        f"{result['resolution_deg']:.3f} deg"
    )
    print(
        f"  Delta / sqrt(12): "
        f"{expected_std_deg:.4f} deg"
    )
    print(
        f"  empirical quantization std: "
        f"{empirical_std_deg:.4f} deg"
    )
    print(
        f"  R = Delta^2 mean NIS: "
        f"{result['wrong']['mean_nis']:.3f}"
    )
    print(
        f"  R = Delta^2/12 mean NIS: "
        f"{result['uniform']['mean_nis']:.3f}"
    )
    print(
        f"  angle RMSE: "
        f"{np.rad2deg(result['wrong']['angle_rmse_rad']):.4f} deg "
        f"(Delta^2) vs "
        f"{np.rad2deg(result['uniform']['angle_rmse_rad']):.4f} deg "
        f"(Delta^2/12)"
    )


if __name__ == "__main__":
    result = run_quantization_noise_comparison()
    path = (
        Path(__file__).parent
        / "figures"
        / "quantization-noise-comparison.png"
    )
    plot_quantization_noise_comparison(
        result,
        path,
    )
    _print_summary(result)
    print(f"saved {path}")

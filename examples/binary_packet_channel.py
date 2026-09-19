"""Track a changing wireless channel using packet success/failure only.

The hidden state is a single link margin in dB:

    x = [m]

where positive margin means the received signal is above the decoder threshold
and negative margin means it is below threshold.

Each transmitted packet produces only one bit of information:

    z = 1   packet received
    z = 0   packet lost

The success probability is a smooth threshold model

    p(success | m) = sigmoid(m / s),

so the observation is nonlinear even though the state dynamics are just a
random walk.

This example deliberately uses no RSSI, CSI, SNR estimate, or channel sounding.
The filter reconstructs a continuous latent channel-quality trajectory from a
binary packet stream.

Because BayesFilter currently represents observations with Gaussian residuals,
the Bernoulli outcome is handled as an assumed-Gaussian measurement of the
success probability. This is an approximation to a true Bernoulli likelihood,
but it is useful for demonstrating nonlinear binary observations with the
current API.

The scenario contains slow motion, a temporary 7 dB shadowing event, and later
recovery. The most important observability feature is that binary packets are
most informative near the decoder threshold. If every packet succeeds or every
packet fails, the filter mostly learns a bound on channel quality; around 50%
success, the same 0/1 observations strongly constrain the latent margin.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


DEFAULT_RATE_HZ = 20.0
DEFAULT_DURATION_S = 40.0

SUCCESS_TRANSITION_WIDTH_DB = 1.3
PROCESS_STD_DB_PER_PACKET = np.sqrt(0.3)
BINARY_OBSERVATION_VARIANCE = 0.10
INITIAL_MARGIN_DB = 4.0
INITIAL_MARGIN_STD_DB = 4.0


def sigmoid(value):
    """Numerically stable logistic function."""
    value = np.asarray(value, dtype=float)
    positive = value >= 0.0
    output = np.empty_like(value, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    if output.ndim == 0:
        return float(output)
    return output


def packet_success_probability(link_margin_db):
    """Probability that a packet is decoded at a given link margin."""
    return sigmoid(
        np.asarray(link_margin_db) / SUCCESS_TRANSITION_WIDTH_DB
    )


def observe_packet_success(state):
    """Predict the Bernoulli success probability from latent link margin."""
    return np.array([packet_success_probability(state[0])])


def packet_success_jacobian(state):
    """Jacobian of success probability with respect to link margin."""
    probability = packet_success_probability(state[0])
    derivative = (
        probability
        * (1.0 - probability)
        / SUCCESS_TRANSITION_WIDTH_DB
    )
    return np.array([[derivative]])


def propagate_link_margin(state, _delta_t_s):
    """Model channel quality as a slowly varying random walk."""
    return np.asarray(state).copy()


def link_margin_jacobian(_state, _delta_t_s):
    return np.eye(1)


def true_link_margin_db(times):
    """Deterministic channel trajectory with motion and temporary shadowing."""
    times = np.asarray(times)

    # Slowly move away until 18 s, then return.
    baseline = np.where(
        times < 18.0,
        7.0 - 0.22 * times,
        3.04 + 0.32 * (times - 18.0),
    )

    # A smooth 7 dB obstruction starts around 11.5 s and clears near 24 s.
    shadow_on = 0.5 * (1.0 + np.tanh((times - 11.5) / 0.5))
    shadow_off = 0.5 * (1.0 + np.tanh((times - 24.0) / 0.7))
    shadowing = -7.0 * (shadow_on - shadow_off)

    # Slow fading keeps the channel from being perfectly piecewise linear.
    fading = 0.7 * np.sin(0.35 * times)
    return baseline + shadowing + fading


def simulate_packet_link(
    seed=7,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Generate packet successes/losses from the hidden channel trajectory."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    margin_db = true_link_margin_db(times)
    probability = packet_success_probability(margin_db)
    packet_received = rng.binomial(1, probability).astype(float)

    return {
        "times": times,
        "true_margin_db": margin_db,
        "true_success_probability": probability,
        "packet_received": packet_received,
        "rate_hz": rate_hz,
    }


def rolling_packet_success(packet_received, window_packets=40):
    """Simple plotting baseline: trailing empirical packet-success fraction."""
    packet_received = np.asarray(packet_received, dtype=float)
    result = np.full_like(packet_received, np.nan)
    for index in range(len(packet_received)):
        start = max(0, index - window_packets + 1)
        result[index] = np.mean(packet_received[start : index + 1])
    return result


def run_binary_packet_channel(
    seed=7,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
    use_jacobian=True,
):
    """Estimate continuous channel quality from binary packet outcomes."""
    data = simulate_packet_link(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    observations = [
        Observation(
            np.array([outcome]),
            np.array([[BINARY_OBSERVATION_VARIANCE]]),
            observe_packet_success,
            packet_success_jacobian,
        )
        for outcome in data["packet_received"][1:]
    ]

    model = StateTransitionModel(
        propagate_link_margin,
        np.array([[PROCESS_STD_DB_PER_PACKET**2]]),
        link_margin_jacobian,
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(
            np.array([INITIAL_MARGIN_DB]),
            np.array([[INITIAL_MARGIN_STD_DB**2]]),
        ),
    )

    filter_states = bayes_filter.run_synchronous(
        observations,
        data["times"],
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        data["times"],
        use_jacobian=use_jacobian,
    )

    filtered_margin_db = np.array(
        [state.mean()[0] for state in filter_states]
    )
    smoothed_margin_db = np.array(
        [state.mean()[0] for state in smoother_states]
    )
    filtered_std_db = np.sqrt(
        np.array([state.covariance()[0, 0] for state in filter_states])
    )
    smoothed_std_db = np.sqrt(
        np.array([state.covariance()[0, 0] for state in smoother_states])
    )

    filtered_probability = packet_success_probability(filtered_margin_db)
    smoothed_probability = packet_success_probability(smoothed_margin_db)

    margin_error = filtered_margin_db - data["true_margin_db"]
    smoothed_margin_error = smoothed_margin_db - data["true_margin_db"]

    threshold_region = np.abs(data["true_margin_db"]) < 2.0
    saturated_region = np.abs(data["true_margin_db"]) > 5.0

    return {
        **data,
        "filtered_margin_db": filtered_margin_db,
        "smoothed_margin_db": smoothed_margin_db,
        "filtered_std_db": filtered_std_db,
        "smoothed_std_db": smoothed_std_db,
        "filtered_success_probability": filtered_probability,
        "smoothed_success_probability": smoothed_probability,
        "rolling_success_probability": rolling_packet_success(
            data["packet_received"]
        ),
        "filtered_margin_rmse_db": float(
            np.sqrt(np.mean(margin_error**2))
        ),
        "smoothed_margin_rmse_db": float(
            np.sqrt(np.mean(smoothed_margin_error**2))
        ),
        "threshold_region_mean_std_db": float(
            np.mean(filtered_std_db[threshold_region])
        ),
        "saturated_region_mean_std_db": float(
            np.mean(filtered_std_db[saturated_region])
        ),
        "packet_success_rate": float(np.mean(data["packet_received"])),
        "use_jacobian": use_jacobian,
    }


def plot_binary_packet_channel(result, output_path=None):
    """Plot latent channel reconstruction from the binary packet stream."""
    import matplotlib.pyplot as plt

    times = result["times"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        times,
        result["true_margin_db"],
        linestyle="--",
        label="true link margin",
    )
    axes[0, 0].plot(
        times,
        result["filtered_margin_db"],
        label="filtered margin",
    )
    axes[0, 0].plot(
        times,
        result["smoothed_margin_db"],
        label="RTS margin",
    )
    axes[0, 0].fill_between(
        times,
        result["filtered_margin_db"] - 2.0 * result["filtered_std_db"],
        result["filtered_margin_db"] + 2.0 * result["filtered_std_db"],
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    axes[0, 0].axhline(0.0, linestyle=":", label="decoder threshold")
    axes[0, 0].set_ylabel("link margin [dB]")
    axes[0, 0].set_title("Continuous channel quality from packet bits")
    axes[0, 0].legend(fontsize=8)

    received = result["packet_received"] > 0.5
    axes[0, 1].scatter(
        times[received],
        result["packet_received"][received],
        s=7,
        label="received",
    )
    axes[0, 1].scatter(
        times[~received],
        result["packet_received"][~received],
        s=7,
        label="lost",
    )
    axes[0, 1].plot(
        times,
        result["true_success_probability"],
        linewidth=2,
        label="true success probability",
    )
    axes[0, 1].plot(
        times,
        result["filtered_success_probability"],
        label="filtered probability",
    )
    axes[0, 1].set_ylim(-0.08, 1.08)
    axes[0, 1].set_ylabel("packet outcome / probability")
    axes[0, 1].set_title("The radio reports only 0 or 1")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        times,
        result["true_success_probability"],
        label="true probability",
    )
    axes[1, 0].plot(
        times,
        result["rolling_success_probability"],
        label="40-packet rolling fraction",
    )
    axes[1, 0].plot(
        times,
        result["filtered_success_probability"],
        label="Bayesian estimate",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("packet-success probability")
    axes[1, 0].set_title("Filtering gives a continuous link estimate")
    axes[1, 0].legend(fontsize=8)

    information_shape = (
        result["true_success_probability"]
        * (1.0 - result["true_success_probability"])
    )
    axes[1, 1].plot(
        times,
        result["filtered_std_db"],
        label="posterior margin sigma",
    )
    axes[1, 1].plot(
        times,
        information_shape / np.max(information_shape)
        * np.max(result["filtered_std_db"]),
        linestyle="--",
        label="binary information shape p(1-p), scaled",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("dB / scaled information")
    axes[1, 1].set_title("Packets are most informative near 50% success")
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
    print("Binary-packet wireless channel tracking")
    print(
        f"  packet success rate: "
        f"{100.0 * result['packet_success_rate']:.1f}%"
    )
    print(
        f"  link-margin RMSE: "
        f"{result['filtered_margin_rmse_db']:.2f} dB filtered, "
        f"{result['smoothed_margin_rmse_db']:.2f} dB RTS"
    )
    print(
        f"  posterior sigma near threshold: "
        f"{result['threshold_region_mean_std_db']:.2f} dB"
    )
    print(
        f"  posterior sigma in saturated regions: "
        f"{result['saturated_region_mean_std_db']:.2f} dB"
    )


if __name__ == "__main__":
    result = run_binary_packet_channel()
    path = Path(__file__).parent / "figures" / "binary-packet-channel.png"
    plot_binary_packet_channel(result, path)
    _print_summary(result)
    print(f"saved {path}")

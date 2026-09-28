"""Infer a slowly rising liquid level from occasional sloshing.

A rectangular tank fills continuously while it is mostly left alone. Every
roughly 15 seconds it receives a short lateral acceleration pulse that excites
its first slosh mode. A dynamic support-load measurement observes the resulting
ring-down, and the Bayesian filter infers the liquid depth from the resonance.

No observation contains liquid level, liquid mass, or total tank weight.

For a rectangular tank of length L, the small-amplitude first slosh mode is

    omega(h)^2 = g k tanh(k h),    k = pi / L.

The state is

    [q, q_dot, h, h_dot, a_tank]

where q is the dominant modal slosh displacement, h is fill depth, h_dot is
the unknown filling rate, and a_tank is a measured lateral acceleration carried
as a tightly observed input state. The slosh dynamics are

    q_ddot + 2 zeta omega(h) q_dot + omega(h)^2 q
        = -Gamma a_tank.

The tank acceleration is included in the state because StateTransitionModel
does not currently expose a separate control-input argument. Its measurement is
very accurate, so this behaves like a normal zero-order-held known input.

The support sensor reports only a dynamic reaction term proportional to q; the
quasistatic rigid-body force/weight contribution is assumed removed. Therefore
fill level is learned from the changing resonance rather than from liquid mass.

The example also demonstrates a physical observability limit. As depth grows,

    tanh(k h) -> 1,

so the natural frequency approaches sqrt(g k) and becomes insensitive to
further filling. Later shakes therefore reduce fill uncertainty much less than
early ones.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_M_S2 = 9.80665
TANK_LENGTH_M = 0.30
SLOSH_WAVENUMBER_PER_M = np.pi / TANK_LENGTH_M
SLOSH_DAMPING_RATIO = 0.055
SLOSH_ACCELERATION_COUPLING = 0.65

REACTION_FORCE_SCALE_N_PER_M = 120.0
REACTION_FORCE_NOISE_STD_N = 0.30
TANK_ACCEL_NOISE_STD_M_S2 = 0.03

DEFAULT_DURATION_S = 120.0
DEFAULT_RATE_HZ = 10.0
SHAKE_START_S = 8.0
SHAKE_INTERVAL_S = 15.0
SHAKE_HALF_DURATION_S = 1.0
SHAKE_PEAK_ACCEL_M_S2 = 1.6

TRUE_INITIAL_DEPTH_M = 0.03
TRUE_FINAL_DEPTH_M = 0.22

Q_INDEX = 0
Q_DOT_INDEX = 1
DEPTH_INDEX = 2
FILL_RATE_INDEX = 3
TANK_ACCEL_INDEX = 4
STATE_SIZE = 5


def slosh_angular_frequency_rad_s(depth_m):
    """First-mode rectangular-tank slosh frequency in rad/s."""
    depth_m = np.clip(np.asarray(depth_m), 0.005, 0.50)
    return np.sqrt(
        GRAVITY_M_S2
        * SLOSH_WAVENUMBER_PER_M
        * np.tanh(SLOSH_WAVENUMBER_PER_M * depth_m)
    )


def slosh_frequency_hz(depth_m):
    """First-mode slosh frequency in Hz."""
    return slosh_angular_frequency_rad_s(depth_m) / (2.0 * np.pi)


def slosh_frequency_sensitivity_hz_per_m(depth_m):
    """Numerical derivative df/dh, useful for visualizing observability."""
    epsilon_m = 1e-4
    return (
        slosh_frequency_hz(depth_m + epsilon_m)
        - slosh_frequency_hz(depth_m - epsilon_m)
    ) / (2.0 * epsilon_m)


def commanded_tank_acceleration(time_s):
    """Short smooth bipolar pulses that excite a broad slosh response."""
    acceleration = 0.0
    centers = np.arange(
        SHAKE_START_S,
        DEFAULT_DURATION_S,
        SHAKE_INTERVAL_S,
    )
    for center_s in centers:
        offset_s = time_s - center_s
        if abs(offset_s) <= SHAKE_HALF_DURATION_S:
            window = 0.5 * (
                1.0
                + np.cos(
                    np.pi * offset_s / SHAKE_HALF_DURATION_S
                )
            )
            acceleration += (
                SHAKE_PEAK_ACCEL_M_S2
                * np.sin(
                    np.pi * offset_s / SHAKE_HALF_DURATION_S
                )
                * window
            )
    return float(acceleration)


def slosh_state_derivative(state):
    """Differentiate [q, q_dot, h, h_dot, a_tank]."""
    q, q_dot, depth_m, fill_rate_m_s, tank_accel = np.asarray(state)
    omega = slosh_angular_frequency_rad_s(depth_m)
    q_ddot = (
        -2.0 * SLOSH_DAMPING_RATIO * omega * q_dot
        - omega**2 * q
        - SLOSH_ACCELERATION_COUPLING * tank_accel
    )
    return np.array(
        [
            q_dot,
            q_ddot,
            fill_rate_m_s,
            0.0,
            0.0,
        ]
    )


def propagate_slosh_state(state, delta_t_s):
    """Advance slosh and filling dynamics with RK4."""
    state = np.asarray(state)
    stage_1 = slosh_state_derivative(state)
    stage_2 = slosh_state_derivative(
        state + 0.5 * delta_t_s * stage_1
    )
    stage_3 = slosh_state_derivative(
        state + 0.5 * delta_t_s * stage_2
    )
    stage_4 = slosh_state_derivative(
        state + delta_t_s * stage_3
    )
    return state + delta_t_s * (
        stage_1
        + 2.0 * stage_2
        + 2.0 * stage_3
        + stage_4
    ) / 6.0


def observe_dynamic_reaction_and_acceleration(state):
    """Observe dynamic support load and lateral tank acceleration."""
    state = np.asarray(state)
    return np.array(
        [
            REACTION_FORCE_SCALE_N_PER_M * state[Q_INDEX],
            state[TANK_ACCEL_INDEX],
        ]
    )


def simulate_filling_tank(
    seed=1203,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Simulate a tank filling from 3 cm to 22 cm while periodically shaken."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    fill_rate_m_s = (
        TRUE_FINAL_DEPTH_M - TRUE_INITIAL_DEPTH_M
    ) / duration_s

    truth = np.zeros((len(times), STATE_SIZE))
    truth[0] = np.array(
        [
            0.0,
            0.0,
            TRUE_INITIAL_DEPTH_M,
            fill_rate_m_s,
            commanded_tank_acceleration(times[0]),
        ]
    )

    for index, delta_t in enumerate(np.diff(times)):
        truth[index, TANK_ACCEL_INDEX] = commanded_tank_acceleration(
            times[index]
        )
        truth[index + 1] = propagate_slosh_state(
            truth[index],
            delta_t,
        )

        # The truth tank fills at exactly the prescribed constant rate. This is
        # restated explicitly to avoid accumulating RK integration error.
        truth[index + 1, DEPTH_INDEX] = (
            TRUE_INITIAL_DEPTH_M
            + fill_rate_m_s * times[index + 1]
        )
        truth[index + 1, FILL_RATE_INDEX] = fill_rate_m_s
        truth[index + 1, TANK_ACCEL_INDEX] = (
            commanded_tank_acceleration(times[index + 1])
        )

    noiseless_measurements = np.array(
        [
            observe_dynamic_reaction_and_acceleration(state)
            for state in truth
        ]
    )
    measurements = noiseless_measurements + rng.normal(
        0.0,
        [REACTION_FORCE_NOISE_STD_N, TANK_ACCEL_NOISE_STD_M_S2],
        size=noiseless_measurements.shape,
    )

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "fill_rate_m_s": fill_rate_m_s,
        "rate_hz": rate_hz,
    }


def run_sloshing_fill_level(
    seed=1203,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Infer fill depth and fill rate from occasional slosh ring-downs."""
    data = simulate_filling_tank(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    measurement_covariance = np.diag(
        [
            REACTION_FORCE_NOISE_STD_N**2,
            TANK_ACCEL_NOISE_STD_M_S2**2,
        ]
    )
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_dynamic_reaction_and_acceleration,
        )
        for measurement in data["measurements"][1:]
    ]

    # q/q_dot process noise allows modest unmodelled fluid dynamics. Depth and
    # fill rate remain much more persistent; acceleration can change rapidly
    # because it is really a measured control input carried in the state.
    process_covariance = np.diag(
        [
            1e-8,
            3e-6,
            1e-10,
            1e-10,
            0.15**2,
        ]
    )
    model = StateTransitionModel(
        propagate_slosh_state,
        process_covariance,
    )

    initial_mean = np.array(
        [
            0.0,
            0.0,
            0.08,
            0.0,
            commanded_tank_acceleration(data["times"][0]),
        ]
    )
    initial_covariance = np.diag(
        [
            0.01**2,
            0.10**2,
            0.06**2,
            0.002**2,
            0.30**2,
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

    filtered_depth_std_m = np.sqrt(
        np.maximum(filtered_covariances[:, DEPTH_INDEX, DEPTH_INDEX], 0.0)
    )
    smoothed_depth_std_m = np.sqrt(
        np.maximum(smoothed_covariances[:, DEPTH_INDEX, DEPTH_INDEX], 0.0)
    )

    truth_depth = data["truth"][:, DEPTH_INDEX]
    filtered_depth_error = filtered[:, DEPTH_INDEX] - truth_depth
    smoothed_depth_error = smoothed[:, DEPTH_INDEX] - truth_depth

    shake_centers_s = np.arange(
        SHAKE_START_S,
        duration_s,
        SHAKE_INTERVAL_S,
    )
    sensitivity_hz_per_m = slosh_frequency_sensitivity_hz_per_m(
        truth_depth
    )

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_depth_std_m": filtered_depth_std_m,
        "smoothed_depth_std_m": smoothed_depth_std_m,
        "filtered_depth_rmse_m": float(
            np.sqrt(np.mean(filtered_depth_error**2))
        ),
        "smoothed_depth_rmse_m": float(
            np.sqrt(np.mean(smoothed_depth_error**2))
        ),
        "final_filtered_depth_m": float(filtered[-1, DEPTH_INDEX]),
        "final_smoothed_depth_m": float(smoothed[-1, DEPTH_INDEX]),
        "final_filtered_fill_rate_m_s": float(
            filtered[-1, FILL_RATE_INDEX]
        ),
        "smoothed_initial_fill_rate_m_s": float(
            smoothed[0, FILL_RATE_INDEX]
        ),
        "shake_centers_s": shake_centers_s,
        "true_slosh_frequency_hz": slosh_frequency_hz(truth_depth),
        "frequency_sensitivity_hz_per_m": sensitivity_hz_per_m,
        "deep_water_limit_hz": float(
            np.sqrt(
                GRAVITY_M_S2 * SLOSH_WAVENUMBER_PER_M
            )
            / (2.0 * np.pi)
        ),
    }


def plot_sloshing_fill_level(result, output_path=None):
    """Plot filling, excitation, resonance, and observability."""
    import matplotlib.pyplot as plt

    times = result["times"]
    truth_depth_cm = 100.0 * result["truth"][:, DEPTH_INDEX]
    filtered_depth_cm = 100.0 * result["filtered"][:, DEPTH_INDEX]
    smoothed_depth_cm = 100.0 * result["smoothed"][:, DEPTH_INDEX]
    filtered_std_cm = 100.0 * result["filtered_depth_std_m"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        times,
        truth_depth_cm,
        linestyle="--",
        label="true fill depth",
    )
    axes[0, 0].plot(
        times,
        filtered_depth_cm,
        label="filtered depth",
    )
    axes[0, 0].plot(
        times,
        smoothed_depth_cm,
        label="RTS depth",
    )
    axes[0, 0].fill_between(
        times,
        filtered_depth_cm - 2.0 * filtered_std_cm,
        filtered_depth_cm + 2.0 * filtered_std_cm,
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    for center_s in result["shake_centers_s"]:
        axes[0, 0].axvline(center_s, linestyle=":", alpha=0.25)
    axes[0, 0].set_ylabel("liquid depth [cm]")
    axes[0, 0].set_title("Fill level inferred from occasional sloshing")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        result["truth"][:, TANK_ACCEL_INDEX],
        label="tank acceleration",
    )
    axes[0, 1].plot(
        times,
        result["measurements"][:, 0],
        alpha=0.7,
        label="dynamic support reaction",
    )
    axes[0, 1].set_ylabel("acceleration [m/s²] / reaction [N]")
    axes[0, 1].set_title("Short shakes followed by free ring-down")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        truth_depth_cm,
        result["true_slosh_frequency_hz"],
        label="first slosh frequency",
    )
    axes[1, 0].axhline(
        result["deep_water_limit_hz"],
        linestyle="--",
        label="deep-water limit",
    )
    axes[1, 0].set_xlabel("liquid depth [cm]")
    axes[1, 0].set_ylabel("frequency [Hz]")
    axes[1, 0].set_title("Resonance becomes insensitive as the tank fills")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        filtered_std_cm,
        label="filtered depth sigma",
    )
    axes[1, 1].plot(
        times,
        0.1 * result["frequency_sensitivity_hz_per_m"],
        linestyle="--",
        label="frequency sensitivity df/dh, scaled",
    )
    for center_s in result["shake_centers_s"]:
        axes[1, 1].axvline(center_s, linestyle=":", alpha=0.25)
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("depth uncertainty [cm] / scaled sensitivity")
    axes[1, 1].set_title("Early shakes constrain depth much more strongly")
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
    true_final_depth_cm = 100.0 * result["truth"][-1, DEPTH_INDEX]
    true_fill_rate_cm_s = 100.0 * result["fill_rate_m_s"]

    print("Sloshing fill-level inference")
    print(
        f"  fill-depth RMSE: "
        f"{100.0 * result['filtered_depth_rmse_m']:.2f} cm filtered -> "
        f"{100.0 * result['smoothed_depth_rmse_m']:.2f} cm RTS"
    )
    print(
        f"  final depth: "
        f"{100.0 * result['final_filtered_depth_m']:.2f} cm "
        f"(true {true_final_depth_cm:.2f} cm)"
    )
    print(
        f"  final fill rate: "
        f"{100.0 * result['final_filtered_fill_rate_m_s']:.3f} cm/s "
        f"(true {true_fill_rate_cm_s:.3f} cm/s)"
    )
    print(
        f"  slosh frequency: "
        f"{slosh_frequency_hz(TRUE_INITIAL_DEPTH_M):.2f} Hz at 3 cm -> "
        f"{slosh_frequency_hz(TRUE_FINAL_DEPTH_M):.2f} Hz at 22 cm "
        f"(deep-water limit {result['deep_water_limit_hz']:.2f} Hz)"
    )


if __name__ == "__main__":
    result = run_sloshing_fill_level()
    path = Path(__file__).parent / "figures" / "sloshing-fill-level.png"
    plot_sloshing_fill_level(result, path)
    _print_summary(result)
    print(f"saved {path}")

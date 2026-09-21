"""Infer road profile while learning suspension stiffness and damping.

This example uses a quarter-car active-suspension model with two distinct
estimation timescales.

Fast hidden state:
    [sprung position/velocity, unsprung position/velocity, road height/slope]

Slow physical parameters:
    [suspension stiffness, suspension damping]

Known/measured input:
    active suspension actuator force

The full state is

    [z_s, v_s, z_u, v_u, r, r_dot, log(k_s), log(c_s), F_act]

where z_s is the sprung-mass displacement, z_u the wheel/unsprung displacement,
and r the road profile relative to the initial local road level.

Sensors provide:

- sprung-mass vertical acceleration,
- wheel-hub vertical acceleration,
- suspension travel z_s - z_u,
- active actuator force.

Road height is never measured directly.

The quarter-car dynamics are

    m_s z_s_ddot = -k_s(z_s-z_u) - c_s(v_s-v_u) + F_act

    m_u z_u_ddot =  k_s(z_s-z_u) + c_s(v_s-v_u)
                    - k_t(z_u-r) - F_act

The road input is represented as a rapidly varying Gauss-Markov state. This
provides a finite-bandwidth prior and fixes the otherwise unobservable common
vertical datum. The true simulated road does not follow this model exactly: it
contains roughness, a bump, a pothole, and several localized features.

Stiffness and damping are represented in log space and receive tiny process
noise. The vehicle therefore tracks a rapidly changing road surface while
simultaneously identifying nearly stationary suspension properties.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


SPRUNG_MASS_KG = 280.0
UNSPRUNG_MASS_KG = 42.0
TIRE_STIFFNESS_N_M = 190000.0

TRUE_SUSPENSION_STIFFNESS_N_M = 18000.0
TRUE_SUSPENSION_DAMPING_N_S_M = 1800.0

ROAD_PRIOR_FREQUENCY_HZ = 1.5
ROAD_PRIOR_DAMPING_RATIO = 0.7

DEFAULT_DURATION_S = 12.0
DEFAULT_RATE_HZ = 100.0

BODY_ACCEL_NOISE_STD_M_S2 = 0.08
WHEEL_ACCEL_NOISE_STD_M_S2 = 0.35
SUSPENSION_TRAVEL_NOISE_STD_M = 0.0005
ACTUATOR_FORCE_NOISE_STD_N = 8.0

SPRUNG_POSITION_INDEX = 0
SPRUNG_VELOCITY_INDEX = 1
UNSPRUNG_POSITION_INDEX = 2
UNSPRUNG_VELOCITY_INDEX = 3
ROAD_HEIGHT_INDEX = 4
ROAD_SLOPE_INDEX = 5
LOG_STIFFNESS_INDEX = 6
LOG_DAMPING_INDEX = 7
ACTUATOR_FORCE_INDEX = 8


def _positive_parameter(log_value, minimum, maximum):
    return np.exp(
        np.clip(
            log_value,
            np.log(minimum),
            np.log(maximum),
        )
    )


def active_actuator_force_n(time_s):
    """Known active-suspension excitation used to improve identifiability."""
    return float(
        700.0
        * np.sin(2.0 * np.pi * 0.7 * time_s)
        * (
            np.exp(-0.5 * ((time_s - 3.6) / 0.6) ** 2)
            + 0.7
            * np.exp(-0.5 * ((time_s - 7.0) / 0.7) ** 2)
        )
    )


def road_profile_m(time_s):
    """Synthetic road relative to its initial local vertical datum."""
    raw = (
        0.004 * np.sin(2.0 * np.pi * 1.1 * time_s)
        + 0.0025
        * np.sin(2.0 * np.pi * 3.2 * time_s + 0.7)
        + 0.035
        * np.exp(-0.5 * ((time_s - 2.2) / 0.12) ** 2)
        - 0.028
        * np.exp(-0.5 * ((time_s - 5.0) / 0.16) ** 2)
        + 0.022
        * np.exp(-0.5 * ((time_s - 8.1) / 0.08) ** 2)
        + 0.012
        * np.exp(-0.5 * ((time_s - 10.4) / 0.25) ** 2)
    )
    initial_raw = 0.0025 * np.sin(0.7)
    return float(raw - initial_raw)


def suspension_derivative(state):
    """Differentiate the quarter-car filter state."""
    (
        sprung_position,
        sprung_velocity,
        unsprung_position,
        unsprung_velocity,
        road_height,
        road_slope,
        log_stiffness,
        log_damping,
        actuator_force,
    ) = np.asarray(state)

    stiffness = _positive_parameter(
        log_stiffness,
        5000.0,
        60000.0,
    )
    damping = _positive_parameter(
        log_damping,
        200.0,
        6000.0,
    )

    suspension_force = (
        stiffness * (sprung_position - unsprung_position)
        + damping * (sprung_velocity - unsprung_velocity)
    )

    sprung_acceleration = (
        -suspension_force + actuator_force
    ) / SPRUNG_MASS_KG

    unsprung_acceleration = (
        suspension_force
        - TIRE_STIFFNESS_N_M * (unsprung_position - road_height)
        - actuator_force
    ) / UNSPRUNG_MASS_KG

    road_omega = 2.0 * np.pi * ROAD_PRIOR_FREQUENCY_HZ
    road_acceleration = (
        -2.0
        * ROAD_PRIOR_DAMPING_RATIO
        * road_omega
        * road_slope
        - road_omega**2 * road_height
    )

    return np.array(
        [
            sprung_velocity,
            sprung_acceleration,
            unsprung_velocity,
            unsprung_acceleration,
            road_slope,
            road_acceleration,
            0.0,
            0.0,
            0.0,
        ]
    )


def propagate_suspension_state(state, delta_t_s):
    """Advance quarter-car and road-prior dynamics with RK4."""
    state = np.asarray(state)
    stage_1 = suspension_derivative(state)
    stage_2 = suspension_derivative(
        state + 0.5 * delta_t_s * stage_1
    )
    stage_3 = suspension_derivative(
        state + 0.5 * delta_t_s * stage_2
    )
    stage_4 = suspension_derivative(
        state + delta_t_s * stage_3
    )
    return state + delta_t_s * (
        stage_1
        + 2.0 * stage_2
        + 2.0 * stage_3
        + stage_4
    ) / 6.0


def observe_suspension(state):
    """Observe accelerometers, suspension travel, and actuator force."""
    (
        sprung_position,
        sprung_velocity,
        unsprung_position,
        unsprung_velocity,
        road_height,
        _road_slope,
        log_stiffness,
        log_damping,
        actuator_force,
    ) = np.asarray(state)

    stiffness = _positive_parameter(
        log_stiffness,
        5000.0,
        60000.0,
    )
    damping = _positive_parameter(
        log_damping,
        200.0,
        6000.0,
    )

    suspension_force = (
        stiffness * (sprung_position - unsprung_position)
        + damping * (sprung_velocity - unsprung_velocity)
    )

    sprung_acceleration = (
        -suspension_force + actuator_force
    ) / SPRUNG_MASS_KG

    unsprung_acceleration = (
        suspension_force
        - TIRE_STIFFNESS_N_M * (unsprung_position - road_height)
        - actuator_force
    ) / UNSPRUNG_MASS_KG

    return np.array(
        [
            sprung_acceleration,
            unsprung_acceleration,
            sprung_position - unsprung_position,
            actuator_force,
        ]
    )


def simulate_active_suspension(
    seed=4242,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Simulate quarter-car truth over a rough road."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    road_height = np.array(
        [road_profile_m(time_s) for time_s in times]
    )
    road_slope = np.gradient(road_height, delta_t_s)
    actuator_force = np.array(
        [active_actuator_force_n(time_s) for time_s in times]
    )

    truth = np.zeros((len(times), 9))
    truth[0] = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            road_height[0],
            road_slope[0],
            np.log(TRUE_SUSPENSION_STIFFNESS_N_M),
            np.log(TRUE_SUSPENSION_DAMPING_N_S_M),
            actuator_force[0],
        ]
    )

    for index, step_s in enumerate(np.diff(times)):
        truth[index, ROAD_HEIGHT_INDEX] = road_height[index]
        truth[index, ROAD_SLOPE_INDEX] = road_slope[index]
        truth[index, ACTUATOR_FORCE_INDEX] = actuator_force[index]

        next_state = propagate_suspension_state(
            truth[index],
            step_s,
        )

        # Road and actuator are exogenous in the truth simulation. The filter
        # must reconstruct them from the sensor data and its process model.
        next_state[ROAD_HEIGHT_INDEX] = road_height[index + 1]
        next_state[ROAD_SLOPE_INDEX] = road_slope[index + 1]
        next_state[LOG_STIFFNESS_INDEX] = np.log(
            TRUE_SUSPENSION_STIFFNESS_N_M
        )
        next_state[LOG_DAMPING_INDEX] = np.log(
            TRUE_SUSPENSION_DAMPING_N_S_M
        )
        next_state[ACTUATOR_FORCE_INDEX] = actuator_force[index + 1]

        truth[index + 1] = next_state

    noiseless_measurements = np.array(
        [observe_suspension(state) for state in truth]
    )
    sensor_noise_std = np.array(
        [
            BODY_ACCEL_NOISE_STD_M_S2,
            WHEEL_ACCEL_NOISE_STD_M_S2,
            SUSPENSION_TRAVEL_NOISE_STD_M,
            ACTUATOR_FORCE_NOISE_STD_N,
        ]
    )
    measurements = noiseless_measurements + rng.normal(
        0.0,
        sensor_noise_std,
        size=noiseless_measurements.shape,
    )

    return {
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "sensor_noise_std": sensor_noise_std,
        "road_height_m": road_height,
        "actuator_force_n": actuator_force,
        "rate_hz": rate_hz,
    }


def run_active_suspension_road_profile(
    seed=4242,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Infer road profile while learning suspension stiffness and damping."""
    data = simulate_active_suspension(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    measurement_covariance = np.diag(
        data["sensor_noise_std"] ** 2
    )
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_suspension,
        )
        for measurement in data["measurements"][1:]
    ]

    process_covariance = np.diag(
        [
            1.0e-9,
            2.0e-5,
            1.0e-9,
            5.0e-4,
            1.0e-5,
            5.0e-2,
            1.0e-8,
            3.0e-8,
            120.0**2,
        ]
    )
    model = StateTransitionModel(
        propagate_suspension_state,
        process_covariance,
    )

    initial_mean = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            np.log(26000.0),
            np.log(900.0),
            0.0,
        ]
    )
    initial_covariance = np.diag(
        [
            0.005**2,
            0.20**2,
            0.005**2,
            0.20**2,
            0.002**2,
            0.08**2,
            0.50**2,
            0.70**2,
            200.0**2,
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

    filtered_road_m = filtered[:, ROAD_HEIGHT_INDEX]
    smoothed_road_m = smoothed[:, ROAD_HEIGHT_INDEX]
    filtered_road_std_m = np.sqrt(
        np.maximum(
            filtered_covariances[
                :, ROAD_HEIGHT_INDEX, ROAD_HEIGHT_INDEX
            ],
            0.0,
        )
    )

    filtered_stiffness_n_m = np.exp(
        filtered[:, LOG_STIFFNESS_INDEX]
    )
    smoothed_stiffness_n_m = np.exp(
        smoothed[:, LOG_STIFFNESS_INDEX]
    )
    filtered_damping_n_s_m = np.exp(
        filtered[:, LOG_DAMPING_INDEX]
    )
    smoothed_damping_n_s_m = np.exp(
        smoothed[:, LOG_DAMPING_INDEX]
    )

    road_truth = data["road_height_m"]

    def road_rmse(estimate):
        return float(
            np.sqrt(
                np.mean((estimate - road_truth) ** 2)
            )
        )

    def road_correlation(estimate):
        return float(
            np.corrcoef(estimate, road_truth)[0, 1]
        )

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_road_m": filtered_road_m,
        "smoothed_road_m": smoothed_road_m,
        "filtered_road_std_m": filtered_road_std_m,
        "filtered_stiffness_n_m": filtered_stiffness_n_m,
        "smoothed_stiffness_n_m": smoothed_stiffness_n_m,
        "filtered_damping_n_s_m": filtered_damping_n_s_m,
        "smoothed_damping_n_s_m": smoothed_damping_n_s_m,
        "filtered_road_rmse_m": road_rmse(filtered_road_m),
        "smoothed_road_rmse_m": road_rmse(smoothed_road_m),
        "filtered_road_correlation": road_correlation(
            filtered_road_m
        ),
        "smoothed_road_correlation": road_correlation(
            smoothed_road_m
        ),
        "final_filtered_stiffness_n_m": float(
            filtered_stiffness_n_m[-1]
        ),
        "final_filtered_damping_n_s_m": float(
            filtered_damping_n_s_m[-1]
        ),
        "initial_stiffness_guess_n_m": 26000.0,
        "initial_damping_guess_n_s_m": 900.0,
    }


def plot_active_suspension_road_profile(
    result,
    output_path=None,
):
    """Plot road reconstruction and slow suspension-parameter learning."""
    import matplotlib.pyplot as plt

    times = result["times"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    road_truth_cm = 100.0 * result["road_height_m"]
    filtered_road_cm = 100.0 * result["filtered_road_m"]
    smoothed_road_cm = 100.0 * result["smoothed_road_m"]
    road_std_cm = 100.0 * result["filtered_road_std_m"]

    axes[0, 0].plot(
        times,
        road_truth_cm,
        linestyle="--",
        label="true road",
    )
    axes[0, 0].plot(
        times,
        filtered_road_cm,
        label="filtered road",
    )
    axes[0, 0].plot(
        times,
        smoothed_road_cm,
        label="RTS road",
    )
    axes[0, 0].fill_between(
        times,
        filtered_road_cm - 2.0 * road_std_cm,
        filtered_road_cm + 2.0 * road_std_cm,
        alpha=0.15,
        label="filtered ±2 sigma",
    )
    axes[0, 0].set_ylabel("road height [cm]")
    axes[0, 0].set_title(
        "Fast latent state: road profile"
    )
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        result["measurements"][:, 0],
        alpha=0.75,
        label="body acceleration",
    )
    axes[0, 1].plot(
        times,
        result["measurements"][:, 1],
        alpha=0.55,
        label="wheel acceleration",
    )
    axes[0, 1].plot(
        times,
        result["actuator_force_n"] / 500.0,
        linestyle="--",
        label="actuator force / 500",
    )
    axes[0, 1].set_ylabel("acceleration [m/s²]")
    axes[0, 1].set_title(
        "Sensor fusion + active excitation"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(
        TRUE_SUSPENSION_STIFFNESS_N_M / 1000.0,
        linestyle="--",
        label="true stiffness",
    )
    axes[1, 0].plot(
        times,
        result["filtered_stiffness_n_m"] / 1000.0,
        label="filtered stiffness",
    )
    axes[1, 0].plot(
        times,
        result["smoothed_stiffness_n_m"] / 1000.0,
        label="RTS stiffness",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("stiffness [kN/m]")
    axes[1, 0].set_title(
        "Slow parameter: suspension stiffness"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(
        TRUE_SUSPENSION_DAMPING_N_S_M,
        linestyle="--",
        label="true damping",
    )
    axes[1, 1].plot(
        times,
        result["filtered_damping_n_s_m"],
        label="filtered damping",
    )
    axes[1, 1].plot(
        times,
        result["smoothed_damping_n_s_m"],
        label="RTS damping",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("damping [N s/m]")
    axes[1, 1].set_title(
        "Slow parameter: suspension damping"
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
    print("Active suspension: road + parameter inference")
    print(
        f"  road RMSE: "
        f"{1000.0 * result['filtered_road_rmse_m']:.2f} mm filtered -> "
        f"{1000.0 * result['smoothed_road_rmse_m']:.2f} mm RTS"
    )
    print(
        f"  road correlation: "
        f"{result['filtered_road_correlation']:.3f} filtered -> "
        f"{result['smoothed_road_correlation']:.3f} RTS"
    )
    print(
        f"  stiffness: "
        f"{result['initial_stiffness_guess_n_m'] / 1000.0:.1f} initial -> "
        f"{result['final_filtered_stiffness_n_m'] / 1000.0:.2f} kN/m "
        f"(true {TRUE_SUSPENSION_STIFFNESS_N_M / 1000.0:.2f})"
    )
    print(
        f"  damping: "
        f"{result['initial_damping_guess_n_s_m']:.0f} initial -> "
        f"{result['final_filtered_damping_n_s_m']:.1f} N s/m "
        f"(true {TRUE_SUSPENSION_DAMPING_N_S_M:.1f})"
    )


if __name__ == "__main__":
    result = run_active_suspension_road_profile()
    path = (
        Path(__file__).parent
        / "figures"
        / "active-suspension-road-profile.png"
    )
    plot_active_suspension_road_profile(result, path)
    _print_summary(result)
    print(f"saved {path}")

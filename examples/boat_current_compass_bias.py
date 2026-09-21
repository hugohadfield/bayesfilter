"""Track a boat while inferring water current and compass bias.

This example demonstrates three distinct estimation timescales:

Fast live state:
    [position, heading, speed, yaw rate]

Slowly varying environment:
    [current_x, current_y]

Stationary calibration:
    [compass_bias]

The full state is

    [x, y, psi, v_water, yaw_rate, current_x, current_y, compass_bias].

Sensors provide:

- GPS position at 1 Hz,
- speed-through-water at 10 Hz,
- yaw-rate gyro at 10 Hz,
- magnetic compass heading at 10 Hz.

The compass is represented as a 2-D unit vector to avoid angle-wrap
discontinuities. The water current is never measured directly.

Kinematics are

    p_dot = v_water [cos(psi), sin(psi)] + current
    psi_dot = yaw_rate.

On a single straight heading, compass bias and current are strongly entangled:
many combinations of heading offset and current vector explain the same ground
track. Deliberate turns change the water-relative velocity direction while the
current remains approximately common, making both the current vector and the
stationary compass bias observable.

The simulated current itself drifts slowly, so the filter must continuously
track the environment while accumulating enough geometry to calibrate the
compass.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


DEFAULT_DURATION_S = 60.0
DEFAULT_RATE_HZ = 10.0
GPS_RATE_HZ = 1.0

TRUE_COMPASS_BIAS_RAD = np.deg2rad(9.0)

GPS_POSITION_NOISE_STD_M = 0.60
WATER_SPEED_NOISE_STD_M_S = 0.05
YAW_RATE_NOISE_STD_RAD_S = np.deg2rad(0.40)
COMPASS_NOISE_STD_RAD = np.deg2rad(1.8)
COMPASS_VECTOR_NOISE_STD = np.sin(COMPASS_NOISE_STD_RAD)

POSITION_X_INDEX = 0
POSITION_Y_INDEX = 1
HEADING_INDEX = 2
WATER_SPEED_INDEX = 3
YAW_RATE_INDEX = 4
CURRENT_X_INDEX = 5
CURRENT_Y_INDEX = 6
COMPASS_BIAS_INDEX = 7


def true_water_speed_m_s(time_s):
    """Slowly varying speed through the water."""
    return 2.7 + 0.18 * np.sin(2.0 * np.pi * time_s / 22.0)


def true_yaw_rate_rad_s(time_s):
    """Piecewise turns separated by long straight segments."""
    if 12.0 <= time_s < 18.0:
        return np.deg2rad(8.0)
    if 28.0 <= time_s < 34.0:
        return np.deg2rad(-10.0)
    if 43.0 <= time_s < 50.0:
        return np.deg2rad(7.0)
    return 0.0


def true_current_m_s(time_s):
    """Slowly drifting two-dimensional water current."""
    return np.array(
        [
            0.45
            + 0.08 * np.sin(2.0 * np.pi * time_s / 75.0),
            -0.28
            + 0.06
            * np.sin(2.0 * np.pi * time_s / 52.0 + 0.7),
        ]
    )


def propagate_boat_state(state, delta_t_s):
    """Advance [x, y, psi, v, yaw_rate, current_x, current_y, bias]."""
    (
        position_x,
        position_y,
        heading,
        water_speed,
        yaw_rate,
        current_x,
        current_y,
        compass_bias,
    ) = np.asarray(state)

    return np.array(
        [
            position_x
            + delta_t_s
            * (water_speed * np.cos(heading) + current_x),
            position_y
            + delta_t_s
            * (water_speed * np.sin(heading) + current_y),
            heading + delta_t_s * yaw_rate,
            water_speed,
            yaw_rate,
            current_x,
            current_y,
            compass_bias,
        ]
    )


def boat_transition_jacobian(state, delta_t_s):
    """Analytic Jacobian of :func:`propagate_boat_state`."""
    heading = state[HEADING_INDEX]
    water_speed = state[WATER_SPEED_INDEX]

    jacobian = np.eye(8)
    jacobian[POSITION_X_INDEX, HEADING_INDEX] = (
        -delta_t_s * water_speed * np.sin(heading)
    )
    jacobian[POSITION_X_INDEX, WATER_SPEED_INDEX] = (
        delta_t_s * np.cos(heading)
    )
    jacobian[POSITION_X_INDEX, CURRENT_X_INDEX] = delta_t_s

    jacobian[POSITION_Y_INDEX, HEADING_INDEX] = (
        delta_t_s * water_speed * np.cos(heading)
    )
    jacobian[POSITION_Y_INDEX, WATER_SPEED_INDEX] = (
        delta_t_s * np.sin(heading)
    )
    jacobian[POSITION_Y_INDEX, CURRENT_Y_INDEX] = delta_t_s

    jacobian[HEADING_INDEX, YAW_RATE_INDEX] = delta_t_s
    return jacobian


def observe_navigation_without_gps(state):
    """Observe speed, yaw rate, and compass heading unit vector."""
    heading_plus_bias = (
        state[HEADING_INDEX] + state[COMPASS_BIAS_INDEX]
    )
    return np.array(
        [
            state[WATER_SPEED_INDEX],
            state[YAW_RATE_INDEX],
            np.cos(heading_plus_bias),
            np.sin(heading_plus_bias),
        ]
    )


def navigation_without_gps_jacobian(state):
    """Jacobian for high-rate speed, gyro, and compass observations."""
    heading_plus_bias = (
        state[HEADING_INDEX] + state[COMPASS_BIAS_INDEX]
    )
    sine = np.sin(heading_plus_bias)
    cosine = np.cos(heading_plus_bias)

    jacobian = np.zeros((4, 8))
    jacobian[0, WATER_SPEED_INDEX] = 1.0
    jacobian[1, YAW_RATE_INDEX] = 1.0

    jacobian[2, HEADING_INDEX] = -sine
    jacobian[2, COMPASS_BIAS_INDEX] = -sine
    jacobian[3, HEADING_INDEX] = cosine
    jacobian[3, COMPASS_BIAS_INDEX] = cosine
    return jacobian


def observe_navigation_with_gps(state):
    """Observe GPS position plus high-rate navigation sensors."""
    return np.concatenate(
        [
            state[[POSITION_X_INDEX, POSITION_Y_INDEX]],
            observe_navigation_without_gps(state),
        ]
    )


def navigation_with_gps_jacobian(state):
    """Jacobian for GPS + speed + gyro + compass observations."""
    jacobian = np.zeros((6, 8))
    jacobian[0, POSITION_X_INDEX] = 1.0
    jacobian[1, POSITION_Y_INDEX] = 1.0
    jacobian[2:, :] = navigation_without_gps_jacobian(state)
    return jacobian


def simulate_boat_navigation(
    seed=808,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Simulate a boat through slowly varying current with deliberate turns."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    truth = np.empty((len(times), 8))
    initial_current = true_current_m_s(times[0])
    truth[0] = np.array(
        [
            0.0,
            0.0,
            np.deg2rad(25.0),
            true_water_speed_m_s(times[0]),
            true_yaw_rate_rad_s(times[0]),
            initial_current[0],
            initial_current[1],
            TRUE_COMPASS_BIAS_RAD,
        ]
    )

    for index, step_s in enumerate(np.diff(times)):
        next_state = propagate_boat_state(
            truth[index],
            step_s,
        )

        next_time = times[index + 1]
        next_state[WATER_SPEED_INDEX] = true_water_speed_m_s(
            next_time
        )
        next_state[YAW_RATE_INDEX] = true_yaw_rate_rad_s(
            next_time
        )
        next_state[
            [CURRENT_X_INDEX, CURRENT_Y_INDEX]
        ] = true_current_m_s(next_time)
        next_state[COMPASS_BIAS_INDEX] = TRUE_COMPASS_BIAS_RAD
        truth[index + 1] = next_state

    gps_period_steps = round(rate_hz / GPS_RATE_HZ)
    observations = []
    gps_available = np.zeros(len(times), dtype=bool)
    measurements = []

    navigation_covariance = np.diag(
        [
            WATER_SPEED_NOISE_STD_M_S**2,
            YAW_RATE_NOISE_STD_RAD_S**2,
            COMPASS_VECTOR_NOISE_STD**2,
            COMPASS_VECTOR_NOISE_STD**2,
        ]
    )
    gps_navigation_covariance = np.diag(
        [
            GPS_POSITION_NOISE_STD_M**2,
            GPS_POSITION_NOISE_STD_M**2,
            WATER_SPEED_NOISE_STD_M_S**2,
            YAW_RATE_NOISE_STD_RAD_S**2,
            COMPASS_VECTOR_NOISE_STD**2,
            COMPASS_VECTOR_NOISE_STD**2,
        ]
    )

    for index in range(1, len(times)):
        state = truth[index]

        if index % gps_period_steps == 0:
            gps_available[index] = True
            noiseless = observe_navigation_with_gps(state)
            noise_std = np.array(
                [
                    GPS_POSITION_NOISE_STD_M,
                    GPS_POSITION_NOISE_STD_M,
                    WATER_SPEED_NOISE_STD_M_S,
                    YAW_RATE_NOISE_STD_RAD_S,
                    COMPASS_VECTOR_NOISE_STD,
                    COMPASS_VECTOR_NOISE_STD,
                ]
            )
            measurement = noiseless + rng.normal(
                0.0,
                noise_std,
            )
            observation = Observation(
                measurement,
                gps_navigation_covariance,
                observe_navigation_with_gps,
                navigation_with_gps_jacobian,
            )
        else:
            noiseless = observe_navigation_without_gps(state)
            noise_std = np.array(
                [
                    WATER_SPEED_NOISE_STD_M_S,
                    YAW_RATE_NOISE_STD_RAD_S,
                    COMPASS_VECTOR_NOISE_STD,
                    COMPASS_VECTOR_NOISE_STD,
                ]
            )
            measurement = noiseless + rng.normal(
                0.0,
                noise_std,
            )
            observation = Observation(
                measurement,
                navigation_covariance,
                observe_navigation_without_gps,
                navigation_without_gps_jacobian,
            )

        observations.append(observation)
        measurements.append(measurement)

    return {
        "times": times,
        "truth": truth,
        "observations": observations,
        "measurements": measurements,
        "gps_available": gps_available,
        "rate_hz": rate_hz,
    }


def run_boat_current_compass_bias(
    seed=808,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Track boat motion while estimating current and compass bias."""
    data = simulate_boat_navigation(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )

    process_covariance = np.diag(
        [
            2.0e-4,
            2.0e-4,
            np.deg2rad(0.03) ** 2,
            0.03**2,
            np.deg2rad(0.8) ** 2,
            0.008**2,
            0.008**2,
            np.deg2rad(0.02) ** 2,
        ]
    )
    model = StateTransitionModel(
        propagate_boat_state,
        process_covariance,
        boat_transition_jacobian,
    )

    rng = np.random.default_rng(seed + 1)

    initial_compass_angle = (
        data["truth"][0, HEADING_INDEX]
        + TRUE_COMPASS_BIAS_RAD
        + rng.normal(0.0, COMPASS_NOISE_STD_RAD)
    )
    initial_mean = np.array(
        [
            data["truth"][0, POSITION_X_INDEX]
            + rng.normal(0.0, GPS_POSITION_NOISE_STD_M),
            data["truth"][0, POSITION_Y_INDEX]
            + rng.normal(0.0, GPS_POSITION_NOISE_STD_M),
            initial_compass_angle,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    initial_covariance = np.diag(
        [
            1.0**2,
            1.0**2,
            np.deg2rad(12.0) ** 2,
            0.40**2,
            np.deg2rad(3.0) ** 2,
            0.80**2,
            0.80**2,
            np.deg2rad(15.0) ** 2,
        ]
    )

    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states = bayes_filter.run_synchronous(
        data["observations"],
        data["times"],
        use_jacobian=True,
    )
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

    truth = data["truth"]

    def vector_rmse(estimate, reference):
        return float(
            np.sqrt(
                np.mean(
                    np.sum((estimate - reference) ** 2, axis=1)
                )
            )
        )

    filtered_position_rmse_m = vector_rmse(
        filtered[:, 0:2],
        truth[:, 0:2],
    )
    smoothed_position_rmse_m = vector_rmse(
        smoothed[:, 0:2],
        truth[:, 0:2],
    )
    filtered_current_rmse_m_s = vector_rmse(
        filtered[:, 5:7],
        truth[:, 5:7],
    )
    smoothed_current_rmse_m_s = vector_rmse(
        smoothed[:, 5:7],
        truth[:, 5:7],
    )

    filtered_bias_error_rad = (
        filtered[:, COMPASS_BIAS_INDEX]
        - TRUE_COMPASS_BIAS_RAD
    )
    smoothed_bias_error_rad = (
        smoothed[:, COMPASS_BIAS_INDEX]
        - TRUE_COMPASS_BIAS_RAD
    )

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_position_rmse_m": filtered_position_rmse_m,
        "smoothed_position_rmse_m": smoothed_position_rmse_m,
        "filtered_current_rmse_m_s": filtered_current_rmse_m_s,
        "smoothed_current_rmse_m_s": smoothed_current_rmse_m_s,
        "filtered_bias_rmse_deg": float(
            np.rad2deg(
                np.sqrt(
                    np.mean(filtered_bias_error_rad**2)
                )
            )
        ),
        "smoothed_bias_rmse_deg": float(
            np.rad2deg(
                np.sqrt(
                    np.mean(smoothed_bias_error_rad**2)
                )
            )
        ),
        "final_filtered_bias_deg": float(
            np.rad2deg(
                filtered[-1, COMPASS_BIAS_INDEX]
            )
        ),
        "final_filtered_current_m_s": filtered[-1, 5:7].copy(),
        "true_final_current_m_s": truth[-1, 5:7].copy(),
        "true_compass_bias_deg": float(
            np.rad2deg(TRUE_COMPASS_BIAS_RAD)
        ),
    }


def plot_boat_current_compass_bias(result, output_path=None):
    """Plot trajectory, current, and compass-bias inference."""
    import matplotlib.pyplot as plt

    times = result["times"]
    truth = result["truth"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        truth[:, POSITION_X_INDEX],
        truth[:, POSITION_Y_INDEX],
        linestyle="--",
        label="true trajectory",
    )
    axes[0, 0].plot(
        result["filtered"][:, POSITION_X_INDEX],
        result["filtered"][:, POSITION_Y_INDEX],
        label="filtered trajectory",
    )
    axes[0, 0].plot(
        result["smoothed"][:, POSITION_X_INDEX],
        result["smoothed"][:, POSITION_Y_INDEX],
        label="RTS trajectory",
    )
    axes[0, 0].set_xlabel("east [m]")
    axes[0, 0].set_ylabel("north [m]")
    axes[0, 0].set_title("Fast state: boat trajectory")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        truth[:, CURRENT_X_INDEX],
        linestyle="--",
        label="true current east",
    )
    axes[0, 1].plot(
        times,
        truth[:, CURRENT_Y_INDEX],
        linestyle="--",
        label="true current north",
    )
    axes[0, 1].plot(
        times,
        result["filtered"][:, CURRENT_X_INDEX],
        label="filtered current east",
    )
    axes[0, 1].plot(
        times,
        result["filtered"][:, CURRENT_Y_INDEX],
        label="filtered current north",
    )
    axes[0, 1].set_ylabel("current [m/s]")
    axes[0, 1].set_title(
        "Slow environment: drifting water current"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(
        result["true_compass_bias_deg"],
        linestyle="--",
        label="true compass bias",
    )
    axes[1, 0].plot(
        times,
        np.rad2deg(
            result["filtered"][:, COMPASS_BIAS_INDEX]
        ),
        label="filtered bias",
    )
    axes[1, 0].plot(
        times,
        np.rad2deg(
            result["smoothed"][:, COMPASS_BIAS_INDEX]
        ),
        label="RTS bias",
    )
    for start_s in (12.0, 28.0, 43.0):
        axes[1, 0].axvline(
            start_s,
            linestyle=":",
            alpha=0.3,
        )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("compass bias [deg]")
    axes[1, 0].set_title(
        "Stationary calibration becomes observable through turns"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        np.rad2deg(truth[:, HEADING_INDEX]),
        linestyle="--",
        label="true heading",
    )
    axes[1, 1].plot(
        times,
        np.rad2deg(result["filtered"][:, HEADING_INDEX]),
        label="filtered heading",
    )
    axes[1, 1].plot(
        times,
        np.rad2deg(truth[:, YAW_RATE_INDEX]),
        label="yaw rate [deg/s]",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("heading / yaw rate [deg, deg/s]")
    axes[1, 1].set_title(
        "Heading changes provide the identifying geometry"
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
    print("Boat navigation: current + compass-bias inference")
    print(
        f"  position RMSE: "
        f"{result['filtered_position_rmse_m']:.2f} m filtered -> "
        f"{result['smoothed_position_rmse_m']:.2f} m RTS"
    )
    print(
        f"  current RMSE: "
        f"{result['filtered_current_rmse_m_s']:.3f} m/s filtered -> "
        f"{result['smoothed_current_rmse_m_s']:.3f} m/s RTS"
    )
    print(
        f"  compass-bias RMSE: "
        f"{result['filtered_bias_rmse_deg']:.2f} deg filtered -> "
        f"{result['smoothed_bias_rmse_deg']:.2f} deg RTS"
    )
    print(
        f"  final compass bias: "
        f"{result['final_filtered_bias_deg']:.2f} deg "
        f"(true {result['true_compass_bias_deg']:.2f} deg)"
    )


if __name__ == "__main__":
    result = run_boat_current_compass_bias()
    path = (
        Path(__file__).parent
        / "figures"
        / "boat-current-compass-bias.png"
    )
    plot_boat_current_compass_bias(result, path)
    _print_summary(result)
    print(f"saved {path}")

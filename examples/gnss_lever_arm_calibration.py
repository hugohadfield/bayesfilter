"""Calibrate a vehicle GNSS antenna lever arm from ordinary motion.

A GNSS antenna is rigidly mounted away from the vehicle body origin. Local
speed and yaw-rate sensors describe motion of the body, while GNSS measures the
antenna position instead. During straight-line motion the unknown body-frame
lever arm is weakly observable because it can largely be absorbed into the
unknown global body position. Turns rotate the lever arm in the world frame and
make the calibration observable.

The hidden state is

``[x, y, heading, speed, yaw_rate, lever_x, lever_y]``.

The GNSS observation model is

``p_gnss = p_body + R(heading) @ lever``.

The example uses a 50 Hz filter/gyroscope, 10 Hz speed measurements, and 2 Hz
GNSS fixes. The vehicle drives straight for the first 15 seconds, then performs
several left/right turns. No direct lever-arm or body-position measurements are
provided.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


X_POSITION = 0
Y_POSITION = 1
HEADING = 2
SPEED = 3
YAW_RATE = 4
LEVER_X = 5
LEVER_Y = 6
STATE_SIZE = 7

DEFAULT_DURATION_S = 90.0
DEFAULT_FILTER_RATE_HZ = 50.0
DEFAULT_SPEED_RATE_HZ = 10.0
DEFAULT_GNSS_RATE_HZ = 2.0
FIRST_TURN_START_S = 15.0
TRUE_LEVER_ARM_M = np.array([2.2, 0.75])
SPEED_STD_M_S = 0.08
GYRO_STD_RAD_S = 0.006
GNSS_STD_M = 0.25


def _smoothstep01(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smooth_window(time_s, rise_start, rise_end, fall_start, fall_end):
    time_s = np.asarray(time_s, dtype=float)
    rise = _smoothstep01((time_s - rise_start) / (rise_end - rise_start))
    fall = 1.0 - _smoothstep01((time_s - fall_start) / (fall_end - fall_start))
    return np.minimum(rise, fall)


def speed_profile(time_s):
    """Synthetic forward-speed profile in metres per second."""
    time_s = np.asarray(time_s, dtype=float)
    return (
        8.0
        + 0.8 * np.sin(2.0 * np.pi * time_s / 37.0)
        + 0.25 * np.sin(2.0 * np.pi * time_s / 9.0 + 0.4)
    )


def yaw_rate_profile(time_s):
    """Straight driving followed by left/right turns with smooth transitions."""
    return (
        0.12 * _smooth_window(time_s, 15.0, 18.0, 32.0, 35.0)
        - 0.15 * _smooth_window(time_s, 42.0, 45.0, 60.0, 63.0)
        + 0.10 * _smooth_window(time_s, 68.0, 71.0, 82.0, 85.0)
    )


def propagate_vehicle_state(state, delta_t_s):
    """Propagate planar body motion while holding the lever arm fixed."""
    x_position, y_position, heading, speed, yaw_rate, lever_x, lever_y = np.asarray(
        state
    )
    midpoint_heading = heading + 0.5 * delta_t_s * yaw_rate
    return np.array(
        [
            x_position + delta_t_s * speed * np.cos(midpoint_heading),
            y_position + delta_t_s * speed * np.sin(midpoint_heading),
            heading + delta_t_s * yaw_rate,
            speed,
            yaw_rate,
            lever_x,
            lever_y,
        ]
    )


def vehicle_transition_jacobian(state, delta_t_s):
    """Analytic Jacobian of :func:`propagate_vehicle_state`."""
    _, _, heading, speed, yaw_rate, _, _ = np.asarray(state)
    midpoint_heading = heading + 0.5 * delta_t_s * yaw_rate
    sine = np.sin(midpoint_heading)
    cosine = np.cos(midpoint_heading)

    jacobian = np.eye(STATE_SIZE)
    jacobian[X_POSITION, HEADING] = -delta_t_s * speed * sine
    jacobian[X_POSITION, SPEED] = delta_t_s * cosine
    jacobian[X_POSITION, YAW_RATE] = -0.5 * delta_t_s**2 * speed * sine
    jacobian[Y_POSITION, HEADING] = delta_t_s * speed * cosine
    jacobian[Y_POSITION, SPEED] = delta_t_s * sine
    jacobian[Y_POSITION, YAW_RATE] = 0.5 * delta_t_s**2 * speed * cosine
    jacobian[HEADING, YAW_RATE] = delta_t_s
    return jacobian


def observe_speed(state):
    """Observe local forward speed."""
    return np.array([np.asarray(state)[SPEED]])


def speed_jacobian(_state):
    jacobian = np.zeros((1, STATE_SIZE))
    jacobian[0, SPEED] = 1.0
    return jacobian


def observe_gyro(state):
    """Observe local yaw rate."""
    return np.array([np.asarray(state)[YAW_RATE]])


def gyro_jacobian(_state):
    jacobian = np.zeros((1, STATE_SIZE))
    jacobian[0, YAW_RATE] = 1.0
    return jacobian


def observe_gnss_antenna_position(state):
    """Observe the antenna position, not the vehicle body origin."""
    x_position, y_position, heading, _, _, lever_x, lever_y = np.asarray(state)
    cosine = np.cos(heading)
    sine = np.sin(heading)
    return np.array(
        [
            x_position + cosine * lever_x - sine * lever_y,
            y_position + sine * lever_x + cosine * lever_y,
        ]
    )


def gnss_antenna_position_jacobian(state):
    """Analytic Jacobian of :func:`observe_gnss_antenna_position`."""
    state = np.asarray(state)
    heading = state[HEADING]
    lever_x = state[LEVER_X]
    lever_y = state[LEVER_Y]
    cosine = np.cos(heading)
    sine = np.sin(heading)

    jacobian = np.zeros((2, STATE_SIZE))
    jacobian[0, X_POSITION] = 1.0
    jacobian[1, Y_POSITION] = 1.0
    jacobian[0, HEADING] = -sine * lever_x - cosine * lever_y
    jacobian[1, HEADING] = cosine * lever_x - sine * lever_y
    jacobian[0, LEVER_X] = cosine
    jacobian[1, LEVER_X] = sine
    jacobian[0, LEVER_Y] = -sine
    jacobian[1, LEVER_Y] = cosine
    return jacobian


def simulate_vehicle_truth(times):
    """Simulate body motion with a fixed off-centre GNSS antenna."""
    times = np.asarray(times)
    truth = np.empty((len(times), STATE_SIZE))
    truth[0] = np.array(
        [
            0.0,
            0.0,
            np.deg2rad(20.0),
            speed_profile(times[0]),
            yaw_rate_profile(times[0]),
            TRUE_LEVER_ARM_M[0],
            TRUE_LEVER_ARM_M[1],
        ]
    )

    for index, delta_t_s in enumerate(np.diff(times)):
        midpoint_time = 0.5 * (times[index] + times[index + 1])
        midpoint_state = truth[index].copy()
        midpoint_state[SPEED] = speed_profile(midpoint_time)
        midpoint_state[YAW_RATE] = yaw_rate_profile(midpoint_time)
        truth[index + 1] = propagate_vehicle_state(midpoint_state, delta_t_s)
        truth[index + 1, SPEED] = speed_profile(times[index + 1])
        truth[index + 1, YAW_RATE] = yaw_rate_profile(times[index + 1])
        truth[index + 1, LEVER_X : LEVER_Y + 1] = TRUE_LEVER_ARM_M

    return truth


def _make_sensor_data(
    truth,
    times,
    rng,
    speed_interval_steps,
    gnss_interval_steps,
):
    gyro_indices = np.arange(1, len(times))
    speed_indices = np.arange(speed_interval_steps, len(times), speed_interval_steps)
    gnss_indices = np.arange(gnss_interval_steps, len(times), gnss_interval_steps)

    gyro_measurements = truth[gyro_indices, YAW_RATE] + rng.normal(
        0.0,
        GYRO_STD_RAD_S,
        len(gyro_indices),
    )
    speed_measurements = truth[speed_indices, SPEED] + rng.normal(
        0.0,
        SPEED_STD_M_S,
        len(speed_indices),
    )
    gnss_truth = np.asarray(
        [observe_gnss_antenna_position(state) for state in truth[gnss_indices]]
    )
    gnss_measurements = gnss_truth + rng.normal(
        0.0,
        GNSS_STD_M,
        gnss_truth.shape,
    )

    events = []
    for index, measurement in zip(gyro_indices, gyro_measurements):
        events.append(
            (
                times[index],
                Observation(
                    np.array([measurement]),
                    np.array([[GYRO_STD_RAD_S**2]]),
                    observe_gyro,
                    gyro_jacobian,
                ),
            )
        )
    for index, measurement in zip(speed_indices, speed_measurements):
        events.append(
            (
                times[index],
                Observation(
                    np.array([measurement]),
                    np.array([[SPEED_STD_M_S**2]]),
                    observe_speed,
                    speed_jacobian,
                ),
            )
        )
    for index, measurement in zip(gnss_indices, gnss_measurements):
        events.append(
            (
                times[index],
                Observation(
                    measurement,
                    GNSS_STD_M**2 * np.eye(2),
                    observe_gnss_antenna_position,
                    gnss_antenna_position_jacobian,
                ),
            )
        )

    events.sort(key=lambda event: event[0])
    return {
        "observations": [event[1] for event in events],
        "event_times": np.asarray([event[0] for event in events]),
        "gyro_times": times[gyro_indices],
        "gyro_measurements": gyro_measurements,
        "speed_times": times[speed_indices],
        "speed_measurements": speed_measurements,
        "gnss_times": times[gnss_indices],
        "gnss_indices": gnss_indices,
        "gnss_truth": gnss_truth,
        "gnss_measurements": gnss_measurements,
    }


def run_gnss_lever_arm_calibration(
    seed=5,
    duration_s=DEFAULT_DURATION_S,
    filter_rate_hz=DEFAULT_FILTER_RATE_HZ,
    speed_rate_hz=DEFAULT_SPEED_RATE_HZ,
    gnss_rate_hz=DEFAULT_GNSS_RATE_HZ,
    use_jacobian=False,
):
    """Estimate the body-frame GNSS antenna lever arm from vehicle motion."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    times = np.arange(round(duration_s * filter_rate_hz) + 1) * delta_t_s
    truth = simulate_vehicle_truth(times)

    speed_interval_steps = round(filter_rate_hz / speed_rate_hz)
    gnss_interval_steps = round(filter_rate_hz / gnss_rate_hz)
    if not np.isclose(speed_interval_steps * speed_rate_hz, filter_rate_hz):
        raise ValueError("speed_rate_hz must divide filter_rate_hz")
    if not np.isclose(gnss_interval_steps * gnss_rate_hz, filter_rate_hz):
        raise ValueError("gnss_rate_hz must divide filter_rate_hz")

    sensor_data = _make_sensor_data(
        truth,
        times,
        rng,
        speed_interval_steps,
        gnss_interval_steps,
    )

    process_std_per_sqrt_s = np.array(
        [
            0.005,
            0.005,
            np.deg2rad(0.005),
            0.03,
            0.003,
            1e-5,
            1e-5,
        ]
    )
    process_std = process_std_per_sqrt_s * np.sqrt(delta_t_s)
    transition_model = StateTransitionModel(
        propagate_vehicle_state,
        np.diag(process_std**2),
        vehicle_transition_jacobian,
    )

    initial_mean = truth[0] + np.array(
        [
            2.0,
            -1.0,
            np.deg2rad(8.0),
            -0.5,
            0.0,
            -TRUE_LEVER_ARM_M[0],
            -TRUE_LEVER_ARM_M[1],
        ]
    )
    initial_std = np.array(
        [
            5.0,
            5.0,
            np.deg2rad(15.0),
            1.0,
            0.08,
            3.0,
            3.0,
        ]
    )
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
    )
    filter_states, filter_times = bayes_filter.run(
        sensor_data["observations"],
        sensor_data["event_times"],
        rate_hz=filter_rate_hz,
        use_jacobian=use_jacobian,
    )
    filter_times = np.asarray(filter_times)
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
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

    aligned_indices = np.rint(filter_times * filter_rate_hz).astype(int)
    aligned_truth = truth[aligned_indices]
    filtered_lever_error = np.linalg.norm(
        filtered[:, LEVER_X : LEVER_Y + 1] - TRUE_LEVER_ARM_M,
        axis=1,
    )
    smoothed_lever_error = np.linalg.norm(
        smoothed[:, LEVER_X : LEVER_Y + 1] - TRUE_LEVER_ARM_M,
        axis=1,
    )
    lever_uncertainty_m = np.sqrt(
        filtered_covariances[:, LEVER_X, LEVER_X]
        + filtered_covariances[:, LEVER_Y, LEVER_Y]
    )

    def body_position_rmse(states):
        return float(
            np.sqrt(
                np.mean(
                    np.sum((states[:, :2] - aligned_truth[:, :2]) ** 2, axis=1)
                )
            )
        )

    gnss_body_truth = truth[sensor_data["gnss_indices"], :2]
    naive_gnss_as_body_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (sensor_data["gnss_measurements"] - gnss_body_truth) ** 2,
                    axis=1,
                )
            )
        )
    )
    pre_turn_index = int(np.argmin(np.abs(filter_times - (FIRST_TURN_START_S - 1.0))))
    post_turn_index = int(np.argmin(np.abs(filter_times - 25.0)))

    return {
        "times": filter_times,
        "truth": aligned_truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_lever_error_m": filtered_lever_error,
        "smoothed_lever_error_m": smoothed_lever_error,
        "lever_uncertainty_m": lever_uncertainty_m,
        "gyro_times": sensor_data["gyro_times"],
        "gyro_measurements": sensor_data["gyro_measurements"],
        "speed_times": sensor_data["speed_times"],
        "speed_measurements": sensor_data["speed_measurements"],
        "gnss_times": sensor_data["gnss_times"],
        "gnss_measurements": sensor_data["gnss_measurements"],
        "gnss_truth": sensor_data["gnss_truth"],
        "final_lever_arm_m": filtered[-1, LEVER_X : LEVER_Y + 1].copy(),
        "final_lever_arm_error_m": float(filtered_lever_error[-1]),
        "filtered_body_position_rmse_m": body_position_rmse(filtered),
        "smoothed_body_position_rmse_m": body_position_rmse(smoothed),
        "naive_gnss_as_body_rmse_m": naive_gnss_as_body_rmse,
        "pre_turn_lever_uncertainty_m": float(lever_uncertainty_m[pre_turn_index]),
        "post_turn_lever_uncertainty_m": float(lever_uncertainty_m[post_turn_index]),
        "filter_rate_hz": filter_rate_hz,
        "speed_rate_hz": speed_rate_hz,
        "gnss_rate_hz": gnss_rate_hz,
        "use_jacobian": use_jacobian,
    }


def plot_gnss_lever_arm_calibration(result, output_path=None):
    """Plot trajectory, lever convergence, and turn-driven observability."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    times = result["times"]

    axes[0, 0].plot(
        result["truth"][:, X_POSITION],
        result["truth"][:, Y_POSITION],
        "--",
        label="true body origin",
    )
    axes[0, 0].scatter(
        result["gnss_measurements"][:, 0],
        result["gnss_measurements"][:, 1],
        s=10,
        alpha=0.45,
        label="GNSS antenna fixes",
    )
    axes[0, 0].plot(
        result["smoothed"][:, X_POSITION],
        result["smoothed"][:, Y_POSITION],
        label="RTS body estimate",
    )
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("GNSS observes an off-centre antenna")
    axes[0, 0].legend()

    axes[0, 1].plot(
        times,
        result["filtered"][:, LEVER_X],
        label="estimated lever x",
    )
    axes[0, 1].plot(
        times,
        result["filtered"][:, LEVER_Y],
        label="estimated lever y",
    )
    axes[0, 1].axhline(TRUE_LEVER_ARM_M[0], linestyle="--", label="true lever x")
    axes[0, 1].axhline(TRUE_LEVER_ARM_M[1], linestyle=":", label="true lever y")
    axes[0, 1].axvline(
        FIRST_TURN_START_S,
        linestyle="-.",
        label="turning begins",
    )
    axes[0, 1].set_ylabel("body-frame lever arm [m]")
    axes[0, 1].set_title("Lever arm becomes observable during turns")
    axes[0, 1].legend()

    axes[1, 0].plot(
        times,
        result["filtered_lever_error_m"],
        label="lever error",
    )
    axes[1, 0].plot(
        times,
        result["lever_uncertainty_m"],
        label="sqrt(trace(P_lever))",
    )
    axes[1, 0].axvline(FIRST_TURN_START_S, linestyle="-.")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("metres")
    axes[1, 0].set_title("Straight motion is weakly informative")
    axes[1, 0].legend()

    filtered_position_error = np.linalg.norm(
        result["filtered"][:, :2] - result["truth"][:, :2],
        axis=1,
    )
    smoothed_position_error = np.linalg.norm(
        result["smoothed"][:, :2] - result["truth"][:, :2],
        axis=1,
    )
    axes[1, 1].plot(times, filtered_position_error, label="filtered body error")
    axes[1, 1].plot(times, smoothed_position_error, label="RTS body error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("position error [m]")
    axes[1, 1].set_title("Calibrated antenna fixes recover body position")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    return figure


def _print_summary(result):
    print("GNSS antenna lever-arm calibration")
    print(
        f"  lever arm: {result['final_lever_arm_m']} m "
        f"(truth {TRUE_LEVER_ARM_M} m)"
    )
    print(f"  final lever error: {result['final_lever_arm_error_m']:.3f} m")
    print(
        f"  body position RMSE: {result['naive_gnss_as_body_rmse_m']:.2f} m "
        f"if GNSS is treated as body origin -> "
        f"{result['filtered_body_position_rmse_m']:.2f} m filtered -> "
        f"{result['smoothed_body_position_rmse_m']:.2f} m RTS"
    )
    print(
        f"  lever uncertainty: {result['pre_turn_lever_uncertainty_m']:.2f} m "
        f"before turning -> {result['post_turn_lever_uncertainty_m']:.2f} m "
        f"after the first turn"
    )


if __name__ == "__main__":
    result = run_gnss_lever_arm_calibration()
    path = Path(__file__).parent / "figures" / "gnss-lever-arm-calibration.png"
    plot_gnss_lever_arm_calibration(result, path)
    _print_summary(result)
    print(f"saved {path}")

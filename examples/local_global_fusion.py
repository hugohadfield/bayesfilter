"""Fuse fast local speed/gyro measurements with slow noisy global positions.

A planar vehicle has high-rate local sensing but only occasional absolute
position fixes. Speed is measured at 10 Hz, yaw rate at 50 Hz, and a noisy
pseudo-GPS position arrives at 1 Hz. The global sensor is deliberately removed
for twenty seconds to demonstrate dead reckoning through an outage and
re-anchoring when absolute measurements return.

The hidden state is

``[x, y, heading, speed, yaw_rate, gyro_bias]``.

The gyroscope observes ``yaw_rate + gyro_bias`` while global position fixes
constrain the curvature of the trajectory. This lets the filter estimate the
gyro bias online rather than integrating it into unbounded heading drift.
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
GYRO_BIAS = 5
STATE_SIZE = 6

DEFAULT_DURATION_S = 120.0
DEFAULT_FILTER_RATE_HZ = 50.0
DEFAULT_SPEED_RATE_HZ = 10.0
DEFAULT_GLOBAL_RATE_HZ = 1.0
GLOBAL_OUTAGE_START_S = 45.0
GLOBAL_OUTAGE_END_S = 65.0
TRUE_GYRO_BIAS_RAD_S = 0.012
SPEED_STD_M_S = 0.18
GYRO_STD_RAD_S = 0.008
GLOBAL_POSITION_STD_M = 4.0


def propagate_local_global_state(state, delta_t_s):
    """Propagate ``[x, y, heading, speed, yaw_rate, gyro_bias]``."""
    x_position, y_position, heading, speed, yaw_rate, gyro_bias = np.asarray(state)
    midpoint_heading = heading + 0.5 * delta_t_s * yaw_rate
    return np.array(
        [
            x_position + delta_t_s * speed * np.cos(midpoint_heading),
            y_position + delta_t_s * speed * np.sin(midpoint_heading),
            heading + delta_t_s * yaw_rate,
            speed,
            yaw_rate,
            gyro_bias,
        ]
    )


def local_global_jacobian(state, delta_t_s):
    """Analytic Jacobian of :func:`propagate_local_global_state`."""
    _, _, heading, speed, yaw_rate, _ = np.asarray(state)
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
    """Observe biased local yaw rate."""
    state = np.asarray(state)
    return np.array([state[YAW_RATE] + state[GYRO_BIAS]])


def gyro_jacobian(_state):
    jacobian = np.zeros((1, STATE_SIZE))
    jacobian[0, YAW_RATE] = 1.0
    jacobian[0, GYRO_BIAS] = 1.0
    return jacobian


def observe_global_position(state):
    """Observe global x-y position."""
    return np.asarray(state)[:2]


def global_position_jacobian(_state):
    jacobian = np.zeros((2, STATE_SIZE))
    jacobian[:, :2] = np.eye(2)
    return jacobian


def speed_profile(time_s):
    """Smooth varying forward speed used by the synthetic trajectory."""
    time_s = np.asarray(time_s)
    return (
        4.5
        + 0.9 * np.sin(2.0 * np.pi * time_s / 38.0)
        + 0.35 * np.sin(2.0 * np.pi * time_s / 11.0 + 0.4)
    )


def yaw_rate_profile(time_s):
    """Smooth left/right turning profile used by the synthetic trajectory."""
    time_s = np.asarray(time_s)
    return (
        0.055 * np.sin(2.0 * np.pi * time_s / 28.0)
        + 0.032 * np.sin(2.0 * np.pi * time_s / 13.0 + 1.0)
        + 0.018 * np.sin(2.0 * np.pi * time_s / 70.0 + 0.2)
    )


def simulate_local_global_truth(times):
    """Simulate a nonholonomic planar vehicle with smoothly varying motion."""
    times = np.asarray(times)
    truth = np.empty((len(times), STATE_SIZE))
    truth[0] = np.array(
        [
            0.0,
            0.0,
            0.0,
            speed_profile(times[0]),
            yaw_rate_profile(times[0]),
            TRUE_GYRO_BIAS_RAD_S,
        ]
    )

    for index, delta_t_s in enumerate(np.diff(times)):
        midpoint_time = 0.5 * (times[index] + times[index + 1])
        midpoint_state = truth[index].copy()
        midpoint_state[SPEED] = speed_profile(midpoint_time)
        midpoint_state[YAW_RATE] = yaw_rate_profile(midpoint_time)
        truth[index + 1] = propagate_local_global_state(
            midpoint_state,
            delta_t_s,
        )
        truth[index + 1, SPEED] = speed_profile(times[index + 1])
        truth[index + 1, YAW_RATE] = yaw_rate_profile(times[index + 1])
        truth[index + 1, GYRO_BIAS] = TRUE_GYRO_BIAS_RAD_S

    return truth


def angle_difference(first, second):
    """Return wrapped angular difference in radians."""
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def _make_sensor_observations(
    truth,
    times,
    rng,
    speed_interval_steps,
    global_interval_steps,
):
    gyro_times = times[1:]
    gyro_truth = truth[1:, YAW_RATE] + truth[1:, GYRO_BIAS]
    gyro_measurements = gyro_truth + rng.normal(
        0.0,
        GYRO_STD_RAD_S,
        gyro_truth.shape,
    )

    speed_indices = np.arange(speed_interval_steps, len(times), speed_interval_steps)
    speed_times = times[speed_indices]
    speed_truth = truth[speed_indices, SPEED]
    speed_measurements = speed_truth + rng.normal(
        0.0,
        SPEED_STD_M_S,
        speed_truth.shape,
    )

    all_global_indices = np.arange(
        global_interval_steps,
        len(times),
        global_interval_steps,
    )
    all_global_times = times[all_global_indices]
    available = (
        (all_global_times < GLOBAL_OUTAGE_START_S)
        | (all_global_times > GLOBAL_OUTAGE_END_S)
    )
    global_indices = all_global_indices[available]
    global_times = times[global_indices]
    global_truth = truth[global_indices, :2]
    global_measurements = global_truth + rng.normal(
        0.0,
        GLOBAL_POSITION_STD_M,
        global_truth.shape,
    )

    events = []
    for time_s, measurement in zip(gyro_times, gyro_measurements):
        events.append(
            (
                time_s,
                Observation(
                    np.array([measurement]),
                    np.array([[GYRO_STD_RAD_S**2]]),
                    observe_gyro,
                    gyro_jacobian,
                ),
            )
        )
    for time_s, measurement in zip(speed_times, speed_measurements):
        events.append(
            (
                time_s,
                Observation(
                    np.array([measurement]),
                    np.array([[SPEED_STD_M_S**2]]),
                    observe_speed,
                    speed_jacobian,
                ),
            )
        )
    for time_s, measurement in zip(global_times, global_measurements):
        events.append(
            (
                time_s,
                Observation(
                    measurement,
                    GLOBAL_POSITION_STD_M**2 * np.eye(2),
                    observe_global_position,
                    global_position_jacobian,
                ),
            )
        )

    events.sort(key=lambda event: event[0])
    return {
        "observations": [event[1] for event in events],
        "event_times": np.array([event[0] for event in events]),
        "gyro_times": gyro_times,
        "gyro_measurements": gyro_measurements,
        "speed_times": speed_times,
        "speed_measurements": speed_measurements,
        "global_times": global_times,
        "global_truth": global_truth,
        "global_measurements": global_measurements,
    }


def integrate_local_dead_reckoning(times, truth, sensor_data):
    """Integrate raw speed and biased gyro measurements with no global fixes."""
    dead_reckoning = np.empty((len(times), 3))
    dead_reckoning[0] = truth[0, :3]
    latest_speed = speed_profile(times[0])
    speed_index = 0

    for index, delta_t_s in enumerate(np.diff(times), start=1):
        while (
            speed_index < len(sensor_data["speed_times"])
            and sensor_data["speed_times"][speed_index] <= times[index]
        ):
            latest_speed = sensor_data["speed_measurements"][speed_index]
            speed_index += 1

        measured_yaw_rate = sensor_data["gyro_measurements"][index - 1]
        previous_heading = dead_reckoning[index - 1, HEADING]
        midpoint_heading = previous_heading + 0.5 * delta_t_s * measured_yaw_rate
        dead_reckoning[index] = np.array(
            [
                dead_reckoning[index - 1, X_POSITION]
                + delta_t_s * latest_speed * np.cos(midpoint_heading),
                dead_reckoning[index - 1, Y_POSITION]
                + delta_t_s * latest_speed * np.sin(midpoint_heading),
                previous_heading + delta_t_s * measured_yaw_rate,
            ]
        )

    return dead_reckoning


def run_local_global_fusion(
    seed=123,
    duration_s=DEFAULT_DURATION_S,
    filter_rate_hz=DEFAULT_FILTER_RATE_HZ,
    speed_rate_hz=DEFAULT_SPEED_RATE_HZ,
    global_rate_hz=DEFAULT_GLOBAL_RATE_HZ,
    use_jacobian=False,
):
    """Fuse local odometry-like sensing with slow global x-y position fixes."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    times = np.arange(round(duration_s * filter_rate_hz) + 1) * delta_t_s
    truth = simulate_local_global_truth(times)

    speed_interval_steps = round(filter_rate_hz / speed_rate_hz)
    global_interval_steps = round(filter_rate_hz / global_rate_hz)
    if not np.isclose(speed_interval_steps * speed_rate_hz, filter_rate_hz):
        raise ValueError("speed_rate_hz must divide filter_rate_hz")
    if not np.isclose(global_interval_steps * global_rate_hz, filter_rate_hz):
        raise ValueError("global_rate_hz must divide filter_rate_hz")

    sensor_data = _make_sensor_observations(
        truth,
        times,
        rng,
        speed_interval_steps,
        global_interval_steps,
    )

    transition_model = StateTransitionModel(
        propagate_local_global_state,
        np.diag(
            [
                0.003**2,
                0.003**2,
                np.deg2rad(0.02) ** 2,
                0.035**2,
                0.004**2,
                0.00008**2,
            ]
        ),
        local_global_jacobian,
    )
    initial_mean = truth[0] + np.array(
        [
            5.0,
            -4.0,
            np.deg2rad(8.0),
            -0.6,
            -truth[0, YAW_RATE],
            -TRUE_GYRO_BIAS_RAD_S,
        ]
    )
    initial_covariance = np.diag(
        [
            10.0**2,
            10.0**2,
            np.deg2rad(15.0) ** 2,
            1.0**2,
            0.08**2,
            0.03**2,
        ]
    )
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states, filter_times = bayes_filter.run(
        sensor_data["observations"],
        sensor_data["event_times"],
        rate_hz=filter_rate_hz,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )

    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])
    aligned_truth = truth[1 : len(filter_times) + 1]
    dead_reckoning_full = integrate_local_dead_reckoning(times, truth, sensor_data)
    dead_reckoning = dead_reckoning_full[1 : len(filter_times) + 1]

    def position_rmse(states):
        return float(
            np.sqrt(
                np.mean(
                    np.sum((states[:, :2] - aligned_truth[:, :2]) ** 2, axis=1)
                )
            )
        )

    def heading_rmse(states):
        errors = angle_difference(states[:, HEADING], aligned_truth[:, HEADING])
        return float(np.sqrt(np.mean(errors**2)))

    global_position_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (sensor_data["global_measurements"] - sensor_data["global_truth"])
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    filtered_position_error = np.linalg.norm(
        filtered[:, :2] - aligned_truth[:, :2],
        axis=1,
    )
    smoothed_position_error = np.linalg.norm(
        smoothed[:, :2] - aligned_truth[:, :2],
        axis=1,
    )
    dead_reckoning_position_error = np.linalg.norm(
        dead_reckoning[:, :2] - aligned_truth[:, :2],
        axis=1,
    )
    filter_times = np.asarray(filter_times)
    outage = (
        (filter_times >= GLOBAL_OUTAGE_START_S)
        & (filter_times <= GLOBAL_OUTAGE_END_S)
    )

    return {
        "times": filter_times,
        "truth": aligned_truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "dead_reckoning": dead_reckoning,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_position_error": filtered_position_error,
        "smoothed_position_error": smoothed_position_error,
        "dead_reckoning_position_error": dead_reckoning_position_error,
        "event_times": sensor_data["event_times"],
        "gyro_times": sensor_data["gyro_times"],
        "gyro_measurements": sensor_data["gyro_measurements"],
        "speed_times": sensor_data["speed_times"],
        "speed_measurements": sensor_data["speed_measurements"],
        "global_times": sensor_data["global_times"],
        "global_measurements": sensor_data["global_measurements"],
        "global_truth": sensor_data["global_truth"],
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "dead_reckoning_position_rmse": position_rmse(dead_reckoning),
        "global_position_rmse": global_position_rmse,
        "filtered_heading_rmse": heading_rmse(filtered),
        "smoothed_heading_rmse": heading_rmse(smoothed),
        "final_gyro_bias_rad_s": float(filtered[-1, GYRO_BIAS]),
        "outage_max_position_error_m": float(filtered_position_error[outage].max()),
        "position_covariance_trace": np.trace(
            filtered_covariances[:, :2, :2],
            axis1=1,
            axis2=2,
        ),
        "use_jacobian": use_jacobian,
        "filter_rate_hz": filter_rate_hz,
        "speed_rate_hz": speed_rate_hz,
        "global_rate_hz": global_rate_hz,
        "outage_start_s": GLOBAL_OUTAGE_START_S,
        "outage_end_s": GLOBAL_OUTAGE_END_S,
    }


def plot_local_global_fusion(result, output_path=None):
    """Plot global anchoring, dead-reckoning drift, bias, and outage uncertainty."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(
        result["truth"][:, X_POSITION],
        result["truth"][:, Y_POSITION],
        "--",
        label="truth",
    )
    axes[0, 0].plot(
        result["dead_reckoning"][:, X_POSITION],
        result["dead_reckoning"][:, Y_POSITION],
        label="local dead reckoning",
    )
    axes[0, 0].scatter(
        result["global_measurements"][:, X_POSITION],
        result["global_measurements"][:, Y_POSITION],
        s=12,
        alpha=0.55,
        label="pseudo-GPS",
    )
    axes[0, 0].plot(
        result["filtered"][:, X_POSITION],
        result["filtered"][:, Y_POSITION],
        label="filtered",
    )
    axes[0, 0].plot(
        result["smoothed"][:, X_POSITION],
        result["smoothed"][:, Y_POSITION],
        label="RTS smoothed",
    )
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("Local motion, globally anchored")
    axes[0, 0].legend()

    axes[0, 1].plot(
        result["times"],
        result["dead_reckoning_position_error"],
        label="local dead reckoning",
    )
    axes[0, 1].plot(
        result["times"],
        result["filtered_position_error"],
        label="filtered",
    )
    axes[0, 1].plot(
        result["times"],
        result["smoothed_position_error"],
        label="RTS smoothed",
    )
    axes[0, 1].set_ylabel("position error [m]")
    axes[0, 1].set_title("Absolute position error")
    axes[0, 1].legend()

    axes[1, 0].plot(
        result["times"],
        result["filtered"][:, GYRO_BIAS],
        label="estimated bias",
    )
    axes[1, 0].plot(
        result["times"],
        result["smoothed"][:, GYRO_BIAS],
        label="RTS smoothed bias",
    )
    axes[1, 0].axhline(
        TRUE_GYRO_BIAS_RAD_S,
        linestyle="--",
        label="true bias",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("gyro bias [rad/s]")
    axes[1, 0].set_title("Global fixes calibrate local heading drift")
    axes[1, 0].legend()

    axes[1, 1].plot(
        result["times"],
        np.sqrt(result["position_covariance_trace"]),
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("sqrt(trace(Pxy)) [m]")
    axes[1, 1].set_title("Uncertainty grows without global fixes")

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axvspan(
            result["outage_start_s"],
            result["outage_end_s"],
            alpha=0.12,
            label="global outage" if axis is axes[1, 1] else None,
        )
        axis.grid(alpha=0.2)
    axes[0, 0].grid(alpha=0.2)
    axes[1, 1].legend()

    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    return figure


def _print_summary(result):
    print("Local-global planar sensor fusion")
    print(f"  pseudo-GPS RMSE: {result['global_position_rmse']:.2f} m")
    print(
        f"  position RMSE: {result['dead_reckoning_position_rmse']:.2f} m local only -> "
        f"{result['filtered_position_rmse']:.2f} m filtered -> "
        f"{result['smoothed_position_rmse']:.2f} m RTS"
    )
    print(
        f"  gyro bias: {result['final_gyro_bias_rad_s']:.5f} rad/s "
        f"(truth {TRUE_GYRO_BIAS_RAD_S:.5f})"
    )
    print(
        f"  max filtered error during {GLOBAL_OUTAGE_END_S - GLOBAL_OUTAGE_START_S:.0f} s "
        f"global outage: {result['outage_max_position_error_m']:.2f} m"
    )


if __name__ == "__main__":
    result = run_local_global_fusion()
    path = Path(__file__).parent / "figures" / "local-global-fusion.png"
    plot_local_global_fusion(result, path)
    _print_summary(result)
    print(f"saved {path}")

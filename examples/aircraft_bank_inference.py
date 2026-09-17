"""Infer aircraft bank and flight-path angle from noisy 3D position tracks.

The only sensor in this example reports global ``[x, y, z]`` position. The
filter nevertheless estimates latent aircraft motion variables by imposing a
coordinated-flight model. In particular, horizontal trajectory curvature is
coupled to bank angle through

``course_rate = g * tan(bank) / (speed * cos(flight_path_angle))``.

The hidden state is

``[x, y, z, speed, course, flight_path_angle, bank]``.

No attitude measurements are supplied. The dynamics are propagated at 10 Hz
while noisy position fixes arrive at only 1 Hz. ``flight_path_angle`` is the
elevation angle of the velocity vector, not body pitch; inferring body pitch
would also require an angle-of-attack or aerodynamic model.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_M_S2 = 9.80665
X_POSITION = 0
Y_POSITION = 1
Z_POSITION = 2
SPEED = 3
COURSE = 4
FLIGHT_PATH_ANGLE = 5
BANK = 6
STATE_SIZE = 7

DEFAULT_DURATION_S = 180.0
DEFAULT_FILTER_RATE_HZ = 10.0
DEFAULT_POSITION_RATE_HZ = 1.0
POSITION_STD_M = np.array([35.0, 35.0, 22.0])


def coordinated_turn_rate(speed, flight_path_angle, bank):
    """Return course rate for a coordinated turn."""
    return (
        GRAVITY_M_S2
        * np.tan(bank)
        / (speed * np.cos(flight_path_angle))
    )


def propagate_aircraft_state(state, delta_t_s):
    """Propagate the coordinated-flight state over one interval."""
    state = np.asarray(state)
    (
        x_position,
        y_position,
        z_position,
        speed,
        course,
        flight_path_angle,
        bank,
    ) = state

    course_rate = coordinated_turn_rate(
        speed,
        flight_path_angle,
        bank,
    )
    midpoint_course = course + 0.5 * delta_t_s * course_rate
    horizontal_speed = speed * np.cos(flight_path_angle)

    return np.array(
        [
            x_position
            + delta_t_s * horizontal_speed * np.cos(midpoint_course),
            y_position
            + delta_t_s * horizontal_speed * np.sin(midpoint_course),
            z_position + delta_t_s * speed * np.sin(flight_path_angle),
            speed,
            course + delta_t_s * course_rate,
            flight_path_angle,
            bank,
        ]
    )


def aircraft_transition_jacobian(state, delta_t_s):
    """Analytic Jacobian of :func:`propagate_aircraft_state`."""
    state = np.asarray(state)
    speed = state[SPEED]
    course = state[COURSE]
    flight_path_angle = state[FLIGHT_PATH_ANGLE]
    bank = state[BANK]

    course_rate = coordinated_turn_rate(
        speed,
        flight_path_angle,
        bank,
    )
    d_rate_d_speed = -course_rate / speed
    d_rate_d_flight_path = course_rate * np.tan(flight_path_angle)
    d_rate_d_bank = (
        GRAVITY_M_S2
        / (
            speed
            * np.cos(flight_path_angle)
            * np.cos(bank) ** 2
        )
    )

    midpoint_course = course + 0.5 * delta_t_s * course_rate
    sine = np.sin(midpoint_course)
    cosine = np.cos(midpoint_course)
    horizontal_speed = speed * np.cos(flight_path_angle)

    d_midpoint_d_speed = 0.5 * delta_t_s * d_rate_d_speed
    d_midpoint_d_flight_path = (
        0.5 * delta_t_s * d_rate_d_flight_path
    )
    d_midpoint_d_bank = 0.5 * delta_t_s * d_rate_d_bank

    jacobian = np.eye(STATE_SIZE)
    jacobian[X_POSITION, COURSE] = -delta_t_s * horizontal_speed * sine
    jacobian[X_POSITION, SPEED] = delta_t_s * (
        np.cos(flight_path_angle) * cosine
        - horizontal_speed * sine * d_midpoint_d_speed
    )
    jacobian[X_POSITION, FLIGHT_PATH_ANGLE] = delta_t_s * (
        -speed * np.sin(flight_path_angle) * cosine
        - horizontal_speed * sine * d_midpoint_d_flight_path
    )
    jacobian[X_POSITION, BANK] = (
        -delta_t_s
        * horizontal_speed
        * sine
        * d_midpoint_d_bank
    )

    jacobian[Y_POSITION, COURSE] = delta_t_s * horizontal_speed * cosine
    jacobian[Y_POSITION, SPEED] = delta_t_s * (
        np.cos(flight_path_angle) * sine
        + horizontal_speed * cosine * d_midpoint_d_speed
    )
    jacobian[Y_POSITION, FLIGHT_PATH_ANGLE] = delta_t_s * (
        -speed * np.sin(flight_path_angle) * sine
        + horizontal_speed * cosine * d_midpoint_d_flight_path
    )
    jacobian[Y_POSITION, BANK] = (
        delta_t_s
        * horizontal_speed
        * cosine
        * d_midpoint_d_bank
    )

    jacobian[Z_POSITION, SPEED] = delta_t_s * np.sin(flight_path_angle)
    jacobian[Z_POSITION, FLIGHT_PATH_ANGLE] = (
        delta_t_s * speed * np.cos(flight_path_angle)
    )

    jacobian[COURSE, SPEED] = delta_t_s * d_rate_d_speed
    jacobian[COURSE, FLIGHT_PATH_ANGLE] = (
        delta_t_s * d_rate_d_flight_path
    )
    jacobian[COURSE, BANK] = delta_t_s * d_rate_d_bank
    return jacobian


def observe_position(state):
    """Observe only global Cartesian aircraft position."""
    return np.asarray(state)[:3]


def position_jacobian(_state):
    """Jacobian of the position-only observation model."""
    jacobian = np.zeros((3, STATE_SIZE))
    jacobian[:, :3] = np.eye(3)
    return jacobian


def _smoothstep01(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smooth_window(time_s, rise_start, rise_end, fall_start, fall_end):
    time_s = np.asarray(time_s, dtype=float)
    rise = _smoothstep01(
        (time_s - rise_start) / (rise_end - rise_start)
    )
    fall = 1.0 - _smoothstep01(
        (time_s - fall_start) / (fall_end - fall_start)
    )
    return np.minimum(rise, fall)


def bank_profile(time_s):
    """Left, right, then gentler left turns for the synthetic aircraft."""
    return (
        np.deg2rad(25.0)
        * _smooth_window(time_s, 20.0, 28.0, 48.0, 58.0)
        - np.deg2rad(30.0)
        * _smooth_window(time_s, 72.0, 82.0, 108.0, 118.0)
        + np.deg2rad(18.0)
        * _smooth_window(time_s, 132.0, 140.0, 156.0, 166.0)
    )


def flight_path_angle_profile(time_s):
    """Climb and descent segments, expressed as velocity elevation angle."""
    return (
        np.deg2rad(5.0)
        * _smooth_window(time_s, 42.0, 52.0, 78.0, 90.0)
        - np.deg2rad(4.0)
        * _smooth_window(time_s, 108.0, 118.0, 142.0, 154.0)
    )


def speed_profile(time_s):
    """Vary airspeed slowly so the filter must infer speed as well as attitude."""
    time_s = np.asarray(time_s, dtype=float)
    return (
        95.0
        + 7.0 * np.sin(2.0 * np.pi * time_s / 85.0)
        + 2.0 * np.sin(2.0 * np.pi * time_s / 23.0 + 0.5)
    )


def simulate_aircraft_truth(times):
    """Simulate a 3D coordinated-flight trajectory."""
    times = np.asarray(times)
    truth = np.empty((len(times), STATE_SIZE))
    truth[0] = np.array(
        [
            0.0,
            0.0,
            1200.0,
            speed_profile(times[0]),
            np.deg2rad(15.0),
            flight_path_angle_profile(times[0]),
            bank_profile(times[0]),
        ]
    )

    for index, delta_t_s in enumerate(np.diff(times)):
        midpoint_time = 0.5 * (times[index] + times[index + 1])
        midpoint_state = truth[index].copy()
        midpoint_state[SPEED] = speed_profile(midpoint_time)
        midpoint_state[FLIGHT_PATH_ANGLE] = flight_path_angle_profile(
            midpoint_time
        )
        midpoint_state[BANK] = bank_profile(midpoint_time)
        truth[index + 1] = propagate_aircraft_state(
            midpoint_state,
            delta_t_s,
        )
        truth[index + 1, SPEED] = speed_profile(times[index + 1])
        truth[index + 1, FLIGHT_PATH_ANGLE] = flight_path_angle_profile(
            times[index + 1]
        )
        truth[index + 1, BANK] = bank_profile(times[index + 1])

    return truth


def run_aircraft_bank_inference(
    seed=231,
    duration_s=DEFAULT_DURATION_S,
    filter_rate_hz=DEFAULT_FILTER_RATE_HZ,
    position_rate_hz=DEFAULT_POSITION_RATE_HZ,
    use_jacobian=False,
):
    """Infer latent aircraft maneuver state from noisy position-only tracks."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    times = np.arange(round(duration_s * filter_rate_hz) + 1) * delta_t_s
    truth = simulate_aircraft_truth(times)

    position_interval_steps = round(filter_rate_hz / position_rate_hz)
    if not np.isclose(position_interval_steps * position_rate_hz, filter_rate_hz):
        raise ValueError("position_rate_hz must divide filter_rate_hz")

    position_indices = np.arange(
        position_interval_steps,
        len(times),
        position_interval_steps,
    )
    position_times = times[position_indices]
    position_truth = truth[position_indices, :3]
    measurements = position_truth + rng.normal(
        0.0,
        POSITION_STD_M,
        size=position_truth.shape,
    )
    observations = [
        Observation(
            measurement,
            np.diag(POSITION_STD_M**2),
            observe_position,
            position_jacobian,
        )
        for measurement in measurements
    ]

    process_std_per_sqrt_s = np.array(
        [
            0.5,
            0.5,
            0.25,
            2.0,
            np.deg2rad(0.15),
            np.deg2rad(0.8),
            np.deg2rad(2.2),
        ]
    )
    process_std = process_std_per_sqrt_s * np.sqrt(delta_t_s)
    transition_model = StateTransitionModel(
        propagate_aircraft_state,
        np.diag(process_std**2),
        aircraft_transition_jacobian,
    )

    # BayesianFilter.run() predicts one fixed-rate step before handling the
    # first observation, so initialise one filter step before the first fix.
    initial_truth_index = position_indices[0] - 1
    initial_mean = truth[initial_truth_index] + np.array(
        [
            100.0,
            -80.0,
            60.0,
            -12.0,
            np.deg2rad(10.0),
            np.deg2rad(3.0),
            np.deg2rad(-8.0),
        ]
    )
    initial_std = np.array(
        [
            150.0,
            150.0,
            100.0,
            18.0,
            np.deg2rad(18.0),
            np.deg2rad(8.0),
            np.deg2rad(18.0),
        ]
    )
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
    )
    filter_states, filter_times = bayes_filter.run(
        observations,
        position_times,
        rate_hz=filter_rate_hz,
        use_jacobian=use_jacobian,
    )
    filter_times = np.asarray(filter_times)
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )

    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.array(
        [state.covariance() for state in smoother_states]
    )

    aligned_indices = np.rint(filter_times * filter_rate_hz).astype(int)
    aligned_truth = truth[aligned_indices]

    def position_rmse(states):
        return float(
            np.sqrt(
                np.mean(
                    np.sum((states[:, :3] - aligned_truth[:, :3]) ** 2, axis=1)
                )
            )
        )

    def angular_rmse_deg(states, state_index):
        error = states[:, state_index] - aligned_truth[:, state_index]
        return float(np.rad2deg(np.sqrt(np.mean(error**2))))

    turning = np.abs(aligned_truth[:, BANK]) > np.deg2rad(10.0)
    raw_measurement_rmse = float(
        np.sqrt(
            np.mean(
                np.sum((measurements - position_truth) ** 2, axis=1)
            )
        )
    )

    return {
        "times": filter_times,
        "truth": aligned_truth,
        "measurement_times": position_times,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "raw_measurement_position_rmse_m": raw_measurement_rmse,
        "filtered_position_rmse_m": position_rmse(filtered),
        "smoothed_position_rmse_m": position_rmse(smoothed),
        "filtered_bank_rmse_deg": angular_rmse_deg(filtered, BANK),
        "smoothed_bank_rmse_deg": angular_rmse_deg(smoothed, BANK),
        "turning_smoothed_bank_rmse_deg": float(
            np.rad2deg(
                np.sqrt(
                    np.mean(
                        (
                            smoothed[turning, BANK]
                            - aligned_truth[turning, BANK]
                        )
                        ** 2
                    )
                )
            )
        ),
        "filtered_flight_path_angle_rmse_deg": angular_rmse_deg(
            filtered,
            FLIGHT_PATH_ANGLE,
        ),
        "smoothed_flight_path_angle_rmse_deg": angular_rmse_deg(
            smoothed,
            FLIGHT_PATH_ANGLE,
        ),
        "use_jacobian": use_jacobian,
        "filter_rate_hz": filter_rate_hz,
        "position_rate_hz": position_rate_hz,
    }


def plot_aircraft_bank_inference(result, output_path=None):
    """Plot the 3D track and latent maneuver variables inferred from it."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 9))
    trajectory_axis = figure.add_subplot(2, 2, 1, projection="3d")
    bank_axis = figure.add_subplot(2, 2, 2)
    path_axis = figure.add_subplot(2, 2, 3)
    error_axis = figure.add_subplot(2, 2, 4)

    trajectory_axis.plot(
        result["truth"][:, X_POSITION],
        result["truth"][:, Y_POSITION],
        result["truth"][:, Z_POSITION],
        "--",
        label="truth",
    )
    trajectory_axis.scatter(
        result["measurements"][:, X_POSITION],
        result["measurements"][:, Y_POSITION],
        result["measurements"][:, Z_POSITION],
        s=7,
        alpha=0.3,
        label="noisy position",
    )
    trajectory_axis.plot(
        result["smoothed"][:, X_POSITION],
        result["smoothed"][:, Y_POSITION],
        result["smoothed"][:, Z_POSITION],
        label="RTS smoothed",
    )
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_zlabel("z [m]")
    trajectory_axis.set_title("Position-only aircraft track")
    trajectory_axis.legend()

    time_s = result["times"]
    bank_axis.plot(
        time_s,
        np.rad2deg(result["truth"][:, BANK]),
        "--",
        label="true bank",
    )
    bank_axis.plot(
        time_s,
        np.rad2deg(result["filtered"][:, BANK]),
        alpha=0.7,
        label="filtered bank",
    )
    bank_axis.plot(
        time_s,
        np.rad2deg(result["smoothed"][:, BANK]),
        label="RTS bank",
    )
    bank_axis.set_ylabel("bank [deg]")
    bank_axis.set_title("Bank inferred from trajectory curvature")
    bank_axis.legend()

    path_axis.plot(
        time_s,
        np.rad2deg(result["truth"][:, FLIGHT_PATH_ANGLE]),
        "--",
        label="true flight-path angle",
    )
    path_axis.plot(
        time_s,
        np.rad2deg(result["smoothed"][:, FLIGHT_PATH_ANGLE]),
        label="RTS estimate",
    )
    path_axis.set_xlabel("time [s]")
    path_axis.set_ylabel("flight-path angle [deg]")
    path_axis.set_title("Climb/descent angle from vertical motion")
    path_axis.legend()

    filtered_position_error = np.linalg.norm(
        result["filtered"][:, :3] - result["truth"][:, :3],
        axis=1,
    )
    smoothed_position_error = np.linalg.norm(
        result["smoothed"][:, :3] - result["truth"][:, :3],
        axis=1,
    )
    error_axis.plot(
        time_s,
        filtered_position_error,
        label="filtered",
    )
    error_axis.plot(
        time_s,
        smoothed_position_error,
        label="RTS smoothed",
    )
    error_axis.set_xlabel("time [s]")
    error_axis.set_ylabel("position error [m]")
    error_axis.set_title("Position reconstruction")
    error_axis.legend()

    for axis in (bank_axis, path_axis, error_axis):
        axis.grid(alpha=0.2)
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    return figure


def _print_summary(result):
    print("Aircraft bank inference from position-only tracks")
    print(
        f"  position RMSE: {result['raw_measurement_position_rmse_m']:.1f} m raw -> "
        f"{result['filtered_position_rmse_m']:.1f} m filtered -> "
        f"{result['smoothed_position_rmse_m']:.1f} m RTS"
    )
    print(
        f"  bank RMSE: {result['filtered_bank_rmse_deg']:.2f} deg filtered -> "
        f"{result['smoothed_bank_rmse_deg']:.2f} deg RTS"
    )
    print(
        f"  bank RMSE while turning: "
        f"{result['turning_smoothed_bank_rmse_deg']:.2f} deg"
    )
    print(
        f"  flight-path-angle RMSE: "
        f"{result['smoothed_flight_path_angle_rmse_deg']:.2f} deg RTS"
    )


if __name__ == "__main__":
    result = run_aircraft_bank_inference()
    path = Path(__file__).parent / "figures" / "aircraft-bank-inference.png"
    plot_aircraft_bank_inference(result, path)
    _print_summary(result)
    print(f"saved {path}")

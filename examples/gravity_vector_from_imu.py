"""Estimate the gravity vector from raw gyroscope and accelerometer data.

State:
    [g_x, g_y, g_z, omega_x, omega_y, omega_z]

Here g is gravity expressed in the IMU/body frame and omega is body angular
velocity. For a rotating body, a world-fixed gravity vector evolves in body
coordinates as

    g_dot = -omega x g.

The raw gyroscope directly observes angular velocity, while a stationary
accelerometer measures specific force approximately equal to -g. During linear
acceleration the accelerometer is not a direct gravity sensor, so the example
injects several translational-acceleration bursts. The filter combines
short-term gyro propagation with the accelerometer's long-term gravity cue and
is substantially less disturbed than the naive estimate g = -a.

This intentionally small example assumes negligible gyro and accelerometer
bias; later examples can augment the state with those calibration terms.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


GRAVITY_M_S2 = 9.80665
DEFAULT_RATE_HZ = 50.0
DEFAULT_DURATION_S = 20.0
GYRO_NOISE_STD_RAD_S = 0.0035
ACCEL_NOISE_STD_M_S2 = 0.08
FILTER_ACCEL_STD_M_S2 = 0.5


def skew(vector):
    x, y, z = np.asarray(vector)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def rotation_vector_to_matrix(rotation_vector):
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.eye(3) + skew(rotation_vector)

    axis = rotation_vector / angle
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )


def propagate_gravity_state(state, delta_t_s):
    """Rotate body-frame gravity using the current angular velocity."""
    state = np.asarray(state)
    propagated = state.copy()
    propagated[:3] = (
        rotation_vector_to_matrix(-state[3:6] * delta_t_s)
        @ state[:3]
    )
    return propagated


def observe_raw_imu(state):
    """Predict [gyro_xyz, accelerometer_xyz]."""
    state = np.asarray(state)
    return np.concatenate([state[3:6], -state[:3]])


def raw_imu_observation_jacobian(_state):
    return np.block(
        [
            [np.zeros((3, 3)), np.eye(3)],
            [-np.eye(3), np.zeros((3, 3))],
        ]
    )


def true_angular_velocity(time_s):
    return np.array(
        [
            0.22 * np.sin(0.55 * time_s)
            + 0.08 * np.sin(1.70 * time_s),
            0.18 * np.cos(0.43 * time_s)
            + 0.06 * np.sin(1.10 * time_s),
            0.30 * np.sin(0.31 * time_s) + 0.10,
        ]
    )


def initial_gravity_vector():
    roll = np.deg2rad(20.0)
    pitch = np.deg2rad(-15.0)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), np.sin(roll)],
            [0.0, -np.sin(roll), np.cos(roll)],
        ]
    )
    rotation_y = np.array(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    return rotation_x @ rotation_y @ np.array([0.0, 0.0, GRAVITY_M_S2])


def simulate_raw_imu(
    seed=4,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Generate raw gyro/accelerometer data with linear-acceleration bursts."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / rate_hz
    times = np.arange(round(duration_s * rate_hz) + 1) * delta_t_s

    truth = np.zeros((len(times), 6))
    truth[0, :3] = initial_gravity_vector()
    truth[0, 3:6] = true_angular_velocity(times[0])

    for index, delta_t in enumerate(np.diff(times)):
        midpoint_time = 0.5 * (times[index] + times[index + 1])
        angular_velocity = true_angular_velocity(midpoint_time)
        truth[index + 1, :3] = (
            rotation_vector_to_matrix(-angular_velocity * delta_t)
            @ truth[index, :3]
        )
        truth[index + 1, 3:6] = true_angular_velocity(times[index + 1])

    linear_acceleration = np.zeros((len(times), 3))
    bursts = (
        (4.0, 6.0, np.array([1.8, -0.4, 0.5])),
        (9.0, 10.5, np.array([-1.2, 1.0, 0.2])),
        (14.0, 17.0, np.array([0.5, -1.5, 0.8])),
    )
    for start_s, end_s, peak_acceleration in bursts:
        active = (times >= start_s) & (times < end_s)
        phase = (times[active] - start_s) / (end_s - start_s)
        envelope = np.sin(np.pi * phase)
        linear_acceleration[active] = (
            envelope[:, None] * peak_acceleration
        )

    gyro = truth[:, 3:6] + rng.normal(
        0.0,
        GYRO_NOISE_STD_RAD_S,
        size=(len(times), 3),
    )
    accelerometer = (
        -truth[:, :3]
        + linear_acceleration
        + rng.normal(
            0.0,
            ACCEL_NOISE_STD_M_S2,
            size=(len(times), 3),
        )
    )

    return {
        "times": times,
        "truth": truth,
        "gyro": gyro,
        "accelerometer": accelerometer,
        "linear_acceleration": linear_acceleration,
        "rate_hz": rate_hz,
    }


def gravity_angle_error_deg(estimate, truth):
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)

    estimate_unit = estimate / np.linalg.norm(
        estimate,
        axis=-1,
        keepdims=True,
    )
    truth_unit = truth / np.linalg.norm(
        truth,
        axis=-1,
        keepdims=True,
    )
    cosine = np.sum(estimate_unit * truth_unit, axis=-1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def run_gravity_vector_from_imu(
    seed=4,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
):
    """Estimate body-frame gravity from raw gyro + accelerometer data."""
    data = simulate_raw_imu(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
    )
    times = data["times"]

    measurement_covariance = np.diag(
        [GYRO_NOISE_STD_RAD_S**2] * 3
        + [FILTER_ACCEL_STD_M_S2**2] * 3
    )
    observations = [
        Observation(
            np.concatenate([gyro, accelerometer]),
            measurement_covariance,
            observe_raw_imu,
            raw_imu_observation_jacobian,
        )
        for gyro, accelerometer in zip(
            data["gyro"][1:],
            data["accelerometer"][1:],
        )
    ]

    process_covariance = np.diag(
        [0.0008**2] * 3 + [0.012**2] * 3
    )
    model = StateTransitionModel(
        propagate_gravity_state,
        process_covariance,
    )

    initial_mean = np.concatenate(
        [-data["accelerometer"][0], data["gyro"][0]]
    )
    initial_covariance = np.diag(
        [0.5**2] * 3 + [0.05**2] * 3
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=False,
    )

    filtered = np.asarray([state.mean() for state in filter_states])
    filtered_covariances = np.asarray(
        [state.covariance() for state in filter_states]
    )
    naive_gravity = -data["accelerometer"]

    filtered_angle_error_deg = gravity_angle_error_deg(
        filtered[:, :3],
        data["truth"][:, :3],
    )
    naive_angle_error_deg = gravity_angle_error_deg(
        naive_gravity,
        data["truth"][:, :3],
    )

    acceleration_active = (
        np.linalg.norm(data["linear_acceleration"], axis=1) > 1e-6
    )

    return {
        **data,
        "filtered": filtered,
        "filtered_covariances": filtered_covariances,
        "naive_gravity": naive_gravity,
        "filtered_angle_error_deg": filtered_angle_error_deg,
        "naive_angle_error_deg": naive_angle_error_deg,
        "filtered_angle_rmse_deg": float(
            np.sqrt(np.mean(filtered_angle_error_deg**2))
        ),
        "naive_angle_rmse_deg": float(
            np.sqrt(np.mean(naive_angle_error_deg**2))
        ),
        "filtered_acceleration_burst_rmse_deg": float(
            np.sqrt(
                np.mean(
                    filtered_angle_error_deg[acceleration_active] ** 2
                )
            )
        ),
        "naive_acceleration_burst_rmse_deg": float(
            np.sqrt(
                np.mean(
                    naive_angle_error_deg[acceleration_active] ** 2
                )
            )
        ),
        "final_angle_error_deg": float(filtered_angle_error_deg[-1]),
    }


def plot_gravity_vector_from_imu(result, output_path=None):
    """Plot gravity estimates and translational-acceleration corruption."""
    import matplotlib.pyplot as plt

    times = result["times"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    labels = ("x", "y", "z")
    for index, label in enumerate(labels):
        axes[0, 0].plot(
            times,
            result["truth"][:, index],
            linestyle="--",
            label=f"true g_{label}",
        )
        axes[0, 0].plot(
            times,
            result["filtered"][:, index],
            label=f"estimated g_{label}",
        )
    axes[0, 0].set_ylabel("gravity component [m/s²]")
    axes[0, 0].set_title("Gravity vector in the moving IMU frame")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].plot(
        times,
        result["filtered_angle_error_deg"],
        label="Bayesian filter",
    )
    axes[0, 1].plot(
        times,
        result["naive_angle_error_deg"],
        alpha=0.75,
        label="naive -accelerometer",
    )
    axes[0, 1].set_ylabel("gravity direction error [deg]")
    axes[0, 1].set_title("Linear acceleration corrupts raw accel direction")
    axes[0, 1].legend(fontsize=8)

    for index, label in enumerate(labels):
        axes[1, 0].plot(
            times,
            result["linear_acceleration"][:, index],
            label=f"a_{label}",
        )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("linear acceleration [m/s²]")
    axes[1, 0].set_title("Unmodelled translational-acceleration bursts")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        np.linalg.norm(result["truth"][:, :3], axis=1),
        linestyle="--",
        label="true |g|",
    )
    axes[1, 1].plot(
        times,
        np.linalg.norm(result["filtered"][:, :3], axis=1),
        label="estimated |g|",
    )
    axes[1, 1].axhline(
        GRAVITY_M_S2,
        linestyle=":",
        label="9.80665 m/s²",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("magnitude [m/s²]")
    axes[1, 1].set_title("Rotation dynamics preserve gravity magnitude")
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
    print("Gravity vector estimation from raw IMU data")
    print(
        f"  gravity direction RMSE: "
        f"{result['naive_angle_rmse_deg']:.2f} deg naive -> "
        f"{result['filtered_angle_rmse_deg']:.2f} deg filtered"
    )
    print(
        f"  during linear acceleration: "
        f"{result['naive_acceleration_burst_rmse_deg']:.2f} deg naive -> "
        f"{result['filtered_acceleration_burst_rmse_deg']:.2f} deg filtered"
    )
    print(
        f"  final gravity direction error: "
        f"{result['final_angle_error_deg']:.2f} deg"
    )


if __name__ == "__main__":
    result = run_gravity_vector_from_imu()
    path = Path(__file__).parent / "figures" / "gravity-vector-from-imu.png"
    plot_gravity_vector_from_imu(result, path)
    _print_summary(result)
    print(f"saved {path}")

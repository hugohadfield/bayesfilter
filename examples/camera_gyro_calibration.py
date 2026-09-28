"""Calibrate camera-to-gyro rotation, clock offset, and gyro bias from motion.

A camera and gyroscope are rigidly attached but record independently to disk.
The gyro produces a high-rate angular-velocity log in its own coordinate frame.
A visual front end (for example visual SLAM, visual odometry, or rotational
optical flow) produces lower-rate camera angular velocities from the video.
The streams are not hardware synchronized.

The hidden calibration state is

``[rotation_vector_camera_from_gyro, time_offset, gyro_bias]``.

For a camera timestamp ``t``, the predicted visual angular velocity is

``omega_camera(t) = R_camera_from_gyro * (omega_gyro(t + offset) - bias)``.

The observation is nonlinear in both the rotation and the timestamp offset:
the filter must rotate the gyro vector and interpolate the recorded gyro log at
a state-dependent time. The first five seconds excite only one rotation axis,
which leaves part of the extrinsic rotation weakly observable. Rich three-axis
motion after that makes the full spatiotemporal calibration observable.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


ROTATION_VECTOR = slice(0, 3)
TIME_OFFSET = 3
GYRO_BIAS = slice(4, 7)
STATE_SIZE = 7

DURATION_S = 30.0
GYRO_RATE_HZ = 200.0
CAMERA_RATE_HZ = 30.0
WEAK_EXCITATION_END_S = 5.0
TRUE_TIME_OFFSET_S = 0.073
TRUE_ROTATION_VECTOR_RAD = np.array([0.09, -0.13, 0.29])
TRUE_GYRO_BIAS_RAD_S = np.array([0.012, -0.008, 0.006])
GYRO_STD_RAD_S = 0.004
VISUAL_ANGULAR_RATE_STD_RAD_S = 0.020


def skew_symmetric(vector):
    """Return the 3x3 skew-symmetric matrix for a three-vector."""
    x_value, y_value, z_value = np.asarray(vector)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ]
    )


def rotation_vector_to_matrix(rotation_vector):
    """Convert a rotation vector to a 3D rotation matrix with Rodrigues' formula."""
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-10:
        skew = skew_symmetric(rotation_vector)
        return np.eye(3) + skew + 0.5 * skew @ skew

    axis = rotation_vector / angle
    skew = skew_symmetric(axis)
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def rotation_error_deg(estimated_rotation_vector):
    """Return the angular error from the true camera-from-gyro rotation."""
    estimated = rotation_vector_to_matrix(estimated_rotation_vector)
    truth = rotation_vector_to_matrix(TRUE_ROTATION_VECTOR_RAD)
    relative = estimated @ truth.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def rig_angular_velocity_gyro_frame(time_s):
    """Synthetic rig angular velocity expressed in the gyro frame."""
    time_s = float(time_s)
    if time_s < WEAK_EXCITATION_END_S:
        return np.array(
            [
                0.0,
                0.0,
                0.65 * np.sin(2.0 * np.pi * 0.55 * time_s)
                + 0.35 * np.sin(2.0 * np.pi * 1.10 * time_s + 0.4),
            ]
        )

    rich_time_s = time_s - WEAK_EXCITATION_END_S
    return np.array(
        [
            0.55 * np.sin(2.0 * np.pi * 0.43 * rich_time_s)
            + 0.18 * np.sin(2.0 * np.pi * 1.20 * rich_time_s + 0.2),
            0.48 * np.sin(2.0 * np.pi * 0.31 * rich_time_s + 0.7)
            + 0.22 * np.sin(2.0 * np.pi * 0.90 * rich_time_s),
            0.62 * np.sin(2.0 * np.pi * 0.37 * rich_time_s + 0.4)
            + 0.25 * np.sin(2.0 * np.pi * 1.05 * rich_time_s + 1.0),
        ]
    )


def simulate_camera_gyro_logs(seed=42):
    """Simulate independently timestamped gyro and camera-rotation logs."""
    rng = np.random.default_rng(seed)

    gyro_step_s = 1.0 / GYRO_RATE_HZ
    gyro_times_s = np.arange(
        -0.5,
        DURATION_S + 0.5 + 0.5 * gyro_step_s,
        gyro_step_s,
    )
    true_gyro_angular_velocity = np.asarray(
        [rig_angular_velocity_gyro_frame(time_s) for time_s in gyro_times_s]
    )
    gyro_measurements = (
        true_gyro_angular_velocity
        + TRUE_GYRO_BIAS_RAD_S
        + rng.normal(
            0.0,
            GYRO_STD_RAD_S,
            size=true_gyro_angular_velocity.shape,
        )
    )

    camera_step_s = 1.0 / CAMERA_RATE_HZ
    camera_times_s = np.arange(
        0.0,
        DURATION_S + 0.5 * camera_step_s,
        camera_step_s,
    )
    camera_from_gyro = rotation_vector_to_matrix(TRUE_ROTATION_VECTOR_RAD)
    true_visual_angular_velocity = np.asarray(
        [
            camera_from_gyro
            @ rig_angular_velocity_gyro_frame(time_s + TRUE_TIME_OFFSET_S)
            for time_s in camera_times_s
        ]
    )
    visual_angular_velocity = true_visual_angular_velocity + rng.normal(
        0.0,
        VISUAL_ANGULAR_RATE_STD_RAD_S,
        size=true_visual_angular_velocity.shape,
    )

    return {
        "gyro_times_s": gyro_times_s,
        "gyro_measurements_rad_s": gyro_measurements,
        "camera_times_s": camera_times_s,
        "visual_angular_velocity_rad_s": visual_angular_velocity,
        "true_visual_angular_velocity_rad_s": true_visual_angular_velocity,
    }


def interpolate_gyro(gyro_times_s, gyro_measurements, query_time_s):
    """Linearly interpolate the recorded gyro log at one timestamp."""
    return np.array(
        [
            np.interp(
                query_time_s,
                gyro_times_s,
                gyro_measurements[:, axis],
            )
            for axis in range(3)
        ]
    )


def make_visual_angular_velocity_observation_func(
    camera_time_s,
    gyro_times_s,
    gyro_measurements,
):
    """Create the camera angular-velocity prediction for one video timestamp."""

    def observe_visual_angular_velocity(state):
        state = np.asarray(state)
        camera_from_gyro = rotation_vector_to_matrix(state[ROTATION_VECTOR])
        time_offset_s = state[TIME_OFFSET]
        gyro_bias = state[GYRO_BIAS]
        measured_gyro = interpolate_gyro(
            gyro_times_s,
            gyro_measurements,
            camera_time_s + time_offset_s,
        )
        return camera_from_gyro @ (measured_gyro - gyro_bias)

    return observe_visual_angular_velocity


def propagate_calibration_state(state, _delta_t_s):
    """Keep the rigid extrinsics, clock offset, and bias approximately constant."""
    return np.asarray(state).copy()


def process_noise_covariance():
    """Allow only a tiny random walk in the nominally static calibration."""
    process_std = np.array(
        [
            2e-5,
            2e-5,
            2e-5,
            2e-5,
            2e-5,
            2e-5,
            2e-5,
        ]
    )
    return np.diag(process_std**2)


def initial_distribution():
    """Start uncalibrated: identity rotation, zero offset, and zero gyro bias."""
    initial_std = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(20.0),
            np.deg2rad(20.0),
            0.12,
            0.06,
            0.06,
            0.06,
        ]
    )
    return Gaussian(
        np.zeros(STATE_SIZE),
        np.diag(initial_std**2),
    )


def _make_observations(sensor_data):
    combined_std = np.sqrt(
        VISUAL_ANGULAR_RATE_STD_RAD_S**2 + GYRO_STD_RAD_S**2
    )
    covariance = np.eye(3) * combined_std**2
    observations = []
    for camera_time_s, visual_measurement in zip(
        sensor_data["camera_times_s"][1:],
        sensor_data["visual_angular_velocity_rad_s"][1:],
    ):
        observations.append(
            Observation(
                visual_measurement,
                covariance,
                make_visual_angular_velocity_observation_func(
                    camera_time_s,
                    sensor_data["gyro_times_s"],
                    sensor_data["gyro_measurements_rad_s"],
                ),
            )
        )
    return observations


def _state_array(states):
    return np.asarray([state.mean() for state in states])


def _covariance_array(states):
    return np.asarray([state.covariance() for state in states])


def _predict_visual_rates(state, sensor_data):
    predictions = []
    for camera_time_s in sensor_data["camera_times_s"]:
        prediction_func = make_visual_angular_velocity_observation_func(
            camera_time_s,
            sensor_data["gyro_times_s"],
            sensor_data["gyro_measurements_rad_s"],
        )
        predictions.append(prediction_func(state))
    return np.asarray(predictions)


def run_camera_gyro_calibration(seed=42):
    """Estimate rotation, timestamp offset, and gyro bias from the two logs."""
    sensor_data = simulate_camera_gyro_logs(seed)
    model = StateTransitionModel(
        propagate_calibration_state,
        process_noise_covariance(),
    )
    bayes_filter = BayesianFilter(model, initial_distribution())
    filtered_states = bayes_filter.run_synchronous(
        _make_observations(sensor_data),
        sensor_data["camera_times_s"],
        use_jacobian=False,
    )
    smoothed_states = RTS(bayes_filter).apply(
        filtered_states,
        sensor_data["camera_times_s"],
        use_jacobian=False,
    )

    filtered = _state_array(filtered_states)
    smoothed = _state_array(smoothed_states)
    filtered_covariances = _covariance_array(filtered_states)
    smoothed_covariances = _covariance_array(smoothed_states)

    rotation_errors_deg = np.asarray(
        [rotation_error_deg(state[ROTATION_VECTOR]) for state in filtered]
    )
    smoothed_rotation_errors_deg = np.asarray(
        [rotation_error_deg(state[ROTATION_VECTOR]) for state in smoothed]
    )

    naive_state = np.zeros(STATE_SIZE)
    naive_predictions = _predict_visual_rates(naive_state, sensor_data)
    calibrated_predictions = _predict_visual_rates(filtered[-1], sensor_data)
    measurements = sensor_data["visual_angular_velocity_rad_s"]

    weak_index = int(
        np.argmin(
            np.abs(sensor_data["camera_times_s"] - WEAK_EXCITATION_END_S)
        )
    )

    return {
        **sensor_data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "rotation_errors_deg": rotation_errors_deg,
        "smoothed_rotation_errors_deg": smoothed_rotation_errors_deg,
        "naive_predictions_rad_s": naive_predictions,
        "calibrated_predictions_rad_s": calibrated_predictions,
        "naive_angular_rate_rmse_rad_s": float(
            np.sqrt(np.mean((naive_predictions - measurements) ** 2))
        ),
        "calibrated_angular_rate_rmse_rad_s": float(
            np.sqrt(np.mean((calibrated_predictions - measurements) ** 2))
        ),
        "final_rotation_error_deg": float(rotation_errors_deg[-1]),
        "final_time_offset_s": float(filtered[-1, TIME_OFFSET]),
        "final_time_offset_error_s": float(
            filtered[-1, TIME_OFFSET] - TRUE_TIME_OFFSET_S
        ),
        "final_gyro_bias_rad_s": filtered[-1, GYRO_BIAS].copy(),
        "weak_excitation_index": weak_index,
    }


def plot_camera_gyro_calibration(result, output_path=None):
    """Plot temporal, rotational, bias, and signal-alignment convergence."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    camera_times_s = result["camera_times_s"]

    axes[0, 0].plot(
        camera_times_s,
        1000.0 * result["filtered"][:, TIME_OFFSET],
        label="filtered",
    )
    axes[0, 0].plot(
        camera_times_s,
        1000.0 * result["smoothed"][:, TIME_OFFSET],
        label="RTS smoothed",
    )
    axes[0, 0].axhline(
        1000.0 * TRUE_TIME_OFFSET_S,
        linestyle="--",
        label="truth",
    )
    axes[0, 0].set_ylabel("camera-to-gyro offset [ms]")
    axes[0, 0].set_title("Temporal calibration")
    axes[0, 0].legend()

    axes[0, 1].plot(
        camera_times_s,
        result["rotation_errors_deg"],
        label="filtered",
    )
    axes[0, 1].plot(
        camera_times_s,
        result["smoothed_rotation_errors_deg"],
        label="RTS smoothed",
    )
    axes[0, 1].axvline(
        WEAK_EXCITATION_END_S,
        linestyle=":",
        label="3-axis excitation begins",
    )
    axes[0, 1].set_ylabel("extrinsic rotation error [deg]")
    axes[0, 1].set_title("Spatial calibration")
    axes[0, 1].legend()

    for axis, label in enumerate(("x", "y", "z")):
        axes[1, 0].plot(
            camera_times_s,
            result["filtered"][:, GYRO_BIAS.start + axis],
            label=f"estimated {label}",
        )
        axes[1, 0].axhline(
            TRUE_GYRO_BIAS_RAD_S[axis],
            linestyle="--",
            alpha=0.55,
        )
    axes[1, 0].set_xlabel("camera time [s]")
    axes[1, 0].set_ylabel("gyro bias [rad/s]")
    axes[1, 0].set_title("Gyroscope bias")
    axes[1, 0].legend()

    display_mask = (camera_times_s >= 7.0) & (camera_times_s <= 10.0)
    axes[1, 1].plot(
        camera_times_s[display_mask],
        result["visual_angular_velocity_rad_s"][display_mask, 0],
        label="visual angular rate x",
    )
    axes[1, 1].plot(
        camera_times_s[display_mask],
        result["naive_predictions_rad_s"][display_mask, 0],
        label="uncalibrated gyro prediction",
    )
    axes[1, 1].plot(
        camera_times_s[display_mask],
        result["calibrated_predictions_rad_s"][display_mask, 0],
        label="calibrated gyro prediction",
    )
    axes[1, 1].set_xlabel("camera time [s]")
    axes[1, 1].set_ylabel("angular velocity [rad/s]")
    axes[1, 1].set_title("Alignment after calibration")
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
    print("Camera-gyro spatiotemporal calibration")
    print(
        f"  time offset: 0.0 ms initial -> "
        f"{1000.0 * result['final_time_offset_s']:.2f} ms "
        f"(truth {1000.0 * TRUE_TIME_OFFSET_S:.1f} ms)"
    )
    print(
        f"  extrinsic rotation error: "
        f"{result['final_rotation_error_deg']:.3f} deg"
    )
    print(
        "  gyro bias [rad/s]: "
        f"{result['final_gyro_bias_rad_s']} "
        f"(truth {TRUE_GYRO_BIAS_RAD_S})"
    )
    print(
        f"  angular-rate RMSE: "
        f"{result['naive_angular_rate_rmse_rad_s']:.3f} rad/s uncalibrated -> "
        f"{result['calibrated_angular_rate_rmse_rad_s']:.3f} rad/s calibrated"
    )


if __name__ == "__main__":
    result = run_camera_gyro_calibration()
    path = Path(__file__).parent / "figures" / "camera-gyro-calibration.png"
    plot_camera_gyro_calibration(result, path)
    _print_summary(result)
    print(f"saved {path}")

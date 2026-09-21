"""Calibrate a wrist-mounted camera from intermittent object-pose detections.

A robot moves its wrist through a known Cartesian trajectory while a fixed
object sits on the table. The wrist camera occasionally detects the object's
6-DoF pose in the camera frame.

Transform convention:

    ^A T_B maps coordinates in frame B into frame A.

For every detection,

    ^B T_O = ^B T_W(t)  ^W T_C  ^C T_O(t),

where

- ^B T_W(t) is the known robot wrist pose,
- ^W T_C is the unknown wrist-to-camera extrinsic,
- ^C T_O(t) is the intermittent camera-frame object detection,
- ^B T_O is the unknown but stationary object pose in the robot base frame.

The filter jointly estimates the two stationary poses

    state = [t_WC, phi_WC, t_BO, phi_BO].

This is the robot-world / hand-eye calibration form of the problem. Estimating
the object pose as part of the state means the example does not assume the
table object has been surveyed in the robot base frame.

The vision system is assumed to have known intrinsics upstream and to output a
6-DoF object pose. Each pose detection is used through a local 6-D SE(3) residual: three
translation coordinates plus the rotation logarithm
`Log(R_meas.T @ R_pred)`. This keeps the detector's translational and
rotational noise scales explicit while avoiding a global rotation-vector
subtraction.

Detections are deliberately intermittent and include two longer dropouts.
Frames without a detection simply produce no Observation; the stationary
calibration state propagates through them with tiny process noise.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS

from examples.rigid_body import rotation_distance, so3_exp, so3_log


DEFAULT_DURATION_S = 12.0
DEFAULT_RATE_HZ = 20.0
DETECTION_PROBABILITY = 0.58
DETECTION_DROPOUTS_S = ((3.0, 4.2), (7.4, 8.6))

TRANSLATION_NOISE_STD_M = 0.003
ROTATION_NOISE_STD_RAD = np.deg2rad(0.7)
CAMERA_TRANSLATION_SLICE = slice(0, 3)
CAMERA_ROTATION_SLICE = slice(3, 6)
OBJECT_TRANSLATION_SLICE = slice(6, 9)
OBJECT_ROTATION_SLICE = slice(9, 12)

TRUE_WRIST_TO_CAMERA_TRANSLATION_M = np.array([0.055, -0.028, 0.082])
TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR = np.array([0.045, -0.115, 0.035])

TRUE_BASE_TO_OBJECT_TRANSLATION_M = np.array([0.62, 0.08, 0.025])
TRUE_BASE_TO_OBJECT_ROTATION_VECTOR = np.array([0.04, -0.03, 0.58])


def compose_pose(
    first_translation,
    first_rotation_vector,
    second_translation,
    second_rotation_vector,
):
    """Compose ^A T_B and ^B T_C to obtain ^A T_C."""
    first_rotation = so3_exp(first_rotation_vector)
    second_rotation = so3_exp(second_rotation_vector)
    translation = (
        np.asarray(first_translation)
        + first_rotation @ np.asarray(second_translation)
    )
    rotation_vector = so3_log(first_rotation @ second_rotation)
    return translation, rotation_vector


def invert_pose(translation, rotation_vector):
    """Invert a pose represented by translation and rotation vector."""
    rotation = so3_exp(rotation_vector)
    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ np.asarray(translation)
    return inverse_translation, so3_log(inverse_rotation)


def relative_pose(
    base_to_wrist_translation,
    base_to_wrist_rotation_vector,
    wrist_to_camera_translation,
    wrist_to_camera_rotation_vector,
    base_to_object_translation,
    base_to_object_rotation_vector,
):
    """Predict ^C T_O from known ^B T_W and candidate calibration state."""
    wrist_to_base_translation, wrist_to_base_rotation = invert_pose(
        base_to_wrist_translation,
        base_to_wrist_rotation_vector,
    )
    camera_to_wrist_translation, camera_to_wrist_rotation = invert_pose(
        wrist_to_camera_translation,
        wrist_to_camera_rotation_vector,
    )

    wrist_to_object_translation, wrist_to_object_rotation = compose_pose(
        wrist_to_base_translation,
        wrist_to_base_rotation,
        base_to_object_translation,
        base_to_object_rotation_vector,
    )
    return compose_pose(
        camera_to_wrist_translation,
        camera_to_wrist_rotation,
        wrist_to_object_translation,
        wrist_to_object_rotation,
    )


def wrist_pose_at_time(time_s):
    """Known, sufficiently exciting Cartesian wrist trajectory."""
    time_s = float(time_s)
    translation = np.array(
        [
            0.37 + 0.075 * np.sin(2.0 * np.pi * time_s / 7.0),
            -0.05
            + 0.065 * np.sin(2.0 * np.pi * time_s / 5.1 + 0.5),
            0.48
            + 0.045 * np.sin(2.0 * np.pi * time_s / 4.3 + 1.0),
        ]
    )
    rotation_vector = np.array(
        [
            0.55 * np.sin(2.0 * np.pi * time_s / 6.4),
            -0.48 * np.sin(2.0 * np.pi * time_s / 5.3 + 0.4),
            0.62 * np.sin(2.0 * np.pi * time_s / 7.7 - 0.3),
        ]
    )
    return translation, rotation_vector


def calibration_transition(state, _delta_t_s):
    """Stationary camera extrinsics and object pose."""
    return np.asarray(state).copy()


def _make_detection_observation(
    measured_camera_to_object_translation,
    measured_camera_to_object_rotation,
    base_to_wrist_translation,
    base_to_wrist_rotation,
):
    """Create a local 6-D SE(3) residual for one detected object pose.

    The Observation value is zero. The observation function returns the
    predicted-minus-measured pose error in the tangent space around the
    measured pose:

        [t_pred - t_meas,
         Log(R_meas.T @ R_pred)].

    This preserves the detector's translation and rotation noise scales
    directly rather than converting orientation into artificial point
    displacements with an arbitrary lever arm.
    """
    measured_rotation_matrix = so3_exp(
        measured_camera_to_object_rotation
    )

    def observation_func(state):
        (
            predicted_camera_to_object_translation,
            predicted_camera_to_object_rotation,
        ) = relative_pose(
            base_to_wrist_translation,
            base_to_wrist_rotation,
            state[CAMERA_TRANSLATION_SLICE],
            state[CAMERA_ROTATION_SLICE],
            state[OBJECT_TRANSLATION_SLICE],
            state[OBJECT_ROTATION_SLICE],
        )

        translation_error = (
            predicted_camera_to_object_translation
            - measured_camera_to_object_translation
        )
        rotation_error = so3_log(
            measured_rotation_matrix.T
            @ so3_exp(predicted_camera_to_object_rotation)
        )
        return np.concatenate(
            (translation_error, rotation_error)
        )

    measurement_covariance = np.diag(
        [
            *([TRANSLATION_NOISE_STD_M**2] * 3),
            *([ROTATION_NOISE_STD_RAD**2] * 3),
        ]
    )
    return Observation(
        np.zeros(6),
        measurement_covariance,
        observation_func,
    )


def simulate_wrist_camera_calibration(
    seed=2026,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
    detection_probability=DETECTION_PROBABILITY,
):
    """Generate known wrist poses and intermittent camera object detections."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(duration_s * rate_hz) + 1) / rate_hz

    wrist_translations = np.empty((len(times), 3))
    wrist_rotation_vectors = np.empty((len(times), 3))
    for index, time_s in enumerate(times):
        wrist_translations[index], wrist_rotation_vectors[index] = (
            wrist_pose_at_time(time_s)
        )

    detection_mask = rng.random(len(times)) < detection_probability
    for start_s, end_s in DETECTION_DROPOUTS_S:
        detection_mask &= ~(
            (times >= start_s) & (times <= end_s)
        )
    detection_mask[1] = True
    detection_mask[-2] = True

    detection_indices = np.flatnonzero(detection_mask)
    detection_indices = detection_indices[detection_indices > 0]

    observations = []
    detection_times = []
    measured_object_poses_camera = []
    true_object_poses_camera = []

    for index in detection_indices:
        (
            true_camera_to_object_translation,
            true_camera_to_object_rotation,
        ) = relative_pose(
            wrist_translations[index],
            wrist_rotation_vectors[index],
            TRUE_WRIST_TO_CAMERA_TRANSLATION_M,
            TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR,
            TRUE_BASE_TO_OBJECT_TRANSLATION_M,
            TRUE_BASE_TO_OBJECT_ROTATION_VECTOR,
        )

        measured_translation = (
            true_camera_to_object_translation
            + rng.normal(0.0, TRANSLATION_NOISE_STD_M, size=3)
        )
        rotation_noise = rng.normal(
            0.0,
            ROTATION_NOISE_STD_RAD,
            size=3,
        )
        measured_rotation = so3_log(
            so3_exp(true_camera_to_object_rotation)
            @ so3_exp(rotation_noise)
        )

        observations.append(
            _make_detection_observation(
                measured_translation,
                measured_rotation,
                wrist_translations[index].copy(),
                wrist_rotation_vectors[index].copy(),
            )
        )
        detection_times.append(times[index])
        measured_object_poses_camera.append(
            np.concatenate((measured_translation, measured_rotation))
        )
        true_object_poses_camera.append(
            np.concatenate(
                (
                    true_camera_to_object_translation,
                    true_camera_to_object_rotation,
                )
            )
        )

    true_state = np.concatenate(
        (
            TRUE_WRIST_TO_CAMERA_TRANSLATION_M,
            TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR,
            TRUE_BASE_TO_OBJECT_TRANSLATION_M,
            TRUE_BASE_TO_OBJECT_ROTATION_VECTOR,
        )
    )

    return {
        "times": times,
        "wrist_translations_m": wrist_translations,
        "wrist_rotation_vectors": wrist_rotation_vectors,
        "detection_mask": detection_mask,
        "detection_indices": detection_indices,
        "detection_times": np.asarray(detection_times),
        "observations": observations,
        "measured_object_poses_camera": np.asarray(
            measured_object_poses_camera
        ),
        "true_object_poses_camera": np.asarray(
            true_object_poses_camera
        ),
        "true_state": true_state,
        "rate_hz": rate_hz,
    }


def run_wrist_camera_calibration(
    seed=2026,
    duration_s=DEFAULT_DURATION_S,
    rate_hz=DEFAULT_RATE_HZ,
    detection_probability=DETECTION_PROBABILITY,
):
    """Estimate wrist-camera extrinsics and fixed base-frame object pose."""
    data = simulate_wrist_camera_calibration(
        seed=seed,
        duration_s=duration_s,
        rate_hz=rate_hz,
        detection_probability=detection_probability,
    )

    initial_camera_translation = (
        TRUE_WRIST_TO_CAMERA_TRANSLATION_M
        + np.array([-0.040, 0.035, 0.045])
    )
    initial_camera_rotation = so3_log(
        so3_exp(TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR)
        @ so3_exp(np.deg2rad(np.array([9.0, -6.0, 7.0])))
    )

    initial_object_translation = (
        TRUE_BASE_TO_OBJECT_TRANSLATION_M
        + np.array([0.080, -0.060, 0.050])
    )
    initial_object_rotation = so3_log(
        so3_exp(TRUE_BASE_TO_OBJECT_ROTATION_VECTOR)
        @ so3_exp(np.deg2rad(np.array([-10.0, 8.0, 12.0])))
    )

    initial_mean = np.concatenate(
        (
            initial_camera_translation,
            initial_camera_rotation,
            initial_object_translation,
            initial_object_rotation,
        )
    )
    initial_covariance = np.diag(
        [
            *([0.08**2] * 3),
            *([np.deg2rad(15.0) ** 2] * 3),
            *([0.12**2] * 3),
            *([np.deg2rad(20.0) ** 2] * 3),
        ]
    )

    model = StateTransitionModel(
        calibration_transition,
        # The mount and table object are static. Retain only a tiny numerical
        # floor instead of modeling either pose as a meaningful random walk.
        np.diag([1.0e-12] * 12),
    )

    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states, filter_times = bayes_filter.run(
        data["observations"],
        data["detection_times"],
        rate_hz=rate_hz,
        use_jacobian=False,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
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
    true_state = data["true_state"]

    def camera_translation_error(states):
        return np.linalg.norm(
            states[:, CAMERA_TRANSLATION_SLICE]
            - true_state[CAMERA_TRANSLATION_SLICE],
            axis=1,
        )

    def camera_rotation_error(states):
        return np.array(
            [
                rotation_distance(
                    true_state[CAMERA_ROTATION_SLICE],
                    state[CAMERA_ROTATION_SLICE],
                )
                for state in states
            ]
        )

    def object_translation_error(states):
        return np.linalg.norm(
            states[:, OBJECT_TRANSLATION_SLICE]
            - true_state[OBJECT_TRANSLATION_SLICE],
            axis=1,
        )

    def object_rotation_error(states):
        return np.array(
            [
                rotation_distance(
                    true_state[OBJECT_ROTATION_SLICE],
                    state[OBJECT_ROTATION_SLICE],
                )
                for state in states
            ]
        )

    return {
        **data,
        "filter_times": np.asarray(filter_times),
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "initial_mean": initial_mean,
        "filtered_camera_translation_error_m":
            camera_translation_error(filtered),
        "smoothed_camera_translation_error_m":
            camera_translation_error(smoothed),
        "filtered_camera_rotation_error_rad":
            camera_rotation_error(filtered),
        "smoothed_camera_rotation_error_rad":
            camera_rotation_error(smoothed),
        "filtered_object_translation_error_m":
            object_translation_error(filtered),
        "smoothed_object_translation_error_m":
            object_translation_error(smoothed),
        "filtered_object_rotation_error_rad":
            object_rotation_error(filtered),
        "smoothed_object_rotation_error_rad":
            object_rotation_error(smoothed),
        "final_filtered_camera_translation_m":
            filtered[-1, CAMERA_TRANSLATION_SLICE].copy(),
        "final_filtered_camera_rotation_vector":
            filtered[-1, CAMERA_ROTATION_SLICE].copy(),
        "true_camera_translation_m":
            TRUE_WRIST_TO_CAMERA_TRANSLATION_M.copy(),
        "true_camera_rotation_vector":
            TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR.copy(),
        "detection_fraction": float(
            len(data["detection_indices"])
            / (len(data["times"]) - 1)
        ),
    }


def plot_wrist_camera_calibration(result, output_path=None):
    """Plot known robot motion, intermittent detections, and convergence."""
    import matplotlib.pyplot as plt

    filter_times = result["filter_times"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        result["wrist_translations_m"][:, 0],
        result["wrist_translations_m"][:, 1],
        label="known wrist xy trajectory",
    )
    detection_indices = result["detection_indices"]
    axes[0, 0].scatter(
        result["wrist_translations_m"][detection_indices, 0],
        result["wrist_translations_m"][detection_indices, 1],
        s=10,
        alpha=0.55,
        label="frames with object detection",
    )
    axes[0, 0].set_xlabel("wrist x [m]")
    axes[0, 0].set_ylabel("wrist y [m]")
    axes[0, 0].set_title(
        "Known robot trajectory with intermittent detections"
    )
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        filter_times,
        1000.0 * result["filtered_camera_translation_error_m"],
        label="filtered extrinsic translation error",
    )
    axes[0, 1].plot(
        filter_times,
        1000.0 * result["smoothed_camera_translation_error_m"],
        label="RTS extrinsic translation error",
    )
    axes[0, 1].set_ylabel("translation error [mm]")
    axes[0, 1].set_title(
        "Wrist-to-camera translation calibration"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        filter_times,
        np.rad2deg(
            result["filtered_camera_rotation_error_rad"]
        ),
        label="filtered extrinsic rotation error",
    )
    axes[1, 0].plot(
        filter_times,
        np.rad2deg(
            result["smoothed_camera_rotation_error_rad"]
        ),
        label="RTS extrinsic rotation error",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("rotation error [deg]")
    axes[1, 0].set_title(
        "Wrist-to-camera orientation calibration"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        filter_times,
        1000.0 * result["filtered_object_translation_error_m"],
        label="object translation error [mm]",
    )
    axes[1, 1].plot(
        filter_times,
        np.rad2deg(
            result["filtered_object_rotation_error_rad"]
        ),
        label="object rotation error [deg]",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("error [mm / deg]")
    axes[1, 1].set_title(
        "Base-frame object pose is learned jointly"
    )
    axes[1, 1].legend(fontsize=8)

    for start_s, end_s in DETECTION_DROPOUTS_S:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.axvspan(start_s, end_s, alpha=0.08)

    for axis in axes.flat:
        axis.grid(alpha=0.2)

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)

    return figure


def _print_summary(result):
    print("Wrist-camera hand-eye calibration")
    print(
        f"  detection fraction: "
        f"{100.0 * result['detection_fraction']:.1f}%"
    )
    print(
        f"  final extrinsic translation error: "
        f"{1000.0 * result['filtered_camera_translation_error_m'][-1]:.2f} mm"
    )
    print(
        f"  final extrinsic rotation error: "
        f"{np.rad2deg(result['filtered_camera_rotation_error_rad'][-1]):.3f} deg"
    )
    print(
        "  estimated wrist->camera translation [m]: "
        f"{result['final_filtered_camera_translation_m']}"
    )


if __name__ == "__main__":
    result = run_wrist_camera_calibration()
    path = (
        Path(__file__).parent
        / "figures"
        / "wrist-camera-hand-eye-calibration.png"
    )
    plot_wrist_camera_calibration(result, path)
    _print_summary(result)
    print(f"saved {path}")

"""Calibrate a MEMS IMU and gyrocompass from stationary multi-position data.

A stationary IMU senses two inertial vectors: the accelerometer sees specific
force opposite local gravity, and the gyroscope sees the Earth's rotation
vector. The Earth-rate vector's horizontal component points true north while
its vertical component depends on latitude.

Realistic MEMS gyro bias can be comparable to, or larger than, the 15 deg/hour
Earth-rate signal. A single stationary pose cannot separate constant sensor
bias from the inertial vector. This example therefore puts the IMU in several
known relative orientations using a keyed cube/fixture. Gravity and Earth rate
rotate in sensor coordinates while the sensor biases remain attached to the
sensor axes.

The filter state is twelve dimensional:
gravity_base(3), earth_rate_base(3), gyro_bias(3), accel_bias(3).

Latitude and true-north heading are derived from the inferred gravity and
Earth-rate vectors. The first three placements intentionally rotate about only
one fixture axis, so their stacked observation matrix remains rank deficient.
The fourth placement changes rotation axis, makes the calibration fully
observable, and collapses latitude/heading uncertainty.

Noise values are based on the Analog Devices ADIS16495-1 tactical-grade MEMS
IMU class: approximately 0.002 deg/s/sqrt(Hz) gyro noise density, 0.8 deg/hour
in-run gyro bias stability, 16 ug/sqrt(Hz) accelerometer noise density, and
3.2 ug in-run accelerometer bias stability. The synthetic turn-on bias is
deliberately larger than Earth rate and a small per-dwell bias random walk is
included.

References:
https://www.analog.com/en/products/adis16495.html
https://www.analog.com/media/en/technical-documentation/data-sheets/ADIS16495.pdf
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


GRAVITY_X = 0
GRAVITY_Y = 1
GRAVITY_Z = 2
EARTH_X = 3
EARTH_Y = 4
EARTH_Z = 5
GYRO_BIAS_X = 6
GYRO_BIAS_Y = 7
GYRO_BIAS_Z = 8
ACCEL_BIAS_X = 9
ACCEL_BIAS_Y = 10
ACCEL_BIAS_Z = 11
STATE_SIZE = 12

EARTH_RATE_DEG_HR = 15.041067
TRUE_LATITUDE_DEG = 47.0
TRUE_HEADING_DEG = 63.0
TRUE_ROLL_DEG = 8.0
TRUE_PITCH_DEG = -11.0

GYRO_NOISE_DENSITY_DEG_S_SQRT_HZ = 0.002
GYRO_BIAS_STABILITY_DEG_HR = 0.8
ACCEL_NOISE_DENSITY_UG_SQRT_HZ = 16.0
ACCEL_BIAS_STABILITY_UG = 3.2

DEFAULT_DWELL_S = 90.0
TRUE_GYRO_BIAS_DEG_HR = np.array([28.0, -17.0, 11.0])
TRUE_ACCEL_BIAS_G = 1e-6 * np.array([600.0, -400.0, 800.0])

GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL = 0.08
ACCEL_BIAS_RW_STD_G_PER_DWELL = 0.5e-6


def rotation_x(angle_rad):
    """Passive coordinate rotation about x."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, sine],
            [0.0, -sine, cosine],
        ]
    )


def rotation_y(angle_rad):
    """Passive coordinate rotation about y."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, 0.0, -sine],
            [0.0, 1.0, 0.0],
            [sine, 0.0, cosine],
        ]
    )


def rotation_z(angle_rad):
    """Passive coordinate rotation about z."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def navigation_to_base_rotation(roll_rad, pitch_rad, heading_rad):
    """Rotate North-East-Down vectors into the reference IMU orientation."""
    return (
        rotation_x(roll_rad)
        @ rotation_y(pitch_rad)
        @ rotation_z(heading_rad)
    )


POSE_NAMES = (
    "base",
    "+90 roll",
    "-90 roll",
    "+90 pitch",
    "-90 pitch",
    "upside down",
    "+90 yaw",
    "mixed",
)
POSE_ROTATIONS = (
    np.eye(3),
    rotation_x(np.deg2rad(90.0)),
    rotation_x(np.deg2rad(-90.0)),
    rotation_y(np.deg2rad(90.0)),
    rotation_y(np.deg2rad(-90.0)),
    rotation_x(np.deg2rad(180.0)),
    rotation_z(np.deg2rad(90.0)),
    rotation_y(np.deg2rad(90.0)) @ rotation_z(np.deg2rad(90.0)),
)


def true_inertial_vectors():
    """Return gravity-down and Earth-rate vectors in the base sensor frame."""
    latitude_rad = np.deg2rad(TRUE_LATITUDE_DEG)
    base_rotation = navigation_to_base_rotation(
        np.deg2rad(TRUE_ROLL_DEG),
        np.deg2rad(TRUE_PITCH_DEG),
        np.deg2rad(TRUE_HEADING_DEG),
    )
    gravity_down_base = base_rotation @ np.array([0.0, 0.0, 1.0])
    earth_rate_ned = EARTH_RATE_DEG_HR * np.array(
        [np.cos(latitude_rad), 0.0, -np.sin(latitude_rad)]
    )
    earth_rate_base = base_rotation @ earth_rate_ned
    return gravity_down_base, earth_rate_base


def dwell_average_white_noise_std(noise_density, dwell_s):
    """RMS of an ideal boxcar average for white noise density per sqrt(Hz)."""
    return noise_density / np.sqrt(2.0 * dwell_s)


def gyro_dwell_white_std_deg_hr(dwell_s=DEFAULT_DWELL_S):
    """White gyro noise on one stationary dwell average."""
    return 3600.0 * dwell_average_white_noise_std(
        GYRO_NOISE_DENSITY_DEG_S_SQRT_HZ,
        dwell_s,
    )


def accel_dwell_white_std_g(dwell_s=DEFAULT_DWELL_S):
    """White accelerometer noise on one stationary dwell average."""
    return 1e-6 * dwell_average_white_noise_std(
        ACCEL_NOISE_DENSITY_UG_SQRT_HZ,
        dwell_s,
    )


def calibration_observation_matrix(pose_rotation):
    """Map calibration state to averaged gyro plus accelerometer readings."""
    pose_rotation = np.asarray(pose_rotation)
    matrix = np.zeros((6, STATE_SIZE))
    matrix[:3, EARTH_X : EARTH_Z + 1] = pose_rotation
    matrix[:3, GYRO_BIAS_X : GYRO_BIAS_Z + 1] = np.eye(3)
    matrix[3:, GRAVITY_X : GRAVITY_Z + 1] = -pose_rotation
    matrix[3:, ACCEL_BIAS_X : ACCEL_BIAS_Z + 1] = np.eye(3)
    return matrix


def make_calibration_observation_model(pose_rotation):
    """Close one known fixture orientation into an observation model."""
    matrix = calibration_observation_matrix(pose_rotation)

    def observe(state):
        return matrix @ np.asarray(state)

    def jacobian(_state):
        return matrix

    return observe, jacobian


def propagate_calibration_state(state, _delta_t_s):
    """Keep inertial vectors fixed while biases execute a small random walk."""
    return np.asarray(state).copy()


def calibration_transition_jacobian(_state, _delta_t_s):
    return np.eye(STATE_SIZE)


def derive_latitude_heading(state):
    """Recover latitude and true-north heading from inferred inertial vectors."""
    state = np.asarray(state)
    down = state[GRAVITY_X : GRAVITY_Z + 1]
    earth_rate = state[EARTH_X : EARTH_Z + 1]

    down = down / np.linalg.norm(down)
    earth_unit = earth_rate / np.linalg.norm(earth_rate)

    latitude_rad = -np.arcsin(
        np.clip(np.dot(earth_unit, down), -1.0, 1.0)
    )

    horizontal_earth = earth_rate - np.dot(earth_rate, down) * down
    north = horizontal_earth / np.linalg.norm(horizontal_earth)
    east = np.cross(down, north)

    heading_rad = np.arctan2(east[0], north[0])
    return float(np.rad2deg(latitude_rad)), float(np.rad2deg(heading_rad) % 360.0)


def angle_difference_deg(angle, reference):
    """Signed wrapped angular difference in degrees."""
    return float((angle - reference + 180.0) % 360.0 - 180.0)


def navigation_jacobian(state):
    """Numerical Jacobian of latitude and heading in degrees."""
    state = np.asarray(state)
    jacobian = np.zeros((2, STATE_SIZE))
    for index in range(6):
        step = 1e-6 if index < 3 else 1e-4
        offset = np.zeros(STATE_SIZE)
        offset[index] = step
        plus = np.array(derive_latitude_heading(state + offset))
        minus = np.array(derive_latitude_heading(state - offset))
        difference = plus - minus
        difference[1] = angle_difference_deg(plus[1], minus[1])
        jacobian[:, index] = difference / (2.0 * step)
    return jacobian


def navigation_std_deg(state, covariance):
    """Approximate one-sigma latitude and heading uncertainty."""
    jacobian = navigation_jacobian(state)
    navigation_covariance = jacobian @ covariance @ jacobian.T
    return np.sqrt(np.maximum(np.diag(navigation_covariance), 0.0))


def stacked_observation_rank(num_poses):
    """Rank of the stacked observation matrix for the first poses."""
    matrix = np.vstack(
        [
            calibration_observation_matrix(pose_rotation)
            for pose_rotation in POSE_ROTATIONS[:num_poses]
        ]
    )
    return int(np.linalg.matrix_rank(matrix))


def simulate_stationary_imu_session(seed=0, dwell_s=DEFAULT_DWELL_S):
    """Generate realistic averaged stationary IMU measurements."""
    rng = np.random.default_rng(seed)
    gravity_down_base, earth_rate_base = true_inertial_vectors()

    gyro_white_std = gyro_dwell_white_std_deg_hr(dwell_s)
    accel_white_std = accel_dwell_white_std_g(dwell_s)

    gyro_bias = TRUE_GYRO_BIAS_DEG_HR.copy()
    accel_bias = TRUE_ACCEL_BIAS_G.copy()

    measurements = []
    gyro_bias_truth = []
    accel_bias_truth = []

    for pose_rotation in POSE_ROTATIONS:
        gyro_bias = gyro_bias + rng.normal(
            0.0, GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL, 3
        )
        accel_bias = accel_bias + rng.normal(
            0.0, ACCEL_BIAS_RW_STD_G_PER_DWELL, 3
        )
        state = np.concatenate(
            [gravity_down_base, earth_rate_base, gyro_bias, accel_bias]
        )
        expected = calibration_observation_matrix(pose_rotation) @ state
        noise = np.concatenate(
            [
                rng.normal(0.0, gyro_white_std, 3),
                rng.normal(0.0, accel_white_std, 3),
            ]
        )
        measurements.append(expected + noise)
        gyro_bias_truth.append(gyro_bias.copy())
        accel_bias_truth.append(accel_bias.copy())

    times = np.arange(len(POSE_ROTATIONS) + 1, dtype=float) * dwell_s
    return {
        "times": times,
        "measurements": np.asarray(measurements),
        "gyro_bias_truth_deg_hr": np.asarray(gyro_bias_truth),
        "accel_bias_truth_g": np.asarray(accel_bias_truth),
        "gravity_down_base": gravity_down_base,
        "earth_rate_base_deg_hr": earth_rate_base,
        "gyro_white_std_deg_hr": gyro_white_std,
        "accel_white_std_g": accel_white_std,
        "dwell_s": dwell_s,
    }


def run_gyrocompass_calibration(
    seed=0,
    dwell_s=DEFAULT_DWELL_S,
    use_jacobian=True,
):
    """Estimate latitude, north, and IMU biases from stationary placements."""
    data = simulate_stationary_imu_session(seed=seed, dwell_s=dwell_s)

    measurement_covariance = np.diag(
        [data["gyro_white_std_deg_hr"] ** 2] * 3
        + [data["accel_white_std_g"] ** 2] * 3
    )

    observations = []
    for measurement, pose_rotation in zip(
        data["measurements"], POSE_ROTATIONS
    ):
        observation_func, jacobian_func = make_calibration_observation_model(
            pose_rotation
        )
        observations.append(
            Observation(
                measurement,
                measurement_covariance,
                observation_func,
                jacobian_func,
            )
        )

    process_covariance = np.zeros((STATE_SIZE, STATE_SIZE))
    process_covariance[:3, :3] = 1e-12 * np.eye(3)
    process_covariance[3:6, 3:6] = 1e-8 * np.eye(3)
    process_covariance[6:9, 6:9] = (
        GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL**2 * np.eye(3)
    )
    process_covariance[9:12, 9:12] = (
        ACCEL_BIAS_RW_STD_G_PER_DWELL**2 * np.eye(3)
    )

    transition_model = StateTransitionModel(
        propagate_calibration_state,
        process_covariance,
        calibration_transition_jacobian,
    )

    initial_mean = np.concatenate(
        [
            np.array([0.0, 0.0, 1.0]),
            np.array([EARTH_RATE_DEG_HR, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
        ]
    )
    initial_std = np.concatenate(
        [
            np.full(3, 0.45),
            np.full(3, 12.0),
            np.full(3, 100.0),
            np.full(3, 0.002),
        ]
    )
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
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

    filtered = np.asarray([state.mean() for state in filter_states])
    smoothed = np.asarray([state.mean() for state in smoother_states])
    filtered_covariances = np.asarray(
        [state.covariance() for state in filter_states]
    )
    smoothed_covariances = np.asarray(
        [state.covariance() for state in smoother_states]
    )

    filtered_navigation = np.asarray(
        [derive_latitude_heading(state) for state in filtered]
    )
    smoothed_navigation = np.asarray(
        [derive_latitude_heading(state) for state in smoothed]
    )
    filtered_navigation_std = np.asarray(
        [
            navigation_std_deg(state, covariance)
            for state, covariance in zip(filtered, filtered_covariances)
        ]
    )
    smoothed_navigation_std = np.asarray(
        [
            navigation_std_deg(state, covariance)
            for state, covariance in zip(smoothed, smoothed_covariances)
        ]
    )

    latitude_error = filtered_navigation[:, 0] - TRUE_LATITUDE_DEG
    heading_error = np.asarray(
        [
            angle_difference_deg(heading, TRUE_HEADING_DEG)
            for heading in filtered_navigation[:, 1]
        ]
    )
    smoothed_latitude_error = (
        smoothed_navigation[:, 0] - TRUE_LATITUDE_DEG
    )
    smoothed_heading_error = np.asarray(
        [
            angle_difference_deg(heading, TRUE_HEADING_DEG)
            for heading in smoothed_navigation[:, 1]
        ]
    )

    final_true_gyro_bias = data["gyro_bias_truth_deg_hr"][-1]
    final_true_accel_bias = data["accel_bias_truth_g"][-1]

    return {
        **data,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_navigation": filtered_navigation,
        "smoothed_navigation": smoothed_navigation,
        "filtered_navigation_std_deg": filtered_navigation_std,
        "smoothed_navigation_std_deg": smoothed_navigation_std,
        "filtered_latitude_error_deg": latitude_error,
        "filtered_heading_error_deg": heading_error,
        "smoothed_latitude_error_deg": smoothed_latitude_error,
        "smoothed_heading_error_deg": smoothed_heading_error,
        "final_latitude_deg": float(filtered_navigation[-1, 0]),
        "final_heading_deg": float(filtered_navigation[-1, 1]),
        "final_latitude_error_deg": float(latitude_error[-1]),
        "final_heading_error_deg": float(heading_error[-1]),
        "final_gyro_bias_deg_hr": filtered[-1, 6:9].copy(),
        "final_gyro_bias_error_deg_hr": float(
            np.linalg.norm(filtered[-1, 6:9] - final_true_gyro_bias)
        ),
        "final_accel_bias_g": filtered[-1, 9:12].copy(),
        "final_accel_bias_error_ug": float(
            1e6
            * np.linalg.norm(filtered[-1, 9:12] - final_true_accel_bias)
        ),
        "same_axis_rank": stacked_observation_rank(3),
        "cross_axis_rank": stacked_observation_rank(4),
        "same_axis_latitude_error_deg": float(latitude_error[3]),
        "same_axis_heading_error_deg": float(heading_error[3]),
        "cross_axis_latitude_error_deg": float(latitude_error[4]),
        "cross_axis_heading_error_deg": float(heading_error[4]),
        "initial_smoothed_latitude_error_deg": float(
            smoothed_latitude_error[0]
        ),
        "initial_smoothed_heading_error_deg": float(
            smoothed_heading_error[0]
        ),
        "use_jacobian": use_jacobian,
    }


def plot_gyrocompass_calibration(result, output_path=None):
    """Plot gyrocompassing, bias calibration, and observability collapse."""
    import matplotlib.pyplot as plt

    times_min = result["times"] / 60.0
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(
        times_min,
        result["filtered_navigation"][:, 0],
        marker="o",
        label="filtered latitude",
    )
    axes[0, 0].plot(
        times_min,
        result["smoothed_navigation"][:, 0],
        label="RTS latitude",
    )
    axes[0, 0].axhline(
        TRUE_LATITUDE_DEG,
        linestyle="--",
        label="true latitude",
    )
    axes[0, 0].axvline(
        4.0 * result["dwell_s"] / 60.0,
        linestyle=":",
        label="first cross-axis placement",
    )
    axes[0, 0].set_ylabel("latitude [deg]")
    axes[0, 0].set_title("Earth-rate tilt reveals latitude")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times_min,
        result["filtered_navigation"][:, 1],
        marker="o",
        label="filtered true-north heading",
    )
    axes[0, 1].plot(
        times_min,
        result["smoothed_navigation"][:, 1],
        label="RTS heading",
    )
    axes[0, 1].axhline(
        TRUE_HEADING_DEG,
        linestyle="--",
        label="true heading",
    )
    axes[0, 1].axvline(
        4.0 * result["dwell_s"] / 60.0,
        linestyle=":",
    )
    axes[0, 1].set_ylabel("heading [deg]")
    axes[0, 1].set_title("Horizontal Earth rate reveals true north")
    axes[0, 1].legend(fontsize=8)

    update_times_min = times_min[1:]
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axes[1, 0].plot(
            times_min,
            result["filtered"][:, GYRO_BIAS_X + axis_index],
            label=f"estimated b_{axis_name}",
        )
        axes[1, 0].plot(
            update_times_min,
            result["gyro_bias_truth_deg_hr"][:, axis_index],
            linestyle="--",
            alpha=0.7,
            label=f"true b_{axis_name}",
        )
    axes[1, 0].axhline(
        EARTH_RATE_DEG_HR,
        linestyle=":",
        label="Earth-rate magnitude",
    )
    axes[1, 0].set_xlabel("elapsed stationary time [min]")
    axes[1, 0].set_ylabel("gyro rate [deg/hr]")
    axes[1, 0].set_title("Gyro bias is larger than the signal being recovered")
    axes[1, 0].legend(ncol=2, fontsize=8)

    axes[1, 1].plot(
        times_min,
        result["filtered_navigation_std_deg"][:, 0],
        marker="o",
        label="latitude 1-sigma",
    )
    axes[1, 1].plot(
        times_min,
        result["filtered_navigation_std_deg"][:, 1],
        marker="o",
        label="heading 1-sigma",
    )
    axes[1, 1].axvline(
        3.0 * result["dwell_s"] / 60.0,
        linestyle="--",
        label=f"3 poses: rank {result['same_axis_rank']}/12",
    )
    axes[1, 1].axvline(
        4.0 * result["dwell_s"] / 60.0,
        linestyle=":",
        label=f"4 poses: rank {result['cross_axis_rank']}/12",
    )
    axes[1, 1].set_xlabel("elapsed stationary time [min]")
    axes[1, 1].set_ylabel("posterior standard deviation [deg]")
    axes[1, 1].set_title("Changing rotation axis makes calibration observable")
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
    print("Stationary multi-position MEMS gyrocompassing")
    print(
        f"  dwell-averaged gyro white noise: "
        f"{result['gyro_white_std_deg_hr']:.3f} deg/hr"
    )
    print(
        f"  true latitude/heading: "
        f"{TRUE_LATITUDE_DEG:.2f} deg / {TRUE_HEADING_DEG:.2f} deg"
    )
    print(
        f"  after three same-axis poses (rank {result['same_axis_rank']}/12): "
        f"latitude error {result['same_axis_latitude_error_deg']:.1f} deg, "
        f"heading error {result['same_axis_heading_error_deg']:.1f} deg"
    )
    print(
        f"  after cross-axis pose (rank {result['cross_axis_rank']}/12): "
        f"latitude error {result['cross_axis_latitude_error_deg']:.2f} deg, "
        f"heading error {result['cross_axis_heading_error_deg']:.2f} deg"
    )
    print(
        f"  final latitude: {result['final_latitude_deg']:.3f} deg "
        f"(error {result['final_latitude_error_deg']:.3f} deg)"
    )
    print(
        f"  final heading: {result['final_heading_deg']:.3f} deg "
        f"(error {result['final_heading_error_deg']:.3f} deg)"
    )
    print(
        f"  final gyro bias error: "
        f"{result['final_gyro_bias_error_deg_hr']:.3f} deg/hr"
    )
    print(
        f"  final accelerometer bias error: "
        f"{result['final_accel_bias_error_ug']:.2f} ug"
    )


if __name__ == "__main__":
    result = run_gyrocompass_calibration()
    path = Path(__file__).parent / "figures" / "gyrocompass-calibration.png"
    plot_gyrocompass_calibration(result, path)
    _print_summary(result)
    print(f"saved {path}")

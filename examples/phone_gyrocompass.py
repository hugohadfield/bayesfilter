"""Phone gyrocompassing from hand reorientation, gravity, and magnetometer data.

This example asks whether an ordinary consumer-phone IMU can recover latitude
and true north from the Earth's tiny rotation signal.

A static phone cannot reliably do this: consumer MEMS gyro bias and drift can
be comparable to or larger than the 15 deg/hour Earth-rate signal. Instead the
user repeatedly holds the phone still, moves it to an arbitrary new orientation,
and holds it still again.

During each stationary dwell:

* accelerometer/gravity gives local down in phone coordinates;
* a calibrated magnetometer gives the local magnetic-field direction;
* those two vectors determine phone orientation relative to a local
  magnetic-North/East/Down frame, without knowing true north;
* the averaged gyro sees the Earth-rate vector rotated into phone coordinates
  plus a slowly drifting sensor-frame gyro bias.

The Bayesian-filter state is

    [earth_rate_magnetic_frame(3), gyro_bias_phone_frame(3)].

From the inferred Earth-rate vector we derive:

* latitude from its vertical component;
* magnetic declination from the horizontal angle between magnetic north and
  the recovered true-north Earth-rate projection.

No GPS, map, Sun, stars, true-heading API, or geomagnetic field model is used.

The synthetic phone noise is based on the Google Pixel 7 Pro measurements in
"Smartphone MEMS Accelerometer and Gyroscope Measurement Errors: Laboratory
Testing and Analysis of the Effects on Positioning Performance" (Sensors 2023).
The reported mean gyro angular-random-walk coefficient is 0.00019018
(rad/s)/sqrt(Hz), and mean bias instability is 9.567e-6 rad/s. The example
uses 24 one-minute stationary dwells with arbitrary human-achievable poses,
a large unknown turn-on gyro bias, slow bias drift, and noisy gravity and
magnetometer vectors.

Reference:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10490716/
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


EARTH_X = 0
EARTH_Y = 1
EARTH_Z = 2
GYRO_BIAS_X = 3
GYRO_BIAS_Y = 4
GYRO_BIAS_Z = 5
STATE_SIZE = 6

EARTH_RATE_DEG_HR = 15.041067
TRUE_LATITUDE_DEG = 47.0
TRUE_DECLINATION_DEG = 12.0

# Pixel 7 Pro mean gyro error parameters reported in Sensors 2023.
PHONE_GYRO_ARW_RAD_S_SQRT_HZ = 0.00019018
PHONE_GYRO_BIAS_INSTABILITY_RAD_S = 9.567e-6

DEFAULT_DWELL_S = 60.0
DEFAULT_NUM_POSES = 24

# Deliberately much larger than Earth rate: this is an uncalibrated turn-on
# offset, not the Allan-variance bias-instability floor.
TRUE_INITIAL_GYRO_BIAS_DEG_HR = np.array([40.0, -28.0, 18.0])

# Slow drift between one-minute stationary dwells.
GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL = 0.2

# A calibrated phone magnetometer is assumed. These normalized-vector errors
# correspond to roughly degree-scale pose uncertainty in benign conditions.
GRAVITY_VECTOR_NOISE_STD = 0.001
MAGNETIC_VECTOR_NOISE_STD = 0.006
RESIDUAL_MAGNETOMETER_BIAS = np.array([0.0008, -0.0007, 0.0003])
MAGNETIC_INCLINATION_DEG = 65.0

# Pose uncertainty is folded into the gyro observation covariance because the
# filter consumes a pose estimate reconstructed from gravity + magnetometer.
POSE_DIRECTION_STD_DEG = 0.8


def rotation_x(angle_rad):
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
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def magnetic_frame_to_phone_rotation(roll_rad, pitch_rad, yaw_rad):
    """Return a passive rotation from magnetic NED into phone coordinates."""
    return (
        rotation_x(roll_rad)
        @ rotation_y(pitch_rad)
        @ rotation_z(yaw_rad)
    )


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def true_earth_rate_magnetic_frame():
    """Earth rate expressed in magnetic-North/East/Down coordinates."""
    latitude = np.deg2rad(TRUE_LATITUDE_DEG)
    declination = np.deg2rad(TRUE_DECLINATION_DEG)

    # Magnetic north is declination east of true north. Therefore the true
    # north horizontal Earth-rate projection has magnetic-frame components
    # [cos(D), -sin(D)].
    return EARTH_RATE_DEG_HR * np.array(
        [
            np.cos(latitude) * np.cos(declination),
            -np.cos(latitude) * np.sin(declination),
            -np.sin(latitude),
        ]
    )


def derive_latitude_declination(earth_rate):
    """Recover latitude and magnetic declination from an Earth-rate vector."""
    earth_rate = np.asarray(earth_rate)
    direction = earth_rate / np.linalg.norm(earth_rate)

    latitude = -np.arcsin(np.clip(direction[2], -1.0, 1.0))
    declination = np.arctan2(-direction[1], direction[0])
    return float(np.rad2deg(latitude)), float(np.rad2deg(declination))


def angle_difference_deg(angle, reference):
    return float((angle - reference + 180.0) % 360.0 - 180.0)


def phone_gyro_dwell_std_deg_hr(dwell_s=DEFAULT_DWELL_S):
    """White gyro noise on one ideal stationary boxcar average."""
    conversion = 180.0 / np.pi * 3600.0
    noise_density_deg_hr = PHONE_GYRO_ARW_RAD_S_SQRT_HZ * conversion
    return noise_density_deg_hr / np.sqrt(2.0 * dwell_s)


def phone_bias_instability_deg_hr():
    conversion = 180.0 / np.pi * 3600.0
    return PHONE_GYRO_BIAS_INSTABILITY_RAD_S * conversion


def estimate_pose_from_gravity_and_magnetometer(
    gravity_down_phone,
    magnetic_field_phone,
):
    """TRIAD-like pose estimate relative to magnetic North/East/Down."""
    down = unit(gravity_down_phone)
    magnetic_horizontal = (
        magnetic_field_phone
        - np.dot(magnetic_field_phone, down) * down
    )
    magnetic_north = unit(magnetic_horizontal)
    magnetic_east = unit(np.cross(down, magnetic_north))
    magnetic_north = unit(np.cross(magnetic_east, down))
    return np.column_stack([magnetic_north, magnetic_east, down])


def human_pose_rotations(seed=0, num_poses=DEFAULT_NUM_POSES):
    """Generate deterministic, diverse orientations achievable by hand."""
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(num_poses):
        roll = np.deg2rad(rng.uniform(-75.0, 75.0))
        pitch = np.deg2rad(rng.uniform(-65.0, 65.0))
        yaw = np.deg2rad(rng.uniform(-180.0, 180.0))
        rotations.append(
            magnetic_frame_to_phone_rotation(roll, pitch, yaw)
        )
    return np.asarray(rotations)


def static_pose_rotations(num_poses=DEFAULT_NUM_POSES):
    """Control experiment: leave the phone in exactly one orientation."""
    rotation = magnetic_frame_to_phone_rotation(
        np.deg2rad(15.0),
        np.deg2rad(-10.0),
        np.deg2rad(30.0),
    )
    return np.repeat(rotation[None, :, :], num_poses, axis=0)


def stacked_observation_rank(pose_rotations):
    """Rank of the Earth-vector + sensor-bias measurement geometry."""
    matrices = []
    for rotation in pose_rotations:
        matrices.append(np.column_stack([rotation, np.eye(3)]))
    return int(np.linalg.matrix_rank(np.vstack(matrices)))


def simulate_phone_stationary_dwells(
    pose_rotations,
    seed=0,
    dwell_s=DEFAULT_DWELL_S,
):
    """Simulate gravity, magnetometer, and averaged gyro for each pose."""
    rng = np.random.default_rng(seed)
    earth_rate = true_earth_rate_magnetic_frame()

    inclination = np.deg2rad(MAGNETIC_INCLINATION_DEG)
    magnetic_field_magnetic_frame = np.array(
        [np.cos(inclination), 0.0, np.sin(inclination)]
    )
    gravity_down_magnetic_frame = np.array([0.0, 0.0, 1.0])

    gyro_white_std = phone_gyro_dwell_std_deg_hr(dwell_s)
    gyro_bias = TRUE_INITIAL_GYRO_BIAS_DEG_HR.copy()

    estimated_rotations = []
    gyro_measurements = []
    gyro_bias_truth = []
    gravity_measurements = []
    magnetic_measurements = []

    for rotation in pose_rotations:
        gravity_phone = (
            rotation @ gravity_down_magnetic_frame
            + rng.normal(0.0, GRAVITY_VECTOR_NOISE_STD, 3)
        )
        magnetic_phone = (
            rotation @ magnetic_field_magnetic_frame
            + RESIDUAL_MAGNETOMETER_BIAS
            + rng.normal(0.0, MAGNETIC_VECTOR_NOISE_STD, 3)
        )
        estimated_rotation = estimate_pose_from_gravity_and_magnetometer(
            gravity_phone,
            magnetic_phone,
        )

        gyro_bias = (
            gyro_bias
            + rng.normal(
                0.0,
                GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL,
                3,
            )
        )
        gyro_phone = (
            rotation @ earth_rate
            + gyro_bias
            + rng.normal(0.0, gyro_white_std, 3)
        )

        estimated_rotations.append(estimated_rotation)
        gyro_measurements.append(gyro_phone)
        gyro_bias_truth.append(gyro_bias.copy())
        gravity_measurements.append(gravity_phone)
        magnetic_measurements.append(magnetic_phone)

    return {
        "estimated_rotations": np.asarray(estimated_rotations),
        "gyro_measurements_deg_hr": np.asarray(gyro_measurements),
        "gyro_bias_truth_deg_hr": np.asarray(gyro_bias_truth),
        "gravity_measurements": np.asarray(gravity_measurements),
        "magnetic_measurements": np.asarray(magnetic_measurements),
        "gyro_white_std_deg_hr": gyro_white_std,
        "dwell_s": dwell_s,
    }


def propagate_phone_state(state, _delta_t_s):
    """Keep Earth rate fixed while allowing gyro bias to drift."""
    return np.asarray(state).copy()


def phone_transition_jacobian(_state, _delta_t_s):
    return np.eye(STATE_SIZE)


def make_gyro_observation_model(pose_rotation):
    """Map Earth rate + phone-frame gyro bias to one dwell-average reading."""
    matrix = np.column_stack([pose_rotation, np.eye(3)])

    def observe(state):
        return matrix @ np.asarray(state)

    def jacobian(_state):
        return matrix

    return observe, jacobian


def navigation_jacobian(state):
    """Numerical Jacobian of [latitude, declination] with respect to state."""
    state = np.asarray(state)
    jacobian = np.zeros((2, STATE_SIZE))
    for index in range(3):
        offset = np.zeros(STATE_SIZE)
        offset[index] = 1e-4
        plus = np.asarray(derive_latitude_declination(state[:3] + offset[:3]))
        minus = np.asarray(derive_latitude_declination(state[:3] - offset[:3]))
        difference = plus - minus
        difference[1] = angle_difference_deg(plus[1], minus[1])
        jacobian[:, index] = difference / (2e-4)
    return jacobian


def navigation_std_deg(state, covariance):
    jacobian = navigation_jacobian(state)
    nav_covariance = jacobian @ covariance @ jacobian.T
    return np.sqrt(np.maximum(np.diag(nav_covariance), 0.0))


def run_phone_sequence(
    pose_rotations,
    seed=0,
    dwell_s=DEFAULT_DWELL_S,
    use_jacobian=True,
):
    """Run one static or hand-reoriented phone gyrocompassing sequence."""
    data = simulate_phone_stationary_dwells(
        pose_rotations,
        seed=seed,
        dwell_s=dwell_s,
    )

    pose_error_equivalent = (
        EARTH_RATE_DEG_HR * np.deg2rad(POSE_DIRECTION_STD_DEG)
    )
    measurement_std = np.sqrt(
        data["gyro_white_std_deg_hr"] ** 2
        + pose_error_equivalent**2
    )
    measurement_covariance = measurement_std**2 * np.eye(3)

    observations = []
    for measurement, rotation in zip(
        data["gyro_measurements_deg_hr"],
        data["estimated_rotations"],
    ):
        observe, jacobian = make_gyro_observation_model(rotation)
        observations.append(
            Observation(
                measurement,
                measurement_covariance,
                observe,
                jacobian,
            )
        )

    transition_covariance = np.diag(
        [1e-8, 1e-8, 1e-8]
        + [GYRO_BIAS_RW_STD_DEG_HR_PER_DWELL**2] * 3
    )
    model = StateTransitionModel(
        propagate_phone_state,
        transition_covariance,
        phone_transition_jacobian,
    )

    initial_mean = np.array(
        [EARTH_RATE_DEG_HR, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    initial_std = np.array(
        [15.0, 15.0, 15.0, 100.0, 100.0, 100.0]
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
    )

    times = np.arange(len(pose_rotations) + 1, dtype=float) * dwell_s
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
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
        [derive_latitude_declination(state[:3]) for state in filtered]
    )
    smoothed_navigation = np.asarray(
        [derive_latitude_declination(state[:3]) for state in smoothed]
    )
    filtered_navigation_std = np.asarray(
        [
            navigation_std_deg(state, covariance)
            for state, covariance in zip(filtered, filtered_covariances)
        ]
    )

    return {
        **data,
        "times": times,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_navigation": filtered_navigation,
        "smoothed_navigation": smoothed_navigation,
        "filtered_navigation_std_deg": filtered_navigation_std,
        "geometry_rank": stacked_observation_rank(
            data["estimated_rotations"]
        ),
        "measurement_std_deg_hr": measurement_std,
    }


def run_phone_gyrocompass(
    seed=0,
    num_poses=DEFAULT_NUM_POSES,
    dwell_s=DEFAULT_DWELL_S,
    use_jacobian=True,
):
    """Compare a static phone with human multi-position gyrocompassing."""
    human_poses = human_pose_rotations(seed=seed, num_poses=num_poses)
    static_poses = static_pose_rotations(num_poses=num_poses)

    human = run_phone_sequence(
        human_poses,
        seed=seed + 100,
        dwell_s=dwell_s,
        use_jacobian=use_jacobian,
    )
    static = run_phone_sequence(
        static_poses,
        seed=seed + 200,
        dwell_s=dwell_s,
        use_jacobian=use_jacobian,
    )

    final_latitude = human["filtered_navigation"][-1, 0]
    final_declination = human["filtered_navigation"][-1, 1]
    final_true_bias = human["gyro_bias_truth_deg_hr"][-1]

    human["final_latitude_error_deg"] = float(
        final_latitude - TRUE_LATITUDE_DEG
    )
    human["final_declination_error_deg"] = angle_difference_deg(
        final_declination,
        TRUE_DECLINATION_DEG,
    )
    human["final_gyro_bias_error_deg_hr"] = float(
        np.linalg.norm(human["filtered"][-1, 3:6] - final_true_bias)
    )

    static_latitude = static["filtered_navigation"][-1, 0]
    static_declination = static["filtered_navigation"][-1, 1]
    static["final_latitude_error_deg"] = float(
        static_latitude - TRUE_LATITUDE_DEG
    )
    static["final_declination_error_deg"] = angle_difference_deg(
        static_declination,
        TRUE_DECLINATION_DEG,
    )

    return {
        "human": human,
        "static": static,
        "num_poses": num_poses,
        "dwell_s": dwell_s,
        "true_earth_rate_deg_hr": true_earth_rate_magnetic_frame(),
        "use_jacobian": use_jacobian,
    }


def plot_phone_gyrocompass(result, output_path=None):
    """Plot failure while static and success after arbitrary hand reorientation."""
    import matplotlib.pyplot as plt

    human = result["human"]
    static = result["static"]
    times_min = human["times"] / 60.0

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(
        times_min,
        human["filtered_navigation"][:, 0],
        marker="o",
        markersize=3,
        label="hand-reoriented phone",
    )
    axes[0, 0].plot(
        times_min,
        static["filtered_navigation"][:, 0],
        label="phone left static",
    )
    axes[0, 0].axhline(
        TRUE_LATITUDE_DEG,
        linestyle="--",
        label="true latitude",
    )
    axes[0, 0].set_ylabel("latitude [deg]")
    axes[0, 0].set_title("Latitude from the tilt of Earth's rotation vector")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times_min,
        human["filtered_navigation"][:, 1],
        marker="o",
        markersize=3,
        label="inferred magnetic declination",
    )
    axes[0, 1].plot(
        times_min,
        static["filtered_navigation"][:, 1],
        label="static-phone estimate",
    )
    axes[0, 1].axhline(
        TRUE_DECLINATION_DEG,
        linestyle="--",
        label="true declination",
    )
    axes[0, 1].set_ylabel("declination [deg]")
    axes[0, 1].set_title("Earth rotation turns magnetic heading into true north")
    axes[0, 1].legend(fontsize=8)

    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axes[1, 0].plot(
            times_min,
            human["filtered"][:, GYRO_BIAS_X + axis_index],
            label=f"estimated b_{axis_name}",
        )
        axes[1, 0].plot(
            times_min[1:],
            human["gyro_bias_truth_deg_hr"][:, axis_index],
            linestyle="--",
            alpha=0.7,
            label=f"true b_{axis_name}",
        )
    axes[1, 0].axhline(
        EARTH_RATE_DEG_HR,
        linestyle=":",
        label="Earth-rate magnitude",
    )
    axes[1, 0].set_xlabel("stationary dwell time [min]")
    axes[1, 0].set_ylabel("angular rate [deg/hr]")
    axes[1, 0].set_title("Large phone gyro bias is estimated alongside Earth rate")
    axes[1, 0].legend(ncol=2, fontsize=8)

    axes[1, 1].plot(
        times_min,
        human["filtered_navigation_std_deg"][:, 0],
        label="hand-reoriented latitude sigma",
    )
    axes[1, 1].plot(
        times_min,
        human["filtered_navigation_std_deg"][:, 1],
        label="hand-reoriented declination sigma",
    )
    axes[1, 1].plot(
        times_min,
        static["filtered_navigation_std_deg"][:, 0],
        linestyle="--",
        label="static latitude sigma",
    )
    axes[1, 1].set_xlabel("stationary dwell time [min]")
    axes[1, 1].set_ylabel("posterior std [deg]")
    axes[1, 1].set_title(
        f"geometry rank: static {static['geometry_rank']}/6, "
        f"hand {human['geometry_rank']}/6"
    )
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
    human = result["human"]
    static = result["static"]

    print("Phone gyrocompassing with arbitrary hand reorientation")
    print(
        f"  gyro dwell white noise: "
        f"{human['gyro_white_std_deg_hr']:.2f} deg/hr"
    )
    print(
        f"  published Pixel-class bias-instability scale: "
        f"{phone_bias_instability_deg_hr():.2f} deg/hr"
    )
    print(
        f"  static geometry rank: {static['geometry_rank']}/6; "
        f"hand-reoriented rank: {human['geometry_rank']}/6"
    )
    print(
        f"  latitude: {human['filtered_navigation'][-1, 0]:.2f} deg "
        f"(truth {TRUE_LATITUDE_DEG:.2f}, "
        f"error {human['final_latitude_error_deg']:.2f})"
    )
    print(
        f"  magnetic declination: "
        f"{human['filtered_navigation'][-1, 1]:.2f} deg "
        f"(truth {TRUE_DECLINATION_DEG:.2f}, "
        f"error {human['final_declination_error_deg']:.2f})"
    )
    print(
        f"  final gyro-bias vector error: "
        f"{human['final_gyro_bias_error_deg_hr']:.2f} deg/hr"
    )


if __name__ == "__main__":
    result = run_phone_gyrocompass()
    path = Path(__file__).parent / "figures" / "phone-gyrocompass.png"
    plot_phone_gyrocompass(result, path)
    _print_summary(result)
    print(f"saved {path}")

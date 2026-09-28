"""Fuse camera bearing with radar range/range-rate for 2-D target tracking.

This is a deliberately traditional complementary-sensor fusion example.

State:
    [x, y, vx, vy]

Camera:
    high-rate bearing only, represented as a unit vector

Radar:
    lower-rate range and radial velocity

The camera constrains direction very well but supplies no instantaneous range.
The radar constrains range and range-rate well but has no bearing measurement.
Together they provide complementary geometry.

The simulated target follows a smooth maneuvering trajectory while the filter
uses a constant-velocity model with white-acceleration process noise.

Two sensor dropouts make the geometry visible:

- during the camera dropout, radar continues to constrain radial motion while
  tangential uncertainty grows;
- during the radar dropout, camera continues to constrain bearing while range
  uncertainty grows.

The example also runs camera-only and radar-only baselines from the same prior
to quantify the value of fusion.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS

from examples.linear_tracking import (
    constant_velocity_matrix,
    white_acceleration_covariance,
)


CAMERA_POSITION_M = np.array([-6.0, -3.0])
RADAR_POSITION_M = np.array([7.0, -5.0])

CAMERA_DROPOUT_S = (6.0, 9.0)
RADAR_DROPOUT_S = (12.0, 15.0)

DEFAULT_DURATION_S = 20.0
DEFAULT_FILTER_RATE_HZ = 20.0
DEFAULT_CAMERA_RATE_HZ = 10.0
DEFAULT_RADAR_RATE_HZ = 2.0

CAMERA_BEARING_STD_RAD = np.deg2rad(0.8)
RADAR_RANGE_STD_M = 0.15
RADAR_RANGE_RATE_STD_M_S = 0.12


def observe_camera_bearing(state, camera_position=CAMERA_POSITION_M):
    """Return unit-vector bearing from camera to target."""
    displacement = np.asarray(state)[:2] - np.asarray(camera_position)
    return displacement / np.linalg.norm(displacement)


def camera_bearing_jacobian(state, camera_position=CAMERA_POSITION_M):
    """Jacobian of the 2-D unit-vector camera bearing."""
    displacement = np.asarray(state)[:2] - np.asarray(camera_position)
    distance = np.linalg.norm(displacement)
    direction = displacement / distance

    jacobian = np.zeros((2, 4))
    jacobian[:, :2] = (
        np.eye(2) - np.outer(direction, direction)
    ) / distance
    return jacobian


def observe_radar_range_rate(state, radar_position=RADAR_POSITION_M):
    """Return radar range and radial velocity."""
    state = np.asarray(state)
    displacement = state[:2] - np.asarray(radar_position)
    distance = np.linalg.norm(displacement)
    radial_velocity = displacement @ state[2:4] / distance
    return np.array([distance, radial_velocity])


def radar_range_rate_jacobian(
    state,
    radar_position=RADAR_POSITION_M,
):
    """Jacobian of radar range and range-rate observation."""
    state = np.asarray(state)
    displacement = state[:2] - np.asarray(radar_position)
    velocity = state[2:4]

    distance = np.linalg.norm(displacement)
    direction = displacement / distance
    radial_velocity = displacement @ velocity / distance

    jacobian = np.zeros((2, 4))
    jacobian[0, :2] = direction
    jacobian[1, :2] = (
        velocity / distance
        - radial_velocity * displacement / distance**2
    )
    jacobian[1, 2:4] = direction
    return jacobian


def simulate_radar_camera_truth(times):
    """Generate a deterministic maneuvering target trajectory."""
    times = np.asarray(times)

    truth = np.zeros((len(times), 4))
    truth[0] = np.array([-8.0, 2.0, 1.6, 0.45])

    for index, delta_t_s in enumerate(np.diff(times)):
        midpoint_s = times[index] + 0.5 * delta_t_s

        acceleration = np.array(
            [
                0.18 * np.sin(0.55 * midpoint_s)
                + 0.25
                * np.exp(
                    -0.5 * ((midpoint_s - 8.0) / 1.1) ** 2
                )
                - 0.22
                * np.exp(
                    -0.5 * ((midpoint_s - 15.0) / 1.0) ** 2
                ),
                0.25 * np.cos(0.42 * midpoint_s)
                + 0.28
                * np.exp(
                    -0.5 * ((midpoint_s - 11.0) / 1.3) ** 2
                ),
            ]
        )

        truth[index + 1, :2] = (
            truth[index, :2]
            + delta_t_s * truth[index, 2:4]
            + 0.5 * delta_t_s**2 * acceleration
        )
        truth[index + 1, 2:4] = (
            truth[index, 2:4]
            + delta_t_s * acceleration
        )

    return truth


def _camera_measurement(truth_state, rng):
    displacement = (
        truth_state[:2] - CAMERA_POSITION_M
    )
    true_angle = np.arctan2(
        displacement[1],
        displacement[0],
    )
    measured_angle = (
        true_angle
        + rng.normal(0.0, CAMERA_BEARING_STD_RAD)
    )
    return np.array(
        [
            np.cos(measured_angle),
            np.sin(measured_angle),
        ]
    )


def _radar_measurement(truth_state, rng):
    return (
        observe_radar_range_rate(truth_state)
        + rng.normal(
            0.0,
            [
                RADAR_RANGE_STD_M,
                RADAR_RANGE_RATE_STD_M_S,
            ],
        )
    )


def _make_sensor_events(
    truth,
    times,
    rng,
    filter_rate_hz,
    camera_rate_hz,
    radar_rate_hz,
):
    camera_interval = round(filter_rate_hz / camera_rate_hz)
    radar_interval = round(filter_rate_hz / radar_rate_hz)

    if not np.isclose(
        camera_interval * camera_rate_hz,
        filter_rate_hz,
    ):
        raise ValueError(
            "camera_rate_hz must divide filter_rate_hz"
        )
    if not np.isclose(
        radar_interval * radar_rate_hz,
        filter_rate_hz,
    ):
        raise ValueError(
            "radar_rate_hz must divide filter_rate_hz"
        )

    camera_covariance = (
        np.sin(CAMERA_BEARING_STD_RAD) ** 2
        * np.eye(2)
    )
    radar_covariance = np.diag(
        [
            RADAR_RANGE_STD_M**2,
            RADAR_RANGE_RATE_STD_M_S**2,
        ]
    )

    records = []
    camera_measurements = []
    camera_times = []
    radar_measurements = []
    radar_times = []

    for index in range(1, len(times)):
        time_s = times[index]

        camera_due = (
            (index - 1) % camera_interval == 0
        )
        camera_available = not (
            CAMERA_DROPOUT_S[0]
            <= time_s
            <= CAMERA_DROPOUT_S[1]
        )

        if camera_due and camera_available:
            measurement = _camera_measurement(
                truth[index],
                rng,
            )
            camera_measurements.append(measurement)
            camera_times.append(time_s)
            records.append(
                (
                    time_s,
                    0,
                    "camera",
                    Observation(
                        measurement,
                        camera_covariance,
                        observe_camera_bearing,
                        camera_bearing_jacobian,
                    ),
                )
            )

        radar_due = (
            (index - 1) % radar_interval == 0
        )
        radar_available = not (
            RADAR_DROPOUT_S[0]
            <= time_s
            <= RADAR_DROPOUT_S[1]
        )

        if radar_due and radar_available:
            measurement = _radar_measurement(
                truth[index],
                rng,
            )
            radar_measurements.append(measurement)
            radar_times.append(time_s)
            records.append(
                (
                    time_s,
                    1,
                    "radar",
                    Observation(
                        measurement,
                        radar_covariance,
                        observe_radar_range_rate,
                        radar_range_rate_jacobian,
                    ),
                )
            )

    records.sort(key=lambda record: (record[0], record[1]))

    return {
        "records": records,
        "camera_measurements": np.asarray(
            camera_measurements
        ),
        "camera_times": np.asarray(camera_times),
        "radar_measurements": np.asarray(
            radar_measurements
        ),
        "radar_times": np.asarray(radar_times),
    }


def _initial_distribution(truth):
    return Gaussian(
        truth[0]
        + np.array([1.5, -1.2, 0.35, -0.25]),
        np.diag(
            [
                2.0**2,
                2.0**2,
                0.7**2,
                0.7**2,
            ]
        ),
    )


def _run_events(
    truth,
    records,
    filter_rate_hz,
    sensor_kind=None,
):
    """Run fusion or one sensor-only baseline."""
    selected = [
        record
        for record in records
        if (
            sensor_kind is None
            or record[2] == sensor_kind
        )
    ]

    observations = [record[3] for record in selected]
    event_times = np.array(
        [record[0] for record in selected]
    )

    delta_t_s = 1.0 / filter_rate_hz

    def transition(state, step_s):
        return (
            constant_velocity_matrix(2, step_s)
            @ state
        )

    def transition_jacobian(_state, step_s):
        return constant_velocity_matrix(2, step_s)

    model = StateTransitionModel(
        transition,
        white_acceleration_covariance(
            2,
            delta_t_s,
            spectral_density=0.35,
        ),
        transition_jacobian,
    )

    bayes_filter = BayesianFilter(
        model,
        _initial_distribution(truth),
    )
    filter_states, filter_times = bayes_filter.run(
        observations,
        event_times,
        rate_hz=filter_rate_hz,
        use_jacobian=True,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
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

    return {
        "times": np.asarray(filter_times),
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
    }


def _align_truth(full_times, truth, filter_times):
    indices = np.rint(
        filter_times
        / (full_times[1] - full_times[0])
    ).astype(int)
    indices = np.clip(indices, 0, len(truth) - 1)
    return truth[indices]


def run_radar_camera_fusion(
    seed=1337,
    duration_s=DEFAULT_DURATION_S,
    filter_rate_hz=DEFAULT_FILTER_RATE_HZ,
    camera_rate_hz=DEFAULT_CAMERA_RATE_HZ,
    radar_rate_hz=DEFAULT_RADAR_RATE_HZ,
):
    """Fuse camera and radar and compare against single-sensor baselines."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    full_times = (
        np.arange(
            round(duration_s * filter_rate_hz) + 1
        )
        * delta_t_s
    )
    truth = simulate_radar_camera_truth(full_times)

    sensor_data = _make_sensor_events(
        truth,
        full_times,
        rng,
        filter_rate_hz,
        camera_rate_hz,
        radar_rate_hz,
    )

    fused = _run_events(
        truth,
        sensor_data["records"],
        filter_rate_hz,
        sensor_kind=None,
    )
    camera_only = _run_events(
        truth,
        sensor_data["records"],
        filter_rate_hz,
        sensor_kind="camera",
    )
    radar_only = _run_events(
        truth,
        sensor_data["records"],
        filter_rate_hz,
        sensor_kind="radar",
    )

    fused_truth = _align_truth(
        full_times,
        truth,
        fused["times"],
    )
    camera_truth = _align_truth(
        full_times,
        truth,
        camera_only["times"],
    )
    radar_truth = _align_truth(
        full_times,
        truth,
        radar_only["times"],
    )

    def position_rmse(states, reference):
        return float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            states[:, :2]
                            - reference[:, :2]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )

    def velocity_rmse(states, reference):
        return float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            states[:, 2:4]
                            - reference[:, 2:4]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )

    fused_position_std_m = np.sqrt(
        np.maximum(
            np.trace(
                fused["filtered_covariances"][:, :2, :2],
                axis1=1,
                axis2=2,
            ),
            0.0,
        )
    )

    return {
        "times": fused["times"],
        "truth": fused_truth,
        "filtered": fused["filtered"],
        "smoothed": fused["smoothed"],
        "filtered_covariances": fused[
            "filtered_covariances"
        ],
        "smoothed_covariances": fused[
            "smoothed_covariances"
        ],
        "fused_position_std_m": fused_position_std_m,
        "camera_only_times": camera_only["times"],
        "camera_only_truth": camera_truth,
        "camera_only_filtered": camera_only["filtered"],
        "radar_only_times": radar_only["times"],
        "radar_only_truth": radar_truth,
        "radar_only_filtered": radar_only["filtered"],
        "camera_times": sensor_data["camera_times"],
        "camera_measurements": sensor_data[
            "camera_measurements"
        ],
        "radar_times": sensor_data["radar_times"],
        "radar_measurements": sensor_data[
            "radar_measurements"
        ],
        "camera_position_m": CAMERA_POSITION_M.copy(),
        "radar_position_m": RADAR_POSITION_M.copy(),
        "camera_dropout_s": CAMERA_DROPOUT_S,
        "radar_dropout_s": RADAR_DROPOUT_S,
        "filtered_position_rmse_m": position_rmse(
            fused["filtered"],
            fused_truth,
        ),
        "smoothed_position_rmse_m": position_rmse(
            fused["smoothed"],
            fused_truth,
        ),
        "filtered_velocity_rmse_m_s": velocity_rmse(
            fused["filtered"],
            fused_truth,
        ),
        "smoothed_velocity_rmse_m_s": velocity_rmse(
            fused["smoothed"],
            fused_truth,
        ),
        "camera_only_position_rmse_m": position_rmse(
            camera_only["filtered"],
            camera_truth,
        ),
        "radar_only_position_rmse_m": position_rmse(
            radar_only["filtered"],
            radar_truth,
        ),
    }


def plot_radar_camera_fusion(result, output_path=None):
    """Plot complementary camera/radar geometry and tracking results."""
    import matplotlib.pyplot as plt

    times = result["times"]
    truth = result["truth"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        truth[:, 0],
        truth[:, 1],
        linestyle="--",
        label="true target",
    )
    axes[0, 0].plot(
        result["filtered"][:, 0],
        result["filtered"][:, 1],
        label="camera + radar",
    )
    axes[0, 0].scatter(
        result["camera_position_m"][0],
        result["camera_position_m"][1],
        marker="^",
        s=70,
        label="camera",
    )
    axes[0, 0].scatter(
        result["radar_position_m"][0],
        result["radar_position_m"][1],
        marker="s",
        s=60,
        label="radar",
    )
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("Complementary 2-D target tracking")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        times,
        np.linalg.norm(
            result["filtered"][:, :2]
            - truth[:, :2],
            axis=1,
        ),
        label="fused position error",
    )
    axes[0, 1].plot(
        times,
        result["fused_position_std_m"],
        linestyle="--",
        label="posterior position sigma",
    )
    axes[0, 1].axvspan(
        *result["camera_dropout_s"],
        alpha=0.12,
        label="camera dropout",
    )
    axes[0, 1].axvspan(
        *result["radar_dropout_s"],
        alpha=0.12,
        label="radar dropout",
    )
    axes[0, 1].set_ylabel("position error / sigma [m]")
    axes[0, 1].set_title(
        "Uncertainty grows when complementary geometry disappears"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        result["camera_only_times"],
        np.linalg.norm(
            result["camera_only_filtered"][:, :2]
            - result["camera_only_truth"][:, :2],
            axis=1,
        ),
        label="camera only",
    )
    axes[1, 0].plot(
        result["radar_only_times"],
        np.linalg.norm(
            result["radar_only_filtered"][:, :2]
            - result["radar_only_truth"][:, :2],
            axis=1,
        ),
        label="radar only",
    )
    axes[1, 0].plot(
        times,
        np.linalg.norm(
            result["filtered"][:, :2]
            - truth[:, :2],
            axis=1,
        ),
        label="fused",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("position error [m]")
    axes[1, 0].set_title("Fusion vs single-sensor baselines")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        truth[:, 2],
        linestyle="--",
        label="true vx",
    )
    axes[1, 1].plot(
        times,
        truth[:, 3],
        linestyle="--",
        label="true vy",
    )
    axes[1, 1].plot(
        times,
        result["filtered"][:, 2],
        label="filtered vx",
    )
    axes[1, 1].plot(
        times,
        result["filtered"][:, 3],
        label="filtered vy",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("velocity [m/s]")
    axes[1, 1].set_title(
        "Radar range-rate helps constrain target velocity"
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
    print("Radar + camera target tracking")
    print(
        f"  fused position RMSE: "
        f"{result['filtered_position_rmse_m']:.3f} m filtered -> "
        f"{result['smoothed_position_rmse_m']:.3f} m RTS"
    )
    print(
        f"  fused velocity RMSE: "
        f"{result['filtered_velocity_rmse_m_s']:.3f} m/s filtered -> "
        f"{result['smoothed_velocity_rmse_m_s']:.3f} m/s RTS"
    )
    print(
        f"  camera-only position RMSE: "
        f"{result['camera_only_position_rmse_m']:.2f} m"
    )
    print(
        f"  radar-only position RMSE: "
        f"{result['radar_only_position_rmse_m']:.2f} m"
    )


if __name__ == "__main__":
    result = run_radar_camera_fusion()
    path = (
        Path(__file__).parent
        / "figures"
        / "radar-camera-fusion.png"
    )
    plot_radar_camera_fusion(result, path)
    _print_summary(result)
    print(f"saved {path}")

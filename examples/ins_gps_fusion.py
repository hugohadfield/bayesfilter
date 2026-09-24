"""Fuse a drifting black-box INS pose with sparse GPS position fixes.

The INS is assumed to be an existing navigation system that already outputs a
smooth 6-DoF body pose in a local dead-reckoning frame L:

    ^L T_B(t)

GPS reports noisy antenna positions in a global frame G. The filter estimates
the slowly wandering transform between the INS local frame and the global
frame. Roll and pitch are taken directly from the gravity-aligned INS attitude;
the correction state estimates global translation and yaw drift:

    x = [t_GL(3), psi_GL, v_drift(3), omega_drift].

The fused pose is

    ^G T_B(t) = ^G T_L(t) ^L T_B(t),

with ^G T_L restricted to translation plus yaw. GPS observations include a
known body-frame antenna lever arm.

The example includes a long GPS outage. During the outage the filter propagates
the learned drift model while uncertainty grows; when GPS returns the global
alignment is corrected. RTS smoothing then uses later GPS fixes to improve the
trajectory retrospectively through the outage.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS

from examples.rigid_body import rotation_distance, so3_exp, so3_log


DEFAULT_DURATION_S = 50.0
DEFAULT_INS_RATE_HZ = 20.0
DEFAULT_GPS_RATE_HZ = 2.0
GPS_OUTAGE_S = (18.0, 30.0)
GPS_POSITION_STD_M = 0.6
GPS_ANTENNA_LEVER_ARM_B_M = np.array([0.7, 0.15, 0.35])

TRANSLATION_SLICE = slice(0, 3)
YAW_INDEX = 3
DRIFT_VELOCITY_SLICE = slice(4, 7)
YAW_RATE_INDEX = 7


def yaw_rotation(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def yaw_rotation_derivative(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [-s, -c, 0.0],
            [c, -s, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )


def rpy_rotation(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_y @ rotation_x


def drift_transition(state, delta_t_s):
    next_state = np.asarray(state).copy()
    next_state[TRANSLATION_SLICE] += (
        delta_t_s * state[DRIFT_VELOCITY_SLICE]
    )
    next_state[YAW_INDEX] += delta_t_s * state[YAW_RATE_INDEX]
    return next_state


def drift_transition_jacobian(_state, delta_t_s):
    jacobian = np.eye(8)
    jacobian[TRANSLATION_SLICE, DRIFT_VELOCITY_SLICE] = (
        delta_t_s * np.eye(3)
    )
    jacobian[YAW_INDEX, YAW_RATE_INDEX] = delta_t_s
    return jacobian


def drift_process_covariance(delta_t_s):
    covariance = np.zeros((8, 8))

    translation_spectral_density = 0.05**2
    block = translation_spectral_density * np.array(
        [
            [delta_t_s**3 / 3.0, delta_t_s**2 / 2.0],
            [delta_t_s**2 / 2.0, delta_t_s],
        ]
    )
    for axis in range(3):
        covariance[axis, axis] = block[0, 0]
        covariance[axis, 4 + axis] = block[0, 1]
        covariance[4 + axis, axis] = block[1, 0]
        covariance[4 + axis, 4 + axis] = block[1, 1]

    yaw_spectral_density = np.deg2rad(0.12) ** 2
    yaw_block = yaw_spectral_density * np.array(
        [
            [delta_t_s**3 / 3.0, delta_t_s**2 / 2.0],
            [delta_t_s**2 / 2.0, delta_t_s],
        ]
    )
    covariance[YAW_INDEX, YAW_INDEX] = yaw_block[0, 0]
    covariance[YAW_INDEX, YAW_RATE_INDEX] = yaw_block[0, 1]
    covariance[YAW_RATE_INDEX, YAW_INDEX] = yaw_block[1, 0]
    covariance[YAW_RATE_INDEX, YAW_RATE_INDEX] = yaw_block[1, 1]

    return covariance


def simulate_ins_gps(seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / DEFAULT_INS_RATE_HZ
    times = np.arange(
        round(DEFAULT_DURATION_S * DEFAULT_INS_RATE_HZ) + 1,
        dtype=float,
    ) * dt

    global_positions = np.column_stack(
        (
            2.0 * times + 8.0 * np.sin(times / 7.0),
            18.0 * np.sin(times / 10.0),
            2.0 + 1.5 * np.sin(times / 8.0),
        )
    )

    global_velocity_x = 2.0 + (8.0 / 7.0) * np.cos(times / 7.0)
    global_velocity_y = 1.8 * np.cos(times / 10.0)
    global_yaw = np.arctan2(global_velocity_y, global_velocity_x)
    global_roll = np.deg2rad(4.0 * np.sin(times / 5.0))
    global_pitch = np.deg2rad(3.0 * np.sin(times / 6.0 + 0.3))

    true_alignment_yaw = np.deg2rad(
        8.0 + 0.25 * times + 2.0 * np.sin(times / 14.0)
    )
    true_alignment_translation = np.column_stack(
        (
            0.6 + 0.05 * times + 0.4 * np.sin(times / 9.0),
            -0.4 - 0.035 * times + 0.25 * np.sin(times / 11.0 + 0.5),
            0.2 + 0.012 * times + 0.15 * np.sin(times / 10.0),
        )
    )

    global_rotation_vectors = []
    local_positions = []
    local_rotation_vectors = []

    for index, _time in enumerate(times):
        global_rotation = rpy_rotation(
            global_roll[index],
            global_pitch[index],
            global_yaw[index],
        )
        alignment_rotation = yaw_rotation(true_alignment_yaw[index])

        local_position = (
            alignment_rotation.T
            @ (
                global_positions[index]
                - true_alignment_translation[index]
            )
        )
        local_rotation = alignment_rotation.T @ global_rotation

        global_rotation_vectors.append(so3_log(global_rotation))
        local_positions.append(local_position)
        local_rotation_vectors.append(so3_log(local_rotation))

    global_rotation_vectors = np.asarray(global_rotation_vectors)
    local_positions = np.asarray(local_positions)
    local_rotation_vectors = np.asarray(local_rotation_vectors)

    gps_step = round(DEFAULT_INS_RATE_HZ / DEFAULT_GPS_RATE_HZ)
    gps_indices = np.arange(0, len(times), gps_step)
    outage_start, outage_end = GPS_OUTAGE_S
    gps_indices = gps_indices[
        ~(
            (times[gps_indices] >= outage_start)
            & (times[gps_indices] <= outage_end)
        )
    ]

    gps_positions = []
    for index in gps_indices:
        global_rotation = so3_exp(global_rotation_vectors[index])
        antenna_position = (
            global_positions[index]
            + global_rotation @ GPS_ANTENNA_LEVER_ARM_B_M
        )
        gps_positions.append(
            antenna_position
            + rng.normal(0.0, GPS_POSITION_STD_M, size=3)
        )

    return {
        "times": times,
        "global_positions_m": global_positions,
        "global_rotation_vectors": global_rotation_vectors,
        "ins_local_positions_m": local_positions,
        "ins_local_rotation_vectors": local_rotation_vectors,
        "true_alignment_translation_m": true_alignment_translation,
        "true_alignment_yaw_rad": true_alignment_yaw,
        "gps_indices": gps_indices,
        "gps_times": times[gps_indices],
        "gps_positions_m": np.asarray(gps_positions),
        "delta_t_s": dt,
    }


def _make_gps_observation(
    measured_position,
    local_position,
    local_rotation_vector,
):
    local_rotation = so3_exp(local_rotation_vector)
    local_antenna_position = (
        local_position + local_rotation @ GPS_ANTENNA_LEVER_ARM_B_M
    )

    def observation_func(state):
        return (
            state[TRANSLATION_SLICE]
            + yaw_rotation(state[YAW_INDEX]) @ local_antenna_position
        )

    def observation_jacobian(state):
        jacobian = np.zeros((3, 8))
        jacobian[:, TRANSLATION_SLICE] = np.eye(3)
        jacobian[:, YAW_INDEX] = (
            yaw_rotation_derivative(state[YAW_INDEX])
            @ local_antenna_position
        )
        return jacobian

    return Observation(
        np.asarray(measured_position),
        GPS_POSITION_STD_M**2 * np.eye(3),
        observation_func,
        observation_jacobian,
    )


def _compose_fused_pose(state, local_position, local_rotation_vector):
    alignment_rotation = yaw_rotation(state[YAW_INDEX])
    global_position = (
        state[TRANSLATION_SLICE]
        + alignment_rotation @ local_position
    )
    global_rotation = (
        alignment_rotation @ so3_exp(local_rotation_vector)
    )
    return global_position, so3_log(global_rotation)


def run_ins_gps_fusion(seed=42):
    data = simulate_ins_gps(seed=seed)
    dt = data["delta_t_s"]

    first_index = data["gps_indices"][0]
    initial_yaw = 0.0
    first_local_rotation = so3_exp(
        data["ins_local_rotation_vectors"][first_index]
    )
    first_local_antenna = (
        data["ins_local_positions_m"][first_index]
        + first_local_rotation @ GPS_ANTENNA_LEVER_ARM_B_M
    )
    initial_translation = (
        data["gps_positions_m"][0]
        - yaw_rotation(initial_yaw) @ first_local_antenna
    )
    initial_mean = np.concatenate(
        (
            initial_translation,
            [initial_yaw],
            np.zeros(3),
            [0.0],
        )
    )
    initial_covariance = np.diag(
        [
            *([2.0**2] * 3),
            np.deg2rad(30.0) ** 2,
            *([0.30**2] * 3),
            np.deg2rad(1.0) ** 2,
        ]
    )

    model = StateTransitionModel(
        drift_transition,
        drift_process_covariance(dt),
        drift_transition_jacobian,
    )

    observations = [
        _make_gps_observation(
            measured,
            data["ins_local_positions_m"][index],
            data["ins_local_rotation_vectors"][index],
        )
        for measured, index in zip(
            data["gps_positions_m"],
            data["gps_indices"],
        )
    ]

    bayes_filter = BayesianFilter(
        model,
        Gaussian(initial_mean, initial_covariance),
    )
    filter_states, filter_times = bayes_filter.run(
        observations,
        data["gps_times"],
        rate_hz=DEFAULT_INS_RATE_HZ,
        use_jacobian=True,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=True,
    )

    filtered_states = np.array(
        [state.mean() for state in filter_states]
    )
    smoothed_states = np.array(
        [state.mean() for state in smoother_states]
    )

    # run() starts at the first GPS time and returns one state per INS-rate
    # grid point over the GPS-supported interval. The first and last GPS fixes
    # are deliberately at the ends of the simulated trajectory.
    local_positions = data["ins_local_positions_m"][: len(filter_times)]
    local_rotations = data["ins_local_rotation_vectors"][: len(filter_times)]
    truth_positions = data["global_positions_m"][: len(filter_times)]
    truth_rotations = data["global_rotation_vectors"][: len(filter_times)]

    filtered_positions = []
    filtered_rotations = []
    smoothed_positions = []
    smoothed_rotations = []

    for filtered_state, smoothed_state, local_position, local_rotation in zip(
        filtered_states,
        smoothed_states,
        local_positions,
        local_rotations,
    ):
        position, rotation = _compose_fused_pose(
            filtered_state,
            local_position,
            local_rotation,
        )
        filtered_positions.append(position)
        filtered_rotations.append(rotation)

        position, rotation = _compose_fused_pose(
            smoothed_state,
            local_position,
            local_rotation,
        )
        smoothed_positions.append(position)
        smoothed_rotations.append(rotation)

    filtered_positions = np.asarray(filtered_positions)
    filtered_rotations = np.asarray(filtered_rotations)
    smoothed_positions = np.asarray(smoothed_positions)
    smoothed_rotations = np.asarray(smoothed_rotations)

    fixed_alignment_state = filtered_states[0].copy()
    dead_reckoning_positions = []
    dead_reckoning_rotations = []
    for local_position, local_rotation in zip(
        local_positions,
        local_rotations,
    ):
        position, rotation = _compose_fused_pose(
            fixed_alignment_state,
            local_position,
            local_rotation,
        )
        dead_reckoning_positions.append(position)
        dead_reckoning_rotations.append(rotation)

    dead_reckoning_positions = np.asarray(dead_reckoning_positions)
    dead_reckoning_rotations = np.asarray(dead_reckoning_rotations)

    def position_rmse(positions, mask=None):
        errors = positions - truth_positions
        if mask is not None:
            errors = errors[mask]
        return float(
            np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        )

    filtered_rotation_error = np.array(
        [
            rotation_distance(expected, actual)
            for expected, actual in zip(
                truth_rotations,
                filtered_rotations,
            )
        ]
    )
    smoothed_rotation_error = np.array(
        [
            rotation_distance(expected, actual)
            for expected, actual in zip(
                truth_rotations,
                smoothed_rotations,
            )
        ]
    )

    filter_times = np.asarray(filter_times)
    outage_mask = (
        (filter_times >= GPS_OUTAGE_S[0])
        & (filter_times <= GPS_OUTAGE_S[1])
    )

    return {
        **data,
        "filter_times": filter_times,
        "filtered_states": filtered_states,
        "smoothed_states": smoothed_states,
        "filtered_positions_m": filtered_positions,
        "smoothed_positions_m": smoothed_positions,
        "dead_reckoning_positions_m": dead_reckoning_positions,
        "filtered_rotation_error_rad": filtered_rotation_error,
        "smoothed_rotation_error_rad": smoothed_rotation_error,
        "dead_reckoning_position_rmse_m":
            position_rmse(dead_reckoning_positions),
        "filtered_position_rmse_m":
            position_rmse(filtered_positions),
        "smoothed_position_rmse_m":
            position_rmse(smoothed_positions),
        "filtered_outage_position_rmse_m":
            position_rmse(filtered_positions, outage_mask),
        "smoothed_outage_position_rmse_m":
            position_rmse(smoothed_positions, outage_mask),
        "outage_mask": outage_mask,
        "initial_mean": initial_mean,
        "initial_covariance": initial_covariance,
    }


def plot_ins_gps_fusion(result, output_path=None):
    import matplotlib.pyplot as plt

    times = result["filter_times"]
    truth = result["global_positions_m"][: len(times)]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        truth[:, 0],
        truth[:, 1],
        label="true global trajectory",
    )
    axes[0, 0].plot(
        result["dead_reckoning_positions_m"][:, 0],
        result["dead_reckoning_positions_m"][:, 1],
        label="INS with fixed initial alignment",
    )
    axes[0, 0].plot(
        result["filtered_positions_m"][:, 0],
        result["filtered_positions_m"][:, 1],
        label="INS + GPS filter",
    )
    axes[0, 0].scatter(
        result["gps_positions_m"][:, 0],
        result["gps_positions_m"][:, 1],
        s=12,
        alpha=0.5,
        label="GPS",
    )
    axes[0, 0].set_xlabel("global x [m]")
    axes[0, 0].set_ylabel("global y [m]")
    axes[0, 0].set_title("Smooth local INS, noisy global GPS")
    axes[0, 0].legend(fontsize=8)

    dead_error = np.linalg.norm(
        result["dead_reckoning_positions_m"] - truth,
        axis=1,
    )
    filtered_error = np.linalg.norm(
        result["filtered_positions_m"] - truth,
        axis=1,
    )
    smoothed_error = np.linalg.norm(
        result["smoothed_positions_m"] - truth,
        axis=1,
    )
    axes[0, 1].plot(times, dead_error, label="fixed INS alignment")
    axes[0, 1].plot(times, filtered_error, label="filtered")
    axes[0, 1].plot(times, smoothed_error, label="RTS smoothed")
    axes[0, 1].set_ylabel("position error [m]")
    axes[0, 1].set_title("GPS continually estimates INS frame wander")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        times,
        np.rad2deg(
            result["filtered_states"][:, YAW_INDEX]
        ),
        label="filtered alignment yaw",
    )
    axes[1, 0].plot(
        times,
        np.rad2deg(
            result["smoothed_states"][:, YAW_INDEX]
        ),
        label="RTS alignment yaw",
    )
    axes[1, 0].plot(
        times,
        np.rad2deg(
            result["true_alignment_yaw_rad"][: len(times)]
        ),
        label="true alignment yaw",
    )
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("yaw [deg]")
    axes[1, 0].set_title("Global heading alignment drifts slowly")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        times,
        np.rad2deg(result["filtered_rotation_error_rad"]),
        label="filtered pose orientation error",
    )
    axes[1, 1].plot(
        times,
        np.rad2deg(result["smoothed_rotation_error_rad"]),
        label="RTS pose orientation error",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("orientation error [deg]")
    axes[1, 1].set_title("Full fused 3-D pose")

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axvspan(
            GPS_OUTAGE_S[0],
            GPS_OUTAGE_S[1],
            alpha=0.10,
            label="GPS outage" if axis is axes[1, 1] else None,
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
    print("Black-box INS + GPS fusion")
    print(
        f"  fixed INS alignment position RMSE: "
        f"{result['dead_reckoning_position_rmse_m']:.2f} m"
    )
    print(
        f"  filtered position RMSE: "
        f"{result['filtered_position_rmse_m']:.2f} m"
    )
    print(
        f"  RTS-smoothed position RMSE: "
        f"{result['smoothed_position_rmse_m']:.2f} m"
    )
    print(
        f"  outage filtered RMSE: "
        f"{result['filtered_outage_position_rmse_m']:.2f} m"
    )
    print(
        f"  outage RTS RMSE: "
        f"{result['smoothed_outage_position_rmse_m']:.2f} m"
    )


if __name__ == "__main__":
    result = run_ins_gps_fusion()
    path = (
        Path(__file__).parent
        / "figures"
        / "ins-gps-fusion.png"
    )
    plot_ins_gps_fusion(result, path)
    _print_summary(result)
    print(f"saved {path}")

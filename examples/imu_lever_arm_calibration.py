"""Infer an accelerometer's 3D mounting position from rotational acceleration.

A rigid body rotates about a stationary reference point. An accelerometer mounted
at an unknown body-frame lever arm ``r`` experiences

``a = alpha x r + omega x (omega x r)``.

The example uses only gyroscope and gravity-compensated accelerometer data. A
short trailing quadratic fit to gyro samples provides angular velocity and
angular acceleration at each accelerometer timestamp. The Bayesian filter then
estimates the static accelerometer lever arm and a constant accelerometer bias.

The hidden calibration state is

``[r_x, r_y, r_z, b_ax, b_ay, b_az]``.

For the first six seconds the body rotates only about its z axis. In that phase
``r_z`` is genuinely unobservable: moving the accelerometer along the rotation
axis does not change its rotational acceleration. Rich three-axis tumbling is
introduced afterwards, making the full lever arm observable. RTS smoothing then
uses the later excitation to improve the inferred mounting position throughout
the earlier ambiguous segment.

This example assumes gravity and acceleration of the reference point have
already been removed so it can focus on the rigid-body lever-arm effect itself.
"""

from pathlib import Path

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


LEVER_X = 0
LEVER_Y = 1
LEVER_Z = 2
BIAS_X = 3
BIAS_Y = 4
BIAS_Z = 5
STATE_SIZE = 6

DEFAULT_DURATION_S = 20.0
DEFAULT_GYRO_RATE_HZ = 200.0
DEFAULT_ACCEL_RATE_HZ = 50.0
GYRO_DERIVATIVE_WINDOW_S = 0.20
RICH_EXCITATION_START_S = 6.0
RICH_EXCITATION_FULL_S = 10.0

TRUE_LEVER_ARM_M = np.array([0.18, -0.11, 0.27])
TRUE_ACCEL_BIAS_M_S2 = np.array([0.018, -0.012, 0.009])
GYRO_STD_RAD_S = 2e-5
ACCEL_STD_M_S2 = 0.025
FILTER_ACCEL_STD_M_S2 = 0.030


def skew(vector):
    """Return the cross-product matrix such that ``skew(a) @ b == a x b``."""
    x_value, y_value, z_value = np.asarray(vector)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ]
    )


def rotational_acceleration(omega, alpha, lever_arm):
    """Acceleration at a rigidly mounted sensor relative to the body origin."""
    omega = np.asarray(omega)
    alpha = np.asarray(alpha)
    lever_arm = np.asarray(lever_arm)
    return np.cross(alpha, lever_arm) + np.cross(
        omega,
        np.cross(omega, lever_arm),
    )


def acceleration_design_matrix(omega, alpha):
    """Linear map from lever arm to rotational acceleration."""
    omega_skew = skew(omega)
    return skew(alpha) + omega_skew @ omega_skew


def _smoothstep01(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smoothstep01_derivative(value):
    value = np.asarray(value)
    return np.where(
        (value > 0.0) & (value < 1.0),
        6.0 * value * (1.0 - value),
        0.0,
    )


def _rich_excitation_gate(time_s):
    normalized = (
        np.asarray(time_s) - RICH_EXCITATION_START_S
    ) / (RICH_EXCITATION_FULL_S - RICH_EXCITATION_START_S)
    return _smoothstep01(normalized)


def _rich_excitation_gate_derivative(time_s):
    duration_s = RICH_EXCITATION_FULL_S - RICH_EXCITATION_START_S
    normalized = (np.asarray(time_s) - RICH_EXCITATION_START_S) / duration_s
    return _smoothstep01_derivative(normalized) / duration_s


def angular_velocity_profile(time_s):
    """Single-axis spin followed by rich 3D rotational excitation."""
    time_s = np.asarray(time_s, dtype=float)
    gate = _rich_excitation_gate(time_s)

    x_base = (
        0.90 * np.sin(0.95 * time_s)
        + 0.32 * np.sin(2.00 * time_s + 0.20)
    )
    y_base = (
        0.72 * np.cos(0.75 * time_s + 0.30)
        + 0.30 * np.sin(1.60 * time_s)
    )
    z_value = (
        1.05
        + 0.42 * np.sin(0.58 * time_s)
        + gate * 0.22 * np.cos(1.30 * time_s + 0.40)
    )
    return np.stack([gate * x_base, gate * y_base, z_value], axis=-1)


def angular_acceleration_profile(time_s):
    """Analytic derivative of :func:`angular_velocity_profile`."""
    time_s = np.asarray(time_s, dtype=float)
    gate = _rich_excitation_gate(time_s)
    gate_derivative = _rich_excitation_gate_derivative(time_s)

    x_base = (
        0.90 * np.sin(0.95 * time_s)
        + 0.32 * np.sin(2.00 * time_s + 0.20)
    )
    x_base_derivative = (
        0.90 * 0.95 * np.cos(0.95 * time_s)
        + 0.32 * 2.00 * np.cos(2.00 * time_s + 0.20)
    )
    y_base = (
        0.72 * np.cos(0.75 * time_s + 0.30)
        + 0.30 * np.sin(1.60 * time_s)
    )
    y_base_derivative = (
        -0.72 * 0.75 * np.sin(0.75 * time_s + 0.30)
        + 0.30 * 1.60 * np.cos(1.60 * time_s)
    )

    x_value = gate_derivative * x_base + gate * x_base_derivative
    y_value = gate_derivative * y_base + gate * y_base_derivative
    z_value = (
        0.42 * 0.58 * np.cos(0.58 * time_s)
        + gate_derivative * 0.22 * np.cos(1.30 * time_s + 0.40)
        - gate * 0.22 * 1.30 * np.sin(1.30 * time_s + 0.40)
    )
    return np.stack([x_value, y_value, z_value], axis=-1)


def estimate_angular_kinematics(
    gyro_times,
    gyro_measurements,
    query_times,
    window_s=GYRO_DERIVATIVE_WINDOW_S,
):
    """Estimate omega and alpha from a causal quadratic fit to recent gyro data.

    Each axis is fit independently with

    ``omega(t + dt) ~= c0 + c1 dt + c2 dt^2``

    over a trailing time window. Evaluating at ``dt = 0`` gives ``omega = c0``
    and ``alpha = c1``. This avoids directly differentiating noisy adjacent
    gyro samples while keeping the estimate causal.
    """
    gyro_times = np.asarray(gyro_times, dtype=float)
    gyro_measurements = np.asarray(gyro_measurements, dtype=float)
    query_times = np.asarray(query_times, dtype=float)

    estimated_omega = []
    estimated_alpha = []
    for query_time in query_times:
        mask = (
            (gyro_times <= query_time + 1e-12)
            & (gyro_times >= query_time - window_s - 1e-12)
        )
        local_times = gyro_times[mask] - query_time
        if local_times.size < 3:
            raise ValueError("not enough gyro samples for quadratic fit")

        design = np.column_stack(
            [
                np.ones_like(local_times),
                local_times,
                local_times**2,
            ]
        )
        coefficients = np.linalg.lstsq(
            design,
            gyro_measurements[mask],
            rcond=None,
        )[0]
        estimated_omega.append(coefficients[0])
        estimated_alpha.append(coefficients[1])

    return np.asarray(estimated_omega), np.asarray(estimated_alpha)


def propagate_calibration_state(state, _delta_t_s):
    """Static calibration parameters with tiny process noise."""
    return np.asarray(state).copy()


def calibration_transition_jacobian(_state, _delta_t_s):
    return np.eye(STATE_SIZE)


def make_accelerometer_model(omega, alpha):
    """Return an accelerometer observation function and its exact Jacobian."""
    lever_matrix = acceleration_design_matrix(omega, alpha)
    observation_matrix = np.column_stack([lever_matrix, np.eye(3)])

    def observe(state):
        return observation_matrix @ np.asarray(state)

    def jacobian(_state):
        return observation_matrix

    return observe, jacobian, observation_matrix


def simulate_imu_measurements(
    seed=4,
    duration_s=DEFAULT_DURATION_S,
    gyro_rate_hz=DEFAULT_GYRO_RATE_HZ,
    accel_rate_hz=DEFAULT_ACCEL_RATE_HZ,
):
    """Generate noisy gyro and accelerometer data for the calibration problem."""
    rng = np.random.default_rng(seed)
    gyro_times = np.arange(
        0.0,
        duration_s + 0.5 / gyro_rate_hz,
        1.0 / gyro_rate_hz,
    )
    gyro_truth = angular_velocity_profile(gyro_times)
    gyro_measurements = gyro_truth + rng.normal(
        0.0,
        GYRO_STD_RAD_S,
        gyro_truth.shape,
    )

    accel_times = np.arange(
        GYRO_DERIVATIVE_WINDOW_S,
        duration_s + 0.5 / accel_rate_hz,
        1.0 / accel_rate_hz,
    )
    omega_truth = angular_velocity_profile(accel_times)
    alpha_truth = angular_acceleration_profile(accel_times)
    acceleration_truth = np.asarray(
        [
            rotational_acceleration(omega, alpha, TRUE_LEVER_ARM_M)
            + TRUE_ACCEL_BIAS_M_S2
            for omega, alpha in zip(omega_truth, alpha_truth)
        ]
    )
    accel_measurements = acceleration_truth + rng.normal(
        0.0,
        ACCEL_STD_M_S2,
        acceleration_truth.shape,
    )

    return {
        "gyro_times": gyro_times,
        "gyro_truth": gyro_truth,
        "gyro_measurements": gyro_measurements,
        "accel_times": accel_times,
        "omega_truth": omega_truth,
        "alpha_truth": alpha_truth,
        "acceleration_truth": acceleration_truth,
        "accel_measurements": accel_measurements,
    }


def run_imu_lever_arm_calibration(
    seed=4,
    duration_s=DEFAULT_DURATION_S,
    gyro_rate_hz=DEFAULT_GYRO_RATE_HZ,
    accel_rate_hz=DEFAULT_ACCEL_RATE_HZ,
    use_jacobian=False,
):
    """Estimate accelerometer position and bias from gyro + accelerometer data."""
    data = simulate_imu_measurements(
        seed=seed,
        duration_s=duration_s,
        gyro_rate_hz=gyro_rate_hz,
        accel_rate_hz=accel_rate_hz,
    )
    estimated_omega, estimated_alpha = estimate_angular_kinematics(
        data["gyro_times"],
        data["gyro_measurements"],
        data["accel_times"],
    )

    observations = []
    observation_matrices = []
    for measurement, omega, alpha in zip(
        data["accel_measurements"],
        estimated_omega,
        estimated_alpha,
    ):
        observation_func, observation_jacobian, observation_matrix = (
            make_accelerometer_model(omega, alpha)
        )
        observations.append(
            Observation(
                measurement,
                FILTER_ACCEL_STD_M_S2**2 * np.eye(3),
                observation_func,
                observation_jacobian,
            )
        )
        observation_matrices.append(observation_matrix)

    transition_model = StateTransitionModel(
        propagate_calibration_state,
        np.diag(np.full(STATE_SIZE, 1e-10)),
        calibration_transition_jacobian,
    )
    initial_mean = np.array([-0.08, 0.06, -0.10, 0.0, 0.0, 0.0])
    initial_std = np.array([0.35, 0.35, 0.35, 0.08, 0.08, 0.08])
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(initial_mean, np.diag(initial_std**2)),
    )

    delta_t_s = 1.0 / accel_rate_hz
    filter_times = np.concatenate(
        [[data["accel_times"][0] - delta_t_s], data["accel_times"]]
    )
    filter_states = bayes_filter.run_synchronous(
        observations,
        filter_times,
        use_jacobian=use_jacobian,
    )
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

    filtered_lever_error = np.linalg.norm(
        filtered[:, :3] - TRUE_LEVER_ARM_M,
        axis=1,
    )
    smoothed_lever_error = np.linalg.norm(
        smoothed[:, :3] - TRUE_LEVER_ARM_M,
        axis=1,
    )
    lever_std = np.sqrt(
        np.stack(
            [
                filtered_covariances[:, LEVER_X, LEVER_X],
                filtered_covariances[:, LEVER_Y, LEVER_Y],
                filtered_covariances[:, LEVER_Z, LEVER_Z],
            ],
            axis=1,
        )
    )

    pre_excitation_index = int(
        np.argmin(np.abs(filter_times - (RICH_EXCITATION_START_S - 0.2)))
    )
    fully_excited_index = int(
        np.argmin(np.abs(filter_times - RICH_EXCITATION_FULL_S))
    )

    return {
        **data,
        "times": filter_times,
        "estimated_omega": estimated_omega,
        "estimated_alpha": estimated_alpha,
        "observation_matrices": np.asarray(observation_matrices),
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_lever_error_m": filtered_lever_error,
        "smoothed_lever_error_m": smoothed_lever_error,
        "lever_std_m": lever_std,
        "final_lever_arm_m": filtered[-1, :3].copy(),
        "final_lever_error_m": float(filtered_lever_error[-1]),
        "final_accel_bias_m_s2": filtered[-1, 3:].copy(),
        "final_accel_bias_error_m_s2": float(
            np.linalg.norm(filtered[-1, 3:] - TRUE_ACCEL_BIAS_M_S2)
        ),
        "filtered_lever_rmse_m": float(
            np.sqrt(np.mean(filtered_lever_error**2))
        ),
        "smoothed_lever_rmse_m": float(
            np.sqrt(np.mean(smoothed_lever_error**2))
        ),
        "pre_excitation_z_std_m": float(
            lever_std[pre_excitation_index, LEVER_Z]
        ),
        "full_excitation_z_std_m": float(
            lever_std[fully_excited_index, LEVER_Z]
        ),
        "pre_excitation_z_error_m": float(
            abs(filtered[pre_excitation_index, LEVER_Z] - TRUE_LEVER_ARM_M[2])
        ),
        "initial_smoothed_lever_error_m": float(smoothed_lever_error[0]),
        "gyro_rate_hz": gyro_rate_hz,
        "accel_rate_hz": accel_rate_hz,
        "use_jacobian": use_jacobian,
    }


def plot_imu_lever_arm_calibration(result, output_path=None):
    """Plot excitation, lever-arm inference, and observability contraction."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    filter_times = result["times"]
    accel_times = result["accel_times"]

    for axis_index, label in enumerate(("omega x", "omega y", "omega z")):
        axes[0, 0].plot(
            accel_times,
            result["estimated_omega"][:, axis_index],
            label=label,
        )
    axes[0, 0].axvline(
        RICH_EXCITATION_START_S,
        linestyle="--",
        label="3D excitation begins",
    )
    axes[0, 0].set_ylabel("angular velocity [rad/s]")
    axes[0, 0].set_title("Gyro excitation: single-axis spin then 3D tumble")
    axes[0, 0].legend()

    component_names = ("r_x", "r_y", "r_z")
    for index, component_name in enumerate(component_names):
        axes[0, 1].plot(
            filter_times,
            result["filtered"][:, index],
            alpha=0.65,
            label=f"filtered {component_name}",
        )
        axes[0, 1].plot(
            filter_times,
            result["smoothed"][:, index],
            label=f"RTS {component_name}",
        )
        axes[0, 1].axhline(
            TRUE_LEVER_ARM_M[index],
            linestyle=":",
            alpha=0.65,
        )
    axes[0, 1].axvline(RICH_EXCITATION_START_S, linestyle="--")
    axes[0, 1].set_ylabel("lever arm [m]")
    axes[0, 1].set_title(
        "Accelerometer position inferred from rotational acceleration"
    )
    axes[0, 1].legend(ncol=2, fontsize=8)

    axes[1, 0].plot(
        filter_times,
        result["filtered_lever_error_m"],
        label="filtered lever error",
    )
    axes[1, 0].plot(
        filter_times,
        result["smoothed_lever_error_m"],
        label="RTS lever error",
    )
    axes[1, 0].axvline(RICH_EXCITATION_START_S, linestyle="--")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("3D lever error [m]")
    axes[1, 0].set_title(
        "Later tumbling resolves the previously hidden dimension"
    )
    axes[1, 0].legend()

    for index, component_name in enumerate(component_names):
        axes[1, 1].plot(
            filter_times,
            result["lever_std_m"][:, index],
            label=f"sigma {component_name}",
        )
    axes[1, 1].axvline(
        RICH_EXCITATION_START_S,
        linestyle="--",
        label="3D excitation begins",
    )
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("posterior standard deviation [m]")
    axes[1, 1].set_title("r_z stays unobservable during pure z-axis rotation")
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
    print("IMU lever-arm calibration from rotational acceleration")
    print(
        f"  lever arm: {result['final_lever_arm_m']} m "
        f"(truth {TRUE_LEVER_ARM_M} m)"
    )
    print(
        f"  final lever error: {result['final_lever_error_m'] * 1000.0:.2f} mm"
    )
    print(
        f"  r_z uncertainty: {result['pre_excitation_z_std_m']:.3f} m before "
        f"3D excitation -> {result['full_excitation_z_std_m']:.3f} m once excited"
    )
    print(
        f"  lever RMSE: {result['filtered_lever_rmse_m']:.3f} m filtered -> "
        f"{result['smoothed_lever_rmse_m']:.3f} m RTS"
    )
    print(
        f"  accelerometer bias: {result['final_accel_bias_m_s2']} m/s^2 "
        f"(truth {TRUE_ACCEL_BIAS_M_S2} m/s^2)"
    )


if __name__ == "__main__":
    result = run_imu_lever_arm_calibration()
    path = Path(__file__).parent / "figures" / "imu-lever-arm-calibration.png"
    plot_imu_lever_arm_calibration(result, path)
    _print_summary(result)
    print(f"saved {path}")

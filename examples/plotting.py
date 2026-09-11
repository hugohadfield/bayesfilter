"""Matplotlib helpers for the runnable examples."""

from pathlib import Path

import numpy as np


COLORS = {
    "truth": "#222222",
    "measurement": "#999999",
    "filter": "#2474B5",
    "smoother": "#D55E00",
}


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )
    return plt


def plot_linear_tracking(result, output_path, show=False):
    plt = _pyplot()
    dimension = result["dimension"]
    derivative_order = result["derivative_order"]
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    measurements = result["measurements"]
    labels = ("x", "y", "z")[:dimension]
    model_name = (
        "Constant velocity" if derivative_order == 1 else "Constant acceleration"
    )

    figure, axes = plt.subplots(
        derivative_order + 1,
        1,
        figsize=(9.2, 2.7 * (derivative_order + 1)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for component, label in enumerate(labels):
        axes[0].scatter(
            times[1:],
            measurements[:, component],
            s=12,
            alpha=0.45,
            color=COLORS["measurement"],
            edgecolors="none",
            label="Measurements" if component == 0 else None,
        )
        axes[0].plot(
            times,
            truth[:, component],
            color=COLORS["truth"],
            linestyle="--",
            label=f"True {label}",
        )
        axes[0].plot(
            times,
            filtered[:, component],
            color=COLORS["filter"],
            alpha=0.8,
            label=f"Filtered {label}",
        )
        axes[0].plot(
            times,
            smoothed[:, component],
            color=COLORS["smoother"],
            label=f"Smoothed {label}",
        )
    axes[0].set_title(
        f"{model_name} tracking in {dimension}D — "
        f"position RMSE {result['filtered_rmse']:.2f} → {result['smoothed_rmse']:.2f}"
    )
    axes[0].set_ylabel("Position")
    axes[0].legend(ncol=min(4, 1 + 3 * dimension), fontsize=8)

    derivative_names = ("Velocity", "Acceleration")
    for derivative_index in range(1, derivative_order + 1):
        offset = derivative_index * dimension
        for component, label in enumerate(labels):
            axes[derivative_index].plot(
                times,
                truth[:, offset + component],
                color=COLORS["truth"],
                linestyle="--",
                label=f"True {label}",
            )
            axes[derivative_index].plot(
                times,
                filtered[:, offset + component],
                color=COLORS["filter"],
                alpha=0.8,
                label=f"Filtered {label}",
            )
            axes[derivative_index].plot(
                times,
                smoothed[:, offset + component],
                color=COLORS["smoother"],
                label=f"Smoothed {label}",
            )
        axes[derivative_index].set_ylabel(derivative_names[derivative_index - 1])
    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def run_and_plot_linear_example(dimension, derivative_order, filename):
    from examples.linear_tracking import run_linear_tracking

    result = run_linear_tracking(
        dimension,
        derivative_order,
        seed=100 * derivative_order + dimension,
    )
    output_path = Path(__file__).parent / "figures" / filename
    plot_linear_tracking(result, output_path)
    print(
        f"filtered RMSE={result['filtered_rmse']:.3f}, "
        f"smoothed RMSE={result['smoothed_rmse']:.3f}"
    )
    print(f"saved {output_path}")


def _add_covariance_ellipse(axis, mean, covariance, color):
    from matplotlib.patches import Ellipse

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    scale_95_percent = np.sqrt(5.991)
    width, height = 2.0 * scale_95_percent * np.sqrt(eigenvalues)
    axis.add_patch(
        Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=0.8,
        )
    )


def plot_planar_tracking(result, output_path, show=False):
    """Plot a curved two-dimensional trajectory and position uncertainty."""
    plt = _pyplot()
    truth = result["truth"]
    measurements = result["measurements"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    axis.scatter(
        measurements[:, 0],
        measurements[:, 1],
        s=17,
        color=COLORS["measurement"],
        alpha=0.55,
        edgecolors="none",
        label="Position measurements",
        zorder=2,
    )
    axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True trajectory",
        zorder=3,
    )
    axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        label=f"Filtered (RMSE {result['filtered_rmse']:.2f})",
        zorder=4,
    )
    axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {result['smoothed_rmse']:.2f})",
        zorder=5,
    )
    for index in range(5, len(smoothed), 10):
        _add_covariance_ellipse(
            axis,
            smoothed[index, :2],
            result["smoothed_covariances"][index, :2, :2],
            COLORS["smoother"],
        )
    axis.scatter(
        truth[0, 0],
        truth[0, 1],
        marker="o",
        s=45,
        facecolor="white",
        edgecolor=COLORS["truth"],
        linewidth=1.5,
        zorder=6,
        label="Start",
    )
    axis.scatter(
        truth[-1, 0],
        truth[-1, 1],
        marker="s",
        s=42,
        facecolor=COLORS["truth"],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label="End",
    )
    axis.set_title("Planar trajectory reconstruction")
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="upper right")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_heading_coupled_tracking(result, output_path, show=False):
    """Plot asynchronous position/gyro tracking of a nonholonomic vehicle."""
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    trajectory_axis, heading_axis, motion_axis, angular_rate_axis = axes.flat

    trajectory_axis.scatter(
        result["position_measurements"][:, 0],
        result["position_measurements"][:, 1],
        s=16,
        color=COLORS["measurement"],
        alpha=0.55,
        edgecolors="none",
        label="Global position measurements",
    )
    trajectory_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True trajectory",
    )
    trajectory_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered",
    )
    trajectory_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    heading_indices = np.linspace(0, len(times) - 1, 13, dtype=int)
    trajectory_axis.quiver(
        smoothed[heading_indices, 0],
        smoothed[heading_indices, 1],
        np.cos(smoothed[heading_indices, 2]),
        np.sin(smoothed[heading_indices, 2]),
        angles="xy",
        scale_units="xy",
        scale=2.2,
        width=0.005,
        color=COLORS["smoother"],
        alpha=0.7,
        label="Estimated heading",
    )
    trajectory_axis.set_title("Global trajectory and vehicle heading")
    trajectory_axis.set_xlabel("x position")
    trajectory_axis.set_ylabel("y position")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.legend(fontsize=8)

    heading_axis.plot(
        times,
        np.degrees(truth[:, 2]),
        color=COLORS["truth"],
        linestyle="--",
        label="True heading",
    )
    heading_axis.plot(
        times,
        np.degrees(filtered[:, 2]),
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered",
    )
    heading_axis.plot(
        times,
        np.degrees(smoothed[:, 2]),
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    heading_axis.axvspan(8.0, 10.0, color="#BBBBBB", alpha=0.18, label="Stopped")
    heading_axis.set_title("Heading freezes when the vehicle stops")
    heading_axis.set_xlabel("Time [s]")
    heading_axis.set_ylabel("Unwrapped heading [deg]")
    heading_axis.legend(fontsize=8)

    motion_axis.plot(
        times,
        truth[:, 3],
        color=COLORS["truth"],
        linestyle="--",
        label="True signed speed",
    )
    motion_axis.plot(
        times,
        smoothed[:, 3],
        color=COLORS["filter"],
        label="Smoothed signed speed",
    )
    motion_axis.axhline(0.0, color="#777777", linewidth=0.8)
    motion_axis.axvspan(8.0, 10.0, color="#BBBBBB", alpha=0.18)
    motion_axis.set_title("Forward, stopped, and reverse motion")
    motion_axis.set_xlabel("Time [s]")
    motion_axis.set_ylabel("Signed speed")
    curvature_axis = motion_axis.twinx()
    curvature_axis.plot(
        times,
        truth[:, 4],
        color="#009E73",
        linestyle="--",
        alpha=0.75,
        label="True curvature",
    )
    curvature_axis.plot(
        times,
        smoothed[:, 4],
        color="#009E73",
        alpha=0.45,
        label="Smoothed curvature",
    )
    curvature_axis.set_ylabel("Curvature", color="#007A59")
    motion_lines, motion_labels = motion_axis.get_legend_handles_labels()
    curvature_lines, curvature_labels = curvature_axis.get_legend_handles_labels()
    motion_axis.legend(
        motion_lines + curvature_lines,
        motion_labels + curvature_labels,
        fontsize=8,
        loc="lower left",
    )

    true_angular_rate = truth[:, 3] * truth[:, 4]
    filtered_angular_rate = filtered[:, 3] * filtered[:, 4]
    smoothed_angular_rate = smoothed[:, 3] * smoothed[:, 4]
    angular_rate_axis.scatter(
        result["angular_rate_times"],
        result["angular_rate_measurements"],
        s=7,
        color=COLORS["measurement"],
        alpha=0.30,
        edgecolors="none",
        label="Local gyro measurements",
    )
    angular_rate_axis.plot(
        times,
        true_angular_rate,
        color=COLORS["truth"],
        linestyle="--",
        label="True vκ",
    )
    angular_rate_axis.plot(
        times,
        filtered_angular_rate,
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered vκ",
    )
    angular_rate_axis.plot(
        times,
        smoothed_angular_rate,
        color=COLORS["smoother"],
        label="Smoothed vκ",
    )
    angular_rate_axis.axhline(0.0, color="#777777", linewidth=0.8)
    angular_rate_axis.axvspan(8.0, 10.0, color="#BBBBBB", alpha=0.18)
    angular_rate_axis.set_title("Body-frame angular rate: ω = vκ")
    angular_rate_axis.set_xlabel("Time [s]")
    angular_rate_axis.set_ylabel("Angular rate [rad/s]")
    angular_rate_axis.legend(fontsize=8, ncol=2)

    mode = "Jacobian" if result["use_jacobian"] else "Unscented"
    figure.suptitle(
        f"{mode} heading-coupled tracking from global position and local gyro",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_bearings_only_tracking(result, output_path, show=False):
    """Plot triangulation, dropout uncertainty, and RTS reconstruction."""
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    dropout_start, dropout_end = result["dropout_interval"]

    figure = plt.figure(figsize=(11.2, 7.2))
    axes = figure.subplot_mosaic(
        [["geometry", "error"], ["geometry", "uncertainty"]],
        width_ratios=(1.2, 1.0),
    )
    geometry_axis = axes["geometry"]
    error_axis = axes["error"]
    uncertainty_axis = axes["uncertainty"]

    sensor_colors = ("#0072B2", "#D55E00")
    for sensor_index, (sensor_position, color) in enumerate(
        zip(result["sensor_positions"], sensor_colors)
    ):
        measurement_times = result["sensor_times"][sensor_index]
        measurements = result["sensor_measurements"][sensor_index]
        selection = np.linspace(
            0,
            len(measurement_times) - 1,
            min(9, len(measurement_times)),
            dtype=int,
        )
        for ray_number, measurement_index in enumerate(selection):
            measurement_time = measurement_times[measurement_index]
            target_index = np.abs(times - measurement_time).argmin()
            target_range = np.linalg.norm(truth[target_index, :2] - sensor_position)
            endpoint = (
                sensor_position + 1.08 * target_range * measurements[measurement_index]
            )
            geometry_axis.plot(
                [sensor_position[0], endpoint[0]],
                [sensor_position[1], endpoint[1]],
                color=color,
                alpha=0.18,
                linewidth=0.9,
                label=(
                    f"Sensor {sensor_index + 1} bearing rays"
                    if ray_number == 0
                    else None
                ),
            )
        geometry_axis.scatter(
            sensor_position[0],
            sensor_position[1],
            marker="^",
            s=75,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
            label=f"Sensor {sensor_index + 1}",
        )

    geometry_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True target",
        zorder=4,
    )
    geometry_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.65,
        label="Filtered",
        zorder=5,
    )
    geometry_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed",
        zorder=6,
    )
    for ellipse_time in (4.5, 7.5, 10.5, 12.0, 14.5):
        index = np.abs(times - ellipse_time).argmin()
        _add_covariance_ellipse(
            geometry_axis,
            smoothed[index, :2],
            result["smoothed_covariances"][index, :2, :2],
            COLORS["smoother"],
        )
    geometry_axis.scatter(
        truth[0, 0],
        truth[0, 1],
        marker="o",
        s=42,
        facecolor="white",
        edgecolor=COLORS["truth"],
        linewidth=1.4,
        zorder=7,
    )
    geometry_axis.scatter(
        truth[-1, 0],
        truth[-1, 1],
        marker="s",
        s=42,
        color=COLORS["truth"],
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    geometry_axis.set_title("Bearing geometry and reconstructed trajectory")
    geometry_axis.set_xlabel("x position")
    geometry_axis.set_ylabel("y position")
    geometry_axis.set_aspect("equal", adjustable="datalim")
    geometry_axis.legend(fontsize=7, ncol=2, loc="upper center")

    filtered_error = np.linalg.norm(filtered[:, :2] - truth[:, :2], axis=1)
    smoothed_error = np.linalg.norm(smoothed[:, :2] - truth[:, :2], axis=1)
    error_axis.plot(
        times,
        filtered_error,
        color=COLORS["filter"],
        label=f"Filtered (RMSE {result['filtered_position_rmse']:.2f})",
    )
    error_axis.plot(
        times,
        smoothed_error,
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {result['smoothed_position_rmse']:.2f})",
    )
    error_axis.axvspan(
        dropout_start,
        dropout_end,
        color="#BBBBBB",
        alpha=0.20,
        label="Sensor 2 unavailable",
    )
    error_axis.set_title("Position error during single-sensor interval")
    error_axis.set_xlabel("Time [s]")
    error_axis.set_ylabel("Euclidean error")
    error_axis.legend(fontsize=8)

    filtered_major_std = np.array(
        [
            np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())
            for covariance in result["filtered_covariances"]
        ]
    )
    smoothed_major_std = np.array(
        [
            np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())
            for covariance in result["smoothed_covariances"]
        ]
    )
    uncertainty_axis.plot(
        times,
        filtered_major_std,
        color=COLORS["filter"],
        label="Filtered",
    )
    uncertainty_axis.plot(
        times,
        smoothed_major_std,
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    uncertainty_axis.axvspan(
        dropout_start,
        dropout_end,
        color="#BBBBBB",
        alpha=0.20,
    )
    uncertainty_axis.vlines(
        result["sensor_times"][1],
        0.0,
        0.06,
        transform=uncertainty_axis.get_xaxis_transform(),
        color=sensor_colors[1],
        alpha=0.55,
        linewidth=1.0,
        label="Sensor 2 bearings",
    )
    uncertainty_axis.set_title("Major-axis position uncertainty")
    uncertainty_axis.set_xlabel("Time [s]")
    uncertainty_axis.set_ylabel("Posterior σ")
    uncertainty_axis.legend(fontsize=8)

    mode = "Jacobian" if result["use_jacobian"] else "Unscented"
    figure.suptitle(
        f"{mode} bearings-only tracking with interrupted triangulation",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_rigid_body_tracking(result, output_path, show=False):
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    measurements = result["measurements"].reshape(-1, 4, 3)
    labels = ("x", "y", "z")

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    position_axis, rotation_axis, omega_axis, invariant_axis = axes.flat
    position_axis.scatter(
        measurements[:, 0, 0],
        measurements[:, 0, 1],
        s=11,
        alpha=0.40,
        color=COLORS["measurement"],
        label="Observed body origin",
    )
    position_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True center",
    )
    position_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        label="Filtered center",
    )
    position_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="Smoothed center",
    )
    position_axis.set_title("Center trajectory, x-y projection")
    position_axis.set_xlabel("x position")
    position_axis.set_ylabel("y position")
    position_axis.legend(fontsize=8)

    rotation_axis.plot(
        times,
        np.degrees(result["filtered_rotation_error"]),
        color=COLORS["filter"],
        label="Filtered",
    )
    rotation_axis.plot(
        times,
        np.degrees(result["smoothed_rotation_error"]),
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    rotation_axis.set_title("Geodesic attitude error")
    rotation_axis.set_xlabel("Time [s]")
    rotation_axis.set_ylabel("Angle [deg]")
    rotation_axis.legend()

    for index, label in enumerate(labels):
        omega_axis.plot(
            times,
            truth[:, 9 + index],
            color=COLORS["truth"],
            linestyle="--",
            alpha=0.7,
            label=f"True ω{label}",
        )
        omega_axis.plot(
            times,
            smoothed[:, 9 + index],
            label=f"Smoothed ω{label}",
        )
    omega_axis.set_title("Body angular velocity")
    omega_axis.set_xlabel("Time [s]")
    omega_axis.set_ylabel("Angular velocity [rad/s]")
    omega_axis.legend(ncol=2, fontsize=8)

    relative_energy = result["truth_energy"] / result["truth_energy"][0] - 1.0
    relative_momentum = result["truth_momentum"] / result["truth_momentum"][0] - 1.0
    invariant_axis.plot(times, relative_energy, label="Kinetic energy")
    invariant_axis.plot(times, relative_momentum, label="|angular momentum|")
    invariant_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    invariant_axis.set_title("Torque-free integration invariants")
    invariant_axis.set_xlabel("Time [s]")
    invariant_axis.set_ylabel("Relative drift")
    invariant_axis.legend()

    figure.suptitle("Rigid-body tracking with known inertia", fontsize=14)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_inertia_estimation(result, output_path, show=False):
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    smoothed = result["smoothed"]
    ratio_truth = result["true_inertia_ratios"]
    ratio_filtered = result["filtered_inertia_ratios"]
    ratio_smoothed = result["smoothed_inertia_ratios"]
    parameter_std = np.sqrt(
        np.stack(
            [
                np.diag(covariance)[12:14]
                for covariance in result["filtered_covariances"]
            ]
        )
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ratio_axis, rotation_axis, omega_axis, contraction_axis = axes.flat
    for index, label in enumerate(("I₂ / I₁", "I₃ / I₁")):
        color = (COLORS["filter"], COLORS["smoother"])[index]
        ratio_axis.axhline(
            ratio_truth[index],
            color=color,
            linestyle="--",
            alpha=0.8,
        )
        ratio_axis.plot(times, ratio_filtered[:, index], color=color, label=label)
        ratio_std = ratio_filtered[:, index] * parameter_std[:, index]
        ratio_axis.fill_between(
            times,
            ratio_filtered[:, index] - 2.0 * ratio_std,
            ratio_filtered[:, index] + 2.0 * ratio_std,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ratio_axis.plot(
            times,
            ratio_smoothed[:, index],
            color=color,
            linewidth=1.0,
            alpha=0.45,
        )
    ratio_axis.set_title("Inferred principal-inertia ratios")
    ratio_axis.set_xlabel("Time [s]")
    ratio_axis.set_ylabel("Ratio")
    ratio_axis.legend()

    rotation_axis.plot(
        times,
        np.degrees(result["filtered_rotation_error"]),
        color=COLORS["filter"],
        label="Filtered",
    )
    rotation_axis.plot(
        times,
        np.degrees(result["smoothed_rotation_error"]),
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    rotation_axis.set_title("Geodesic attitude error")
    rotation_axis.set_xlabel("Time [s]")
    rotation_axis.set_ylabel("Angle [deg]")
    rotation_axis.legend()

    for index, label in enumerate(("x", "y", "z")):
        omega_axis.plot(
            times,
            truth[:, 9 + index],
            linestyle="--",
            alpha=0.65,
            label=f"True ω{label}",
        )
        omega_axis.plot(
            times,
            smoothed[:, 9 + index],
            label=f"Smoothed ω{label}",
        )
    omega_axis.set_title("Body angular velocity")
    omega_axis.set_xlabel("Time [s]")
    omega_axis.set_ylabel("Angular velocity [rad/s]")
    omega_axis.legend(ncol=2, fontsize=8)

    contraction_axis.plot(
        times,
        parameter_std[:, 0] / parameter_std[0, 0],
        color=COLORS["filter"],
        label="log(I₂ / I₁)",
    )
    contraction_axis.plot(
        times,
        parameter_std[:, 1] / parameter_std[0, 1],
        color=COLORS["smoother"],
        label="log(I₃ / I₁)",
    )
    contraction_axis.set_title("Parameter uncertainty contraction")
    contraction_axis.set_xlabel("Time [s]")
    contraction_axis.set_ylabel("Posterior σ / initial σ")
    contraction_axis.legend()

    figure.suptitle(
        "Rigid-body inertia estimation from world-space points", fontsize=14
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_readme_inertia_summary(result, output_path, show=False):
    """Plot a compact rigid-body inference summary for the main README."""
    from examples.rigid_body import transform_canonical_points

    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    smoothed = result["smoothed"]
    measurements = result["measurements"].reshape(-1, 4, 3)
    ratio_truth = result["true_inertia_ratios"]
    ratio_filtered = result["filtered_inertia_ratios"]
    parameter_std = np.sqrt(
        np.stack(
            [
                np.diag(covariance)[12:14]
                for covariance in result["filtered_covariances"]
            ]
        )
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    trajectory_axis, ratio_axis, attitude_axis = axes

    for marker_index in range(measurements.shape[1]):
        trajectory_axis.scatter(
            measurements[:, marker_index, 0],
            measurements[:, marker_index, 1],
            s=6,
            alpha=0.10,
            color=COLORS["measurement"],
            edgecolors="none",
            label="Noisy world points" if marker_index == 0 else None,
        )
    trajectory_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True center",
    )
    trajectory_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed center",
    )
    body_axis_colors = ("#0072B2", "#D55E00", "#009E73")
    snapshot_indices = np.linspace(0, len(times) - 1, 6, dtype=int)
    for snapshot_number, snapshot_index in enumerate(snapshot_indices):
        world_points = transform_canonical_points(
            smoothed[snapshot_index, :12]
        ).reshape(-1, 3)
        for body_axis_index, color in enumerate(body_axis_colors, start=1):
            trajectory_axis.plot(
                world_points[[0, body_axis_index], 0],
                world_points[[0, body_axis_index], 1],
                color=color,
                alpha=0.62,
                linewidth=1.2,
                label=(
                    f"Body axis {body_axis_index}" if snapshot_number == 0 else None
                ),
            )
    trajectory_axis.set_title("World-space point observations")
    trajectory_axis.set_xlabel("x position")
    trajectory_axis.set_ylabel("y position")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.legend(fontsize=7, loc="best")

    for index, (label, color) in enumerate(
        (("I₂ / I₁", "#0072B2"), ("I₃ / I₁", "#D55E00"))
    ):
        ratio_std = ratio_filtered[:, index] * parameter_std[:, index]
        ratio_axis.fill_between(
            times,
            ratio_filtered[:, index] - 2.0 * ratio_std,
            ratio_filtered[:, index] + 2.0 * ratio_std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ratio_axis.plot(
            times,
            ratio_filtered[:, index],
            color=color,
            label=f"Estimated {label}",
        )
        ratio_axis.axhline(
            ratio_truth[index],
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"True {label}",
        )
    ratio_axis.set_title("Inferred inertia ratios")
    ratio_axis.set_xlabel("Time [s]")
    ratio_axis.set_ylabel("Principal-inertia ratio")
    ratio_axis.legend(fontsize=7, ncol=2)

    attitude_axis.plot(
        times,
        np.degrees(result["filtered_rotation_error"]),
        color=COLORS["filter"],
        label="Filtered",
    )
    attitude_axis.plot(
        times,
        np.degrees(result["smoothed_rotation_error"]),
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    attitude_axis.fill_between(
        times,
        0.0,
        np.degrees(result["smoothed_rotation_error"]),
        color=COLORS["smoother"],
        alpha=0.08,
        linewidth=0,
    )
    attitude_axis.set_title("Geodesic attitude error")
    attitude_axis.set_xlabel("Time [s]")
    attitude_axis.set_ylabel("Angle [deg]")
    attitude_axis.legend()

    figure.suptitle(
        "Rigid-body state and inertia estimation from noisy point measurements",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_ballistic_tracking(result, output_path, show=False):
    """Plot radar geometry, trajectory recovery, and drag inference."""
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    measurements = result["measurements"]
    radar_position = result["radar_position"]
    measured_positions = radar_position + measurements[:, :1] * np.column_stack(
        (np.cos(measurements[:, 1]), np.sin(measurements[:, 1]))
    )

    figure = plt.figure(figsize=(11.2, 7.4))
    axes = figure.subplot_mosaic(
        [["trajectory", "altitude"], ["trajectory", "drag"]],
        width_ratios=(1.2, 1.0),
    )
    trajectory_axis = axes["trajectory"]
    altitude_axis = axes["altitude"]
    drag_axis = axes["drag"]

    trajectory_axis.scatter(
        measured_positions[:, 0],
        measured_positions[:, 1],
        s=8,
        alpha=0.22,
        color=COLORS["measurement"],
        edgecolors="none",
        label="Radar measurements",
    )
    for ray_number, measurement_index in enumerate(
        np.linspace(0, len(measurements) - 1, 9, dtype=int)
    ):
        measured_position = measured_positions[measurement_index]
        trajectory_axis.plot(
            [radar_position[0], measured_position[0]],
            [radar_position[1], measured_position[1]],
            color="#888888",
            alpha=0.16,
            linewidth=0.8,
            label="Selected radar rays" if ray_number == 0 else None,
        )
    trajectory_axis.scatter(
        radar_position[0],
        radar_position[1],
        marker="^",
        s=85,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
        label="Radar",
    )
    trajectory_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True trajectory",
    )
    trajectory_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered",
    )
    trajectory_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    trajectory_axis.set_title("Range/elevation geometry")
    trajectory_axis.set_xlabel("Downrange [km]")
    trajectory_axis.set_ylabel("Altitude [km]")
    trajectory_axis.set_ylim(
        -1.0,
        max(measured_positions[:, 1].max(), truth[:, 1].max()) + 1.0,
    )
    trajectory_axis.legend(fontsize=8, loc="best")

    altitude_axis.plot(
        times,
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True altitude",
    )
    altitude_axis.plot(
        times,
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered",
    )
    altitude_axis.plot(
        times,
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    altitude_axis.set_title(
        "Altitude reconstruction — position RMSE "
        f"{result['filtered_position_rmse']:.3f} → "
        f"{result['smoothed_position_rmse']:.3f} km"
    )
    altitude_axis.set_xlabel("Time [s]")
    altitude_axis.set_ylabel("Altitude [km]")
    altitude_axis.legend(fontsize=8)

    true_drag = result["true_drag_coefficient"]
    filtered_drag = result["filtered_drag_coefficient"]
    smoothed_drag = result["smoothed_drag_coefficient"]
    log_drag_std = np.sqrt(result["filtered_covariances"][:, 4, 4])
    drag_axis.fill_between(
        times,
        filtered_drag * np.exp(-2.0 * log_drag_std),
        filtered_drag * np.exp(2.0 * log_drag_std),
        color=COLORS["filter"],
        alpha=0.14,
        linewidth=0,
        label="Filtered 95% interval",
    )
    drag_axis.plot(
        times,
        filtered_drag,
        color=COLORS["filter"],
        label="Filtered estimate",
    )
    drag_axis.plot(
        times,
        smoothed_drag,
        color=COLORS["smoother"],
        label="RTS smoothed estimate",
    )
    drag_axis.axhline(
        true_drag,
        color=COLORS["truth"],
        linestyle="--",
        label=f"True coefficient ({true_drag:.3f})",
    )
    drag_axis.set_title("Effective drag-coefficient inference")
    drag_axis.set_xlabel("Time [s]")
    drag_axis.set_ylabel("Drag coefficient [1/km]")
    drag_axis.legend(fontsize=8)

    figure.suptitle(
        "Ballistic radar tracking with aerodynamic parameter inference",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_orbit_determination(result, output_path, show=False):
    """Plot orbit geometry, errors, and ground-station visibility."""
    from matplotlib.patches import Circle

    from examples.orbit_determination import EARTH_RADIUS_KM

    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.3))
    geometry_axis, altitude_axis, error_axis, uncertainty_axis = axes.flat
    geometry_axis.add_patch(
        Circle(
            (0.0, 0.0),
            EARTH_RADIUS_KM,
            facecolor="#E8EEF3",
            edgecolor="#607D8B",
            linewidth=1.2,
            zorder=0,
            label="Earth",
        )
    )
    station_colors = ("#0072B2", "#009E73", "#CC79A7", "#E69F00")
    initial_stations = result["station_positions"][0]
    geometry_axis.scatter(
        initial_stations[:, 0],
        initial_stations[:, 1],
        marker="^",
        s=30,
        color="#607D8B",
        edgecolor="white",
        linewidth=0.5,
        label="Ground stations at start",
        zorder=6,
    )
    for ray_number, observation_index in enumerate(
        np.linspace(0, len(result["visible_station_indices"]) - 1, 12, dtype=int)
    ):
        station_index = result["visible_station_indices"][observation_index][0]
        station_position = result["station_positions"][observation_index, station_index]
        target_position = truth[observation_index + 1, :2]
        geometry_axis.plot(
            [station_position[0], target_position[0]],
            [station_position[1], target_position[1]],
            color=station_colors[ray_number % len(station_colors)],
            alpha=0.20,
            linewidth=0.8,
            label="Visible tracking passes" if ray_number == 0 else None,
            zorder=1,
        )
    geometry_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True orbit",
        zorder=3,
    )
    geometry_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.70,
        label="Filtered",
        zorder=4,
    )
    geometry_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS smoothed",
        zorder=5,
    )
    geometry_axis.set_title("Rotating-station observation geometry")
    geometry_axis.set_xlabel("Inertial x [km]")
    geometry_axis.set_ylabel("Inertial y [km]")
    geometry_axis.set_aspect("equal", adjustable="box")
    geometry_axis.legend(fontsize=7, loc="upper right")

    truth_altitude = np.linalg.norm(truth[:, :2], axis=1) - EARTH_RADIUS_KM
    filtered_altitude = np.linalg.norm(filtered[:, :2], axis=1) - EARTH_RADIUS_KM
    smoothed_altitude = np.linalg.norm(smoothed[:, :2], axis=1) - EARTH_RADIUS_KM
    altitude_axis.plot(
        times / 60.0,
        truth_altitude,
        color=COLORS["truth"],
        linestyle="--",
        label="True",
    )
    altitude_axis.plot(
        times / 60.0,
        filtered_altitude,
        color=COLORS["filter"],
        alpha=0.70,
        label="Filtered",
    )
    altitude_axis.plot(
        times / 60.0,
        smoothed_altitude,
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    altitude_axis.set_title("Recovered orbital altitude")
    altitude_axis.set_xlabel("Time [min]")
    altitude_axis.set_ylabel("Altitude [km]")
    altitude_axis.legend(fontsize=8)

    filtered_error = np.linalg.norm(filtered[:, :2] - truth[:, :2], axis=1)
    smoothed_error = np.linalg.norm(smoothed[:, :2] - truth[:, :2], axis=1)
    error_axis.semilogy(
        times / 60.0,
        filtered_error,
        color=COLORS["filter"],
        label=f"Filtered (RMSE {result['filtered_position_rmse']:.2f} km)",
    )
    error_axis.semilogy(
        times / 60.0,
        smoothed_error,
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {result['smoothed_position_rmse']:.2f} km)",
    )
    error_axis.set_title("Position error over the tracking arc")
    error_axis.set_xlabel("Time [min]")
    error_axis.set_ylabel("Euclidean error [km]")
    error_axis.legend(fontsize=8)

    major_position_std = np.array(
        [
            np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())
            for covariance in result["filtered_covariances"]
        ]
    )
    visible_counts = np.array(
        [len(indices) for indices in result["visible_station_indices"]]
    )
    uncertainty_axis.semilogy(
        times / 60.0,
        major_position_std,
        color=COLORS["filter"],
        label="Filtered major-axis σ",
    )
    visibility_axis = uncertainty_axis.twinx()
    visibility_axis.step(
        times[1:] / 60.0,
        visible_counts,
        where="post",
        color="#009E73",
        alpha=0.55,
        label="Visible stations",
    )
    uncertainty_axis.set_title("Uncertainty and station visibility")
    uncertainty_axis.set_xlabel("Time [min]")
    uncertainty_axis.set_ylabel("Position σ [km]")
    visibility_axis.set_ylabel("Visible stations")
    visibility_axis.set_yticks(np.arange(visible_counts.max() + 1))
    lines, labels = uncertainty_axis.get_legend_handles_labels()
    extra_lines, extra_labels = visibility_axis.get_legend_handles_labels()
    uncertainty_axis.legend(lines + extra_lines, labels + extra_labels, fontsize=8)

    figure.suptitle(
        "Low-Earth orbit determination from range and range rate",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)

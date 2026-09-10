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

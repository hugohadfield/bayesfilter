"""Matplotlib visualization for the TDOA localization example."""

from pathlib import Path

import numpy as np

from examples.plotting import COLORS, _add_covariance_ellipse, _pyplot


def plot_tdoa_emitter_localization(result, output_path, show=False):
    """Plot TDOA hyperbolas, trajectory recovery, and clock calibration."""
    from matplotlib.lines import Line2D

    from examples.tdoa_emitter_localization import MICROSECONDS_PER_KM

    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    receiver_positions = result["receiver_positions"]
    receiver_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

    figure = plt.figure(figsize=(11.2, 7.5))
    axes = figure.subplot_mosaic(
        [["geometry", "error"], ["geometry", "bias"]],
        width_ratios=(1.2, 1.0),
    )
    geometry_axis = axes["geometry"]
    error_axis = axes["error"]
    bias_axis = axes["bias"]

    selected_state_index = np.abs(times - 100.0).argmin()
    selected_measurement_index = selected_state_index - 1
    corrected_measurement = (
        result["measurements"][selected_measurement_index]
        - smoothed[selected_state_index, 4:]
    )
    x_coordinates = np.linspace(-15.0, 15.0, 320)
    y_coordinates = np.linspace(-14.0, 14.0, 300)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    reference_range = np.hypot(
        grid_x - receiver_positions[0, 0],
        grid_y - receiver_positions[0, 1],
    )
    hyperbola_handles = []
    for receiver_index, color in enumerate(receiver_colors, start=1):
        receiver_range = np.hypot(
            grid_x - receiver_positions[receiver_index, 0],
            grid_y - receiver_positions[receiver_index, 1],
        )
        range_difference_us = MICROSECONDS_PER_KM * (
            receiver_range - reference_range
        )
        geometry_axis.contour(
            grid_x,
            grid_y,
            range_difference_us,
            levels=[corrected_measurement[receiver_index - 1]],
            colors=[color],
            linewidths=1.1,
            alpha=0.60,
        )
        hyperbola_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.1,
                label=f"R{receiver_index + 1} − R1 hyperbola",
            )
        )

    geometry_axis.scatter(
        receiver_positions[1:, 0],
        receiver_positions[1:, 1],
        marker="^",
        s=60,
        color=receiver_colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=7,
    )
    geometry_axis.scatter(
        receiver_positions[0, 0],
        receiver_positions[0, 1],
        marker="*",
        s=150,
        color="#222222",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
        label="Reference receiver R1",
    )
    for receiver_index, position in enumerate(receiver_positions):
        geometry_axis.annotate(
            f"R{receiver_index + 1}",
            position,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    geometry_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True emitter",
        zorder=4,
    )
    geometry_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.70,
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
    for ellipse_time in (0.0, 20.0, 60.0, 120.0, 180.0):
        state_index = np.abs(times - ellipse_time).argmin()
        _add_covariance_ellipse(
            geometry_axis,
            filtered[state_index, :2],
            result["filtered_covariances"][state_index, :2, :2],
            COLORS["filter"],
        )
    geometry_axis.scatter(
        truth[selected_state_index, 0],
        truth[selected_state_index, 1],
        marker="o",
        s=45,
        facecolor="white",
        edgecolor=COLORS["truth"],
        linewidth=1.3,
        zorder=8,
        label="Hyperbola epoch",
    )
    geometry_axis.set_title("Receiver geometry and one TDOA epoch")
    geometry_axis.set_xlabel("World x [km]")
    geometry_axis.set_ylabel("World y [km]")
    geometry_axis.set_aspect("equal", adjustable="box")
    geometry_axis.set_xlim(-15.0, 15.0)
    geometry_axis.set_ylim(-14.0, 14.0)
    handles, labels = geometry_axis.get_legend_handles_labels()
    geometry_axis.legend(
        hyperbola_handles + handles,
        [handle.get_label() for handle in hyperbola_handles] + labels,
        fontsize=7,
        ncol=2,
        loc="upper center",
    )

    filtered_error = np.linalg.norm(filtered[:, :2] - truth[:, :2], axis=1)
    smoothed_error = np.linalg.norm(smoothed[:, :2] - truth[:, :2], axis=1)
    error_axis.semilogy(
        times,
        filtered_error,
        color=COLORS["filter"],
        label=f"Filtered (RMSE {result['filtered_position_rmse']:.3f} km)",
    )
    error_axis.semilogy(
        times,
        smoothed_error,
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {result['smoothed_position_rmse']:.3f} km)",
    )
    error_axis.set_title("Emitter position error")
    error_axis.set_xlabel("Time [s]")
    error_axis.set_ylabel("Euclidean error [km]")
    error_axis.legend(fontsize=8)

    for bias_index, color in enumerate(receiver_colors):
        bias_axis.plot(
            times,
            filtered[:, 4 + bias_index],
            color=color,
            alpha=0.25,
            linewidth=1.0,
        )
        bias_axis.plot(
            times,
            smoothed[:, 4 + bias_index],
            color=color,
            label=f"R{bias_index + 2} − R1",
        )
        bias_axis.axhline(
            result["true_clock_biases_us"][bias_index],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
    bias_axis.set_title("Clock-bias inference (solid smoothed, dashed truth)")
    bias_axis.set_xlabel("Time [s]")
    bias_axis.set_ylabel("Relative clock bias [μs]")
    bias_axis.legend(fontsize=8, ncol=2)

    mode = "Jacobian" if result["use_jacobian"] else "Unscented"
    figure.suptitle(
        f"{mode} TDOA localization with receiver clock calibration",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)

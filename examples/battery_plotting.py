"""Matplotlib visualization for the battery state-estimation example."""

from pathlib import Path

import numpy as np

from examples.plotting import COLORS, _pyplot


def plot_battery_state_of_charge(result, output_path, show=False):
    """Plot voltage fit, charge, current, and resistance convergence."""
    plt = _pyplot()
    times_min = result["times"] / 60.0
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]

    figure = plt.figure(figsize=(11.2, 7.8))
    axes = figure.subplot_mosaic(
        [["voltage", "voltage"], ["charge", "resistance"], ["current", "resistance"]],
        height_ratios=(1.0, 1.0, 0.62),
    )

    axes["voltage"].scatter(
        times_min[1:],
        result["measurements"][:, 0],
        s=7,
        alpha=0.22,
        color=COLORS["measurement"],
        edgecolors="none",
        label="Noisy terminal-voltage measurements",
    )
    axes["voltage"].plot(
        times_min,
        result["true_voltage_v"],
        color=COLORS["truth"],
        linestyle="--",
        label="True voltage",
    )
    axes["voltage"].plot(
        times_min,
        result["filtered_voltage_v"],
        color=COLORS["filter"],
        alpha=0.75,
        label="Filtered voltage",
    )
    axes["voltage"].plot(
        times_min,
        result["smoothed_voltage_v"],
        color=COLORS["smoother"],
        label="RTS-smoothed voltage",
    )
    axes["voltage"].set_title(
        "Battery equivalent-circuit estimation — "
        f"charge RMSE {100 * result['filtered_charge_rmse']:.2f} → "
        f"{100 * result['smoothed_charge_rmse']:.2f} percentage points"
    )
    axes["voltage"].set_ylabel("Terminal voltage [V]")
    axes["voltage"].legend(ncol=4, fontsize=8, loc="lower left")

    charge_std = np.sqrt(np.maximum(result["smoothed_covariances"][:, 0, 0], 0.0))
    axes["charge"].fill_between(
        times_min,
        100.0 * (smoothed[:, 0] - 1.96 * charge_std),
        100.0 * (smoothed[:, 0] + 1.96 * charge_std),
        color=COLORS["smoother"],
        alpha=0.13,
        linewidth=0.0,
        label="Smoothed 95% interval",
    )
    axes["charge"].plot(
        times_min,
        100.0 * truth[:, 0],
        color=COLORS["truth"],
        linestyle="--",
        label="True charge",
    )
    axes["charge"].plot(
        times_min,
        100.0 * filtered[:, 0],
        color=COLORS["filter"],
        alpha=0.8,
        label="Filtered",
    )
    axes["charge"].plot(
        times_min,
        100.0 * smoothed[:, 0],
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    axes["charge"].set_ylabel("State of charge [%]")
    axes["charge"].legend(fontsize=8)

    axes["current"].scatter(
        times_min[1:],
        result["measurements"][:, 1],
        s=6,
        alpha=0.25,
        color=COLORS["measurement"],
        edgecolors="none",
        label="Measured",
    )
    axes["current"].plot(
        times_min,
        truth[:, 3],
        color=COLORS["truth"],
        linewidth=1.3,
        label="True load",
    )
    axes["current"].axhline(0.0, color="#777777", linewidth=0.7)
    axes["current"].set_xlabel("Time [min]")
    axes["current"].set_ylabel("Current [A]")
    axes["current"].legend(fontsize=8, ncol=2)

    resistance_std = np.sqrt(
        np.maximum(result["smoothed_covariances"][:, 2, 2], 0.0)
    )
    smoothed_resistance = 1e3 * result["smoothed_resistance_ohm"]
    lower_resistance = smoothed_resistance * np.exp(-1.96 * resistance_std)
    upper_resistance = smoothed_resistance * np.exp(1.96 * resistance_std)
    axes["resistance"].fill_between(
        times_min,
        lower_resistance,
        upper_resistance,
        color=COLORS["smoother"],
        alpha=0.13,
        linewidth=0.0,
        label="Smoothed 95% interval",
    )
    axes["resistance"].axhline(
        1e3 * result["true_resistance_ohm"],
        color=COLORS["truth"],
        linestyle="--",
        label="True resistance",
    )
    axes["resistance"].plot(
        times_min,
        1e3 * result["filtered_resistance_ohm"],
        color=COLORS["filter"],
        alpha=0.8,
        label="Filtered",
    )
    axes["resistance"].plot(
        times_min,
        smoothed_resistance,
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    axes["resistance"].set_xlabel("Time [min]")
    axes["resistance"].set_ylabel("Ohmic resistance [mΩ]")
    axes["resistance"].legend(fontsize=8)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)

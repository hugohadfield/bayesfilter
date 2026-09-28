"""Plotting helpers for the island-navigation example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.boat_navigation import (
    CURRENT,
    HEADING,
    ISLAND_RADIUS_M,
    TRUE_CURRENT_MPS,
    angle_difference,
)


def plot_boat_navigation(result, save_path=None):
    times_h = result["times"] / 3600.0
    truth = result["truth"]
    sun_filtered = result["filtered"]
    sun_rts = result["smoothed"]
    no_sun_filtered = result["no_sun_filtered"]
    no_sun_rts = result["no_sun_smoothed"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    route_axis = axes[0]
    island_angle = np.linspace(0.0, 2.0 * np.pi, 200)
    route_axis.fill(
        ISLAND_RADIUS_M * np.cos(island_angle) / 1000.0,
        ISLAND_RADIUS_M * np.sin(island_angle) / 1000.0,
        alpha=0.12,
        label="Island",
    )
    headlands = result["headlands"] / 1000.0
    route_axis.scatter(headlands[:, 0], headlands[:, 1], marker="^", label="Headlands")
    route_axis.plot(truth[:, 0] / 1000.0, truth[:, 1] / 1000.0, label="Truth")
    route_axis.plot(
        no_sun_rts[:, 0] / 1000.0,
        no_sun_rts[:, 1] / 1000.0,
        label="No Sun RTS",
    )
    route_axis.plot(
        sun_rts[:, 0] / 1000.0,
        sun_rts[:, 1] / 1000.0,
        label="Sun RTS",
    )
    for record in result["landmark_records"]:
        boat = truth[np.argmin(np.abs(result["times"] - record["time"])), :2] / 1000.0
        landmark = record["landmark"] / 1000.0
        route_axis.plot(
            [boat[0], landmark[0]],
            [boat[1], landmark[1]],
            linewidth=0.7,
            alpha=0.35,
        )
    route_axis.set_aspect("equal", adjustable="box")
    route_axis.set_xlabel("East (km)")
    route_axis.set_ylabel("North (km)")
    route_axis.set_title("Sailing around an island")
    route_axis.legend(fontsize=8)

    heading_axis = axes[1]
    heading_axis.plot(
        times_h,
        np.rad2deg(angle_difference(no_sun_filtered[:, HEADING], truth[:, HEADING])),
        alpha=0.65,
        label="No Sun filter",
    )
    heading_axis.plot(
        times_h,
        np.rad2deg(angle_difference(no_sun_rts[:, HEADING], truth[:, HEADING])),
        label="No Sun RTS",
    )
    heading_axis.plot(
        times_h,
        np.rad2deg(angle_difference(sun_filtered[:, HEADING], truth[:, HEADING])),
        alpha=0.65,
        label="Sun filter",
    )
    heading_axis.plot(
        times_h,
        np.rad2deg(angle_difference(sun_rts[:, HEADING], truth[:, HEADING])),
        label="Sun RTS",
    )
    heading_axis.axhline(0.0, linewidth=0.8)
    heading_axis.set_xlabel("Time (h)")
    heading_axis.set_ylabel("Heading error (deg)")
    heading_axis.set_title("Global orientation")
    heading_axis.legend(fontsize=8)

    current_axis = axes[2]
    current_axis.plot(times_h, no_sun_rts[:, CURRENT.start], label="No Sun RTS east")
    current_axis.plot(times_h, no_sun_rts[:, CURRENT.start + 1], label="No Sun RTS north")
    current_axis.plot(times_h, sun_rts[:, CURRENT.start], label="Sun RTS east")
    current_axis.plot(times_h, sun_rts[:, CURRENT.start + 1], label="Sun RTS north")
    current_axis.axhline(TRUE_CURRENT_MPS[0], linestyle="--", label="True east")
    current_axis.axhline(TRUE_CURRENT_MPS[1], linestyle="--", label="True north")
    current_axis.set_xlabel("Time (h)")
    current_axis.set_ylabel("Current (m/s)")
    current_axis.set_title("Estimated ocean current")
    current_axis.legend(fontsize=8)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160)
    return figure

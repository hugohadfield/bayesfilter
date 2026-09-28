"""Plots for examples that originally shipped without figure helpers."""

from pathlib import Path

import numpy as np

from examples.plotting import COLORS, _pyplot


def _save(figure, output_path, show):
    plt = _pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)


def plot_house_thermal_estimation(result, output_path, show=False):
    """Plot temperature reconstruction and inferred building parameters."""
    from examples.thermal_house import (
        TRUE_CAPACITANCE_KWH_PER_K,
        TRUE_RESISTANCE_K_PER_KW,
    )

    plt = _pyplot()
    days = result["times"] / 86400.0
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    temperature_axis, environment_axis, resistance_axis, capacitance_axis = axes.flat

    temperature_axis.plot(days, result["truth"][:, 0], "--", color=COLORS["truth"], label="True indoor")
    temperature_axis.plot(days, result["filtered"][:, 0], color=COLORS["filter"], alpha=0.75, label="Filtered")
    temperature_axis.plot(days, result["smoothed"][:, 0], color=COLORS["smoother"], label="RTS smoothed")
    temperature_axis.scatter(
        days[result["observation_indices"]], result["measurements"], s=5,
        color=COLORS["measurement"], alpha=0.25, label="Thermostat measurements",
    )
    temperature_axis.set_title(
        "Indoor temperature — RMSE "
        f"{result['filtered_temperature_rmse_c']:.2f} → "
        f"{result['smoothed_temperature_rmse_c']:.2f} °C"
    )
    temperature_axis.set_ylabel("Temperature [°C]")
    temperature_axis.legend(fontsize=8)

    environment_axis.plot(days, result["outdoor_temperature_c"], label="Outdoor temperature")
    environment_axis.plot(days, result["setpoint_temperature_c"], label="Thermostat setpoint")
    heater_axis = environment_axis.twinx()
    heater_axis.fill_between(
        days[:-1],
        0,
        result["heater_commands"],
        color="#E69F00",
        alpha=0.15,
        step="post",
    )
    heater_axis.set_ylabel("Heater command")
    environment_axis.set_title("Known environmental forcing")
    environment_axis.set_ylabel("Temperature [°C]")
    environment_axis.legend(fontsize=8)

    resistance_axis.plot(days, result["filtered_resistance_k_per_kw"], color=COLORS["filter"], label="Filtered")
    resistance_axis.plot(days, result["smoothed_resistance_k_per_kw"], color=COLORS["smoother"], label="RTS smoothed")
    resistance_axis.axhline(TRUE_RESISTANCE_K_PER_KW, color=COLORS["truth"], linestyle="--", label="True")
    resistance_axis.set_title("Thermal resistance")
    resistance_axis.set_xlabel("Time [days]")
    resistance_axis.set_ylabel("R [K/kW]")
    resistance_axis.legend(fontsize=8)

    capacitance_axis.plot(days, result["filtered_capacitance_kwh_per_k"], color=COLORS["filter"], label="Filtered")
    capacitance_axis.plot(days, result["smoothed_capacitance_kwh_per_k"], color=COLORS["smoother"], label="RTS smoothed")
    capacitance_axis.axhline(TRUE_CAPACITANCE_KWH_PER_K, color=COLORS["truth"], linestyle="--", label="True")
    capacitance_axis.set_title("Thermal capacitance")
    capacitance_axis.set_xlabel("Time [days]")
    capacitance_axis.set_ylabel("C [kWh/K]")
    capacitance_axis.legend(fontsize=8)
    figure.suptitle("House thermal-state and parameter estimation", fontsize=14)
    figure.tight_layout()
    _save(figure, output_path, show)


def plot_vehicle_wheel_calibration(result, output_path, show=False):
    """Plot speed tracking, GPS outage, wheel radius, and speedometer bias."""
    from examples.vehicle_wheel_calibration import (
        GPS_OUTAGE_END_S,
        GPS_OUTAGE_START_S,
        LOG_WHEEL_RADIUS,
        SPEED,
        SPEEDOMETER_BIAS,
        TRUE_SPEEDOMETER_BIAS_MPS,
        TRUE_WHEEL_RADIUS_M,
    )

    plt = _pyplot()
    minutes = result["times"] / 60.0
    figure, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=True)
    axes[0].plot(minutes, result["truth"][:, SPEED], "--", color=COLORS["truth"], label="True speed")
    axes[0].plot(minutes, result["filtered"][:, SPEED], color=COLORS["filter"], alpha=0.75, label="Filtered")
    axes[0].plot(minutes, result["smoothed"][:, SPEED], color=COLORS["smoother"], label="RTS smoothed")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].set_title(
        "Vehicle calibration — speed RMSE "
        f"{result['filtered_speed_rmse_mps']:.2f} → {result['smoothed_speed_rmse_mps']:.2f} m/s"
    )
    axes[0].legend(fontsize=8, ncol=3)

    axes[1].plot(minutes, np.exp(result["filtered"][:, LOG_WHEEL_RADIUS]), color=COLORS["filter"], label="Filtered radius")
    axes[1].plot(minutes, np.exp(result["smoothed"][:, LOG_WHEEL_RADIUS]), color=COLORS["smoother"], label="RTS radius")
    axes[1].axhline(TRUE_WHEEL_RADIUS_M, color=COLORS["truth"], linestyle="--", label="True radius")
    axes[1].set_ylabel("Wheel radius [m]")
    axes[1].legend(fontsize=8)

    axes[2].plot(minutes, result["filtered"][:, SPEEDOMETER_BIAS], color=COLORS["filter"], label="Filtered bias")
    axes[2].plot(minutes, result["smoothed"][:, SPEEDOMETER_BIAS], color=COLORS["smoother"], label="RTS bias")
    axes[2].axhline(TRUE_SPEEDOMETER_BIAS_MPS, color=COLORS["truth"], linestyle="--", label="True bias")
    axes[2].set_xlabel("Time [min]")
    axes[2].set_ylabel("Speedometer bias [m/s]")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.axvspan(GPS_OUTAGE_START_S / 60.0, GPS_OUTAGE_END_S / 60.0, color="#CC79A7", alpha=0.10)
    axes[0].text(15.0, axes[0].get_ylim()[1], "GPS outage", ha="center", va="top", fontsize=8)
    figure.tight_layout()
    _save(figure, output_path, show)


def plot_wheel_slip_estimation(result, output_path, show=False):
    """Plot speed and inferred longitudinal slip across terrain changes."""
    from examples.wheel_slip_estimation import SPEED, TERRAIN_SEGMENTS

    plt = _pyplot()
    minutes = result["times"] / 60.0
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True)
    axes[0].plot(minutes, result["truth"][:, SPEED], "--", color=COLORS["truth"], label="True speed")
    axes[0].plot(minutes, result["filtered"][:, SPEED], color=COLORS["filter"], alpha=0.75, label="Filtered")
    axes[0].plot(minutes, result["smoothed"][:, SPEED], color=COLORS["smoother"], label="RTS smoothed")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(minutes, 100 * result["true_slip"], "--", color=COLORS["truth"], label="True slip")
    axes[1].plot(minutes, 100 * result["filtered_slip"], color=COLORS["filter"], alpha=0.75, label="Filtered")
    axes[1].plot(minutes, 100 * result["smoothed_slip"], color=COLORS["smoother"], label="RTS smoothed")
    axes[1].set_xlabel("Time [min]")
    axes[1].set_ylabel("Longitudinal slip [%]")
    axes[1].legend(fontsize=8, ncol=3)
    terrain_colors = ("#56B4E9", "#E69F00", "#D55E00", "#009E73", "#56B4E9")
    for (start_s, end_s, name, _slip), color in zip(TERRAIN_SEGMENTS, terrain_colors):
        for axis in axes:
            axis.axvspan(start_s / 60.0, end_s / 60.0, color=color, alpha=0.07)
        axes[1].text((start_s + end_s) / 120.0, axes[1].get_ylim()[1], name, ha="center", va="top", fontsize=8)
    figure.suptitle(
        "Wheel-slip estimation across changing terrain — RMSE "
        f"{100 * result['filtered_slip_rmse']:.2f} → {100 * result['smoothed_slip_rmse']:.2f} points",
        fontsize=14,
    )
    figure.tight_layout()
    _save(figure, output_path, show)


def plot_vehicle_coastdown(result, output_path, show=False):
    """Plot coast-down speed and drag/rolling-resistance inference."""
    from examples.vehicle_coastdown import (
        DRAG_AREA,
        ROLLING_RESISTANCE,
        SPEED,
        TRUE_DRAG_AREA_M2,
        TRUE_ROLLING_RESISTANCE,
    )

    plt = _pyplot()
    times = result["times"]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    axes[0].scatter(times, result["gps_measurements"], s=9, color=COLORS["measurement"], alpha=0.35, label="GPS speed")
    axes[0].plot(times, result["truth"][:, SPEED], "--", color=COLORS["truth"], label="True speed")
    axes[0].plot(times, result["filtered"][:, SPEED], color=COLORS["filter"], alpha=0.75, label="Filtered")
    axes[0].plot(times, result["smoothed"][:, SPEED], color=COLORS["smoother"], label="RTS smoothed")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].set_title(
        "Flat-road coast-down — speed RMSE "
        f"{result['filtered_speed_rmse_mps']:.2f} → {result['smoothed_speed_rmse_mps']:.2f} m/s"
    )
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].plot(times, result["filtered"][:, ROLLING_RESISTANCE], color=COLORS["filter"], label="Filtered rolling resistance")
    axes[1].plot(times, result["smoothed"][:, ROLLING_RESISTANCE], color=COLORS["smoother"], label="RTS rolling resistance")
    axes[1].axhline(TRUE_ROLLING_RESISTANCE, color=COLORS["truth"], linestyle="--", label="True rolling resistance")
    drag_axis = axes[1].twinx()
    drag_axis.plot(times, result["filtered"][:, DRAG_AREA], color="#009E73", alpha=0.75, label="Filtered drag area")
    drag_axis.plot(times, result["smoothed"][:, DRAG_AREA], color="#CC79A7", label="RTS drag area")
    drag_axis.axhline(TRUE_DRAG_AREA_M2, color="#555555", linestyle=":", label="True drag area")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Rolling coefficient")
    drag_axis.set_ylabel("Drag area CdA [m²]")
    lines, labels = axes[1].get_legend_handles_labels()
    extra_lines, extra_labels = drag_axis.get_legend_handles_labels()
    axes[1].legend(lines + extra_lines, labels + extra_labels, fontsize=8, ncol=2)
    figure.tight_layout()
    _save(figure, output_path, show)

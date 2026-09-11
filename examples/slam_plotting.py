"""Matplotlib visualization for the known-correspondence SLAM example."""

from pathlib import Path

import numpy as np

from examples.plotting import COLORS, _add_covariance_ellipse, _pyplot


def plot_known_correspondence_slam(result, output_path, show=False):
    """Plot SLAM geometry, errors, landmark convergence, and visibility."""
    plt = _pyplot()
    times = result["times"]
    truth = result["truth"]
    filtered = result["filtered"]
    smoothed = result["smoothed"]
    landmark_positions = result["landmark_positions"]
    final_landmarks = filtered[-1, 5:].reshape(-1, 2)

    figure = plt.figure(figsize=(11.4, 7.8))
    axes = figure.subplot_mosaic(
        [["map", "position"], ["map", "landmarks"], ["map", "visibility"]],
        width_ratios=(1.35, 1.0),
    )
    map_axis = axes["map"]

    ray_indices = np.linspace(0, len(result["visible_landmark_indices"]) - 1, 13, dtype=int)
    for ray_number, measurement_index in enumerate(ray_indices):
        state_index = measurement_index + 1
        visible = result["visible_landmark_indices"][measurement_index]
        if not visible:
            continue
        landmark_index = visible[ray_number % len(visible)]
        landmark = landmark_positions[landmark_index]
        robot = truth[state_index, :2]
        map_axis.plot(
            [robot[0], landmark[0]],
            [robot[1], landmark[1]],
            color="#009E73",
            alpha=0.18,
            linewidth=0.9,
            label="Range/bearing observations" if ray_number == 0 else None,
            zorder=1,
        )

    map_axis.plot(
        truth[:, 0],
        truth[:, 1],
        color=COLORS["truth"],
        linestyle="--",
        label="True robot trajectory",
        zorder=3,
    )
    map_axis.plot(
        filtered[:, 0],
        filtered[:, 1],
        color=COLORS["filter"],
        alpha=0.7,
        label="Filtered trajectory",
        zorder=4,
    )
    map_axis.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=COLORS["smoother"],
        label="RTS-smoothed trajectory",
        zorder=5,
    )
    for state_index in range(30, len(smoothed), 65):
        _add_covariance_ellipse(
            map_axis,
            smoothed[state_index, :2],
            result["smoothed_covariances"][state_index, :2, :2],
            COLORS["smoother"],
        )
    map_axis.scatter(
        result["initial_landmark_estimates"][:, 0],
        result["initial_landmark_estimates"][:, 1],
        marker="x",
        s=48,
        linewidth=1.4,
        color=COLORS["measurement"],
        label="Initial landmark guesses",
        zorder=4,
    )
    map_axis.scatter(
        landmark_positions[:, 0],
        landmark_positions[:, 1],
        marker="*",
        s=105,
        color=COLORS["truth"],
        edgecolor="white",
        linewidth=0.7,
        label="True landmarks",
        zorder=7,
    )
    map_axis.scatter(
        final_landmarks[:, 0],
        final_landmarks[:, 1],
        marker="o",
        s=38,
        facecolor=COLORS["smoother"],
        edgecolor="white",
        linewidth=0.7,
        label="Final landmark estimates",
        zorder=8,
    )
    for landmark_index, landmark in enumerate(final_landmarks):
        coordinates = slice(5 + 2 * landmark_index, 7 + 2 * landmark_index)
        _add_covariance_ellipse(
            map_axis,
            landmark,
            result["filtered_covariances"][-1, coordinates, coordinates],
            COLORS["smoother"],
        )
        map_axis.annotate(
            f"L{landmark_index + 1}",
            landmark_positions[landmark_index],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for state_index in range(0, len(truth), 65):
        direction = 0.55 * np.array(
            [np.cos(truth[state_index, 2]), np.sin(truth[state_index, 2])]
        )
        map_axis.arrow(
            truth[state_index, 0],
            truth[state_index, 1],
            direction[0],
            direction[1],
            width=0.025,
            head_width=0.18,
            color=COLORS["truth"],
            alpha=0.7,
            length_includes_head=True,
            zorder=6,
        )
    map_axis.set_title("Joint trajectory and landmark-map reconstruction")
    map_axis.set_xlabel("World x [m]")
    map_axis.set_ylabel("World y [m]")
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.legend(fontsize=7, loc="lower right")

    filtered_position_error = np.linalg.norm(filtered[:, :2] - truth[:, :2], axis=1)
    smoothed_position_error = np.linalg.norm(smoothed[:, :2] - truth[:, :2], axis=1)
    axes["position"].semilogy(
        times,
        filtered_position_error,
        color=COLORS["filter"],
        label=f"Filtered (RMSE {result['filtered_position_rmse']:.3f} m)",
    )
    axes["position"].semilogy(
        times,
        smoothed_position_error,
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {result['smoothed_position_rmse']:.3f} m)",
    )
    axes["position"].set_title("Robot position error")
    axes["position"].set_xlabel("Time [s]")
    axes["position"].set_ylabel("Euclidean error [m]")
    axes["position"].legend(fontsize=8)

    filtered_landmarks = filtered[:, 5:].reshape(len(filtered), -1, 2)
    smoothed_landmarks = smoothed[:, 5:].reshape(len(smoothed), -1, 2)
    filtered_map_error = np.sqrt(
        np.mean(np.sum((filtered_landmarks - landmark_positions) ** 2, axis=2), axis=1)
    )
    smoothed_map_error = np.sqrt(
        np.mean(np.sum((smoothed_landmarks - landmark_positions) ** 2, axis=2), axis=1)
    )
    axes["landmarks"].semilogy(
        times,
        filtered_map_error,
        color=COLORS["filter"],
        label="Filtered map",
    )
    axes["landmarks"].semilogy(
        times,
        smoothed_map_error,
        color=COLORS["smoother"],
        label="RTS-smoothed map",
    )
    axes["landmarks"].axhline(
        result["final_landmark_rmse"],
        color=COLORS["truth"],
        linestyle=":",
        linewidth=1.0,
        label=f"Final RMSE {result['final_landmark_rmse']:.3f} m",
    )
    axes["landmarks"].set_title("Landmark-map convergence")
    axes["landmarks"].set_xlabel("Time [s]")
    axes["landmarks"].set_ylabel("Map RMSE [m]")
    axes["landmarks"].legend(fontsize=8)

    visible_counts = np.array(
        [len(indices) for indices in result["visible_landmark_indices"]]
    )
    axes["visibility"].step(
        times[1:],
        visible_counts,
        where="post",
        color="#009E73",
        label="Visible landmarks",
    )
    axes["visibility"].fill_between(
        times[1:],
        0,
        visible_counts,
        step="post",
        color="#009E73",
        alpha=0.12,
    )
    axes["visibility"].set_title("Changing observation geometry")
    axes["visibility"].set_xlabel("Time [s]")
    axes["visibility"].set_ylabel("Landmarks in view")
    axes["visibility"].set_yticks(np.arange(visible_counts.max() + 1))
    axes["visibility"].set_ylim(0, visible_counts.max() + 0.5)

    figure.suptitle(
        "Known-correspondence SLAM from odometry, range, and bearing",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)

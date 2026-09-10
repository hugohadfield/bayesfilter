"""Generate the figures used in the project README.

Matplotlib is intentionally a documentation-only dependency. Run this script
from the repository root after installing Matplotlib to regenerate the PNGs.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


OUTPUT_DIRECTORY = ROOT / "docs" / "figures"

COLORS = {
    "truth": "#222222",
    "measurement": "#8C8C8C",
    "filter": "#2474B5",
    "smoother": "#D55E00",
}


def _configure_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _observation(value, covariance, observation_func, jacobian_func):
    return Observation(
        observation=np.asarray(value),
        noise_covariance=covariance,
        observation_func=observation_func,
        jacobian_func=jacobian_func,
    )


def generate_constant_velocity_figure():
    rng = np.random.default_rng(7)
    times = np.linspace(0.0, 12.0, 61)
    true_position = 0.7 * times + 0.6 * np.sin(0.7 * times)
    true_velocity = 0.7 + 0.42 * np.cos(0.7 * times)
    measurement_std = 0.55
    measurements = true_position[1:] + rng.normal(
        0.0,
        measurement_std,
        len(times) - 1,
    )

    def transition(state, delta_t_s):
        position, velocity = state
        return np.array([position + delta_t_s * velocity, velocity])

    def transition_jacobian(_state, delta_t_s):
        return np.array([[1.0, delta_t_s], [0.0, 1.0]])

    def observe_position(state):
        return np.array([state[0]])

    def observation_jacobian(_state):
        return np.array([[1.0, 0.0]])

    model = StateTransitionModel(
        transition,
        np.diag([0.012, 0.025]),
        transition_jacobian,
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(
            mean=np.array([true_position[0] + 0.8, 0.0]),
            covariance=np.diag([1.2, 0.5]),
        ),
    )
    measurement_covariance = np.array([[measurement_std**2]])
    observations = [
        _observation(
            [measurement],
            measurement_covariance,
            observe_position,
            observation_jacobian,
        )
        for measurement in measurements
    ]
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=True,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=True,
    )

    filtered_position = np.array([state.mean()[0] for state in filter_states])
    smoothed_position = np.array([state.mean()[0] for state in smoother_states])
    filtered_std = np.sqrt(
        [state.covariance()[0, 0] for state in filter_states]
    )
    smoothed_std = np.sqrt(
        [state.covariance()[0, 0] for state in smoother_states]
    )
    filter_rmse = np.sqrt(np.mean((filtered_position - true_position) ** 2))
    smoother_rmse = np.sqrt(np.mean((smoothed_position - true_position) ** 2))

    figure, (position_axis, velocity_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    position_axis.scatter(
        times[1:],
        measurements,
        s=18,
        color=COLORS["measurement"],
        alpha=0.65,
        edgecolors="none",
        label="Measurements",
        zorder=2,
    )
    position_axis.plot(
        times,
        true_position,
        color=COLORS["truth"],
        linestyle="--",
        label="True position",
        zorder=3,
    )
    position_axis.plot(
        times,
        filtered_position,
        color=COLORS["filter"],
        label=f"Filtered (RMSE {filter_rmse:.2f})",
        zorder=4,
    )
    position_axis.fill_between(
        times,
        filtered_position - 2.0 * filtered_std,
        filtered_position + 2.0 * filtered_std,
        color=COLORS["filter"],
        alpha=0.12,
        linewidth=0,
    )
    position_axis.plot(
        times,
        smoothed_position,
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {smoother_rmse:.2f})",
        zorder=5,
    )
    position_axis.fill_between(
        times,
        smoothed_position - 2.0 * smoothed_std,
        smoothed_position + 2.0 * smoothed_std,
        color=COLORS["smoother"],
        alpha=0.10,
        linewidth=0,
    )
    position_axis.set_title("Constant-velocity filtering and RTS smoothing")
    position_axis.set_ylabel("Position")
    position_axis.legend(ncol=2, loc="upper left")

    filtered_velocity = np.array([state.mean()[1] for state in filter_states])
    smoothed_velocity = np.array([state.mean()[1] for state in smoother_states])
    velocity_axis.plot(
        times,
        true_velocity,
        color=COLORS["truth"],
        linestyle="--",
        label="True velocity",
    )
    velocity_axis.plot(
        times,
        filtered_velocity,
        color=COLORS["filter"],
        label="Filtered",
    )
    velocity_axis.plot(
        times,
        smoothed_velocity,
        color=COLORS["smoother"],
        label="RTS smoothed",
    )
    velocity_axis.set_xlabel("Time [s]")
    velocity_axis.set_ylabel("Velocity")
    velocity_axis.legend(ncol=3, loc="upper right")

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "constant-velocity-tracking.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _add_covariance_ellipse(axis, mean, covariance, color):
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


def generate_planar_tracking_figure():
    rng = np.random.default_rng(11)
    times = np.linspace(0.0, 15.0, 76)
    angle = -1.1 + 0.22 * times
    true_x = 4.0 * np.cos(angle)
    true_y = 3.0 * np.sin(angle)
    true_vx = -0.88 * np.sin(angle)
    true_vy = 0.66 * np.cos(angle)
    truth = np.column_stack((true_x, true_y, true_vx, true_vy))
    measurement_std = 0.45
    measurements = truth[1:, :2] + rng.normal(
        0.0,
        measurement_std,
        (len(times) - 1, 2),
    )

    def transition(state, delta_t_s):
        x_position, y_position, x_velocity, y_velocity = state
        return np.array(
            [
                x_position + delta_t_s * x_velocity,
                y_position + delta_t_s * y_velocity,
                x_velocity,
                y_velocity,
            ]
        )

    def transition_jacobian(_state, delta_t_s):
        return np.array(
            [
                [1.0, 0.0, delta_t_s, 0.0],
                [0.0, 1.0, 0.0, delta_t_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def observe_position(state):
        return state[:2]

    def observation_jacobian(_state):
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    model = StateTransitionModel(
        transition,
        np.diag([0.008, 0.008, 0.03, 0.03]),
        transition_jacobian,
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(
            mean=truth[0] + np.array([0.7, -0.5, 0.2, -0.2]),
            covariance=np.diag([0.9, 0.9, 0.35, 0.35]),
        ),
    )
    measurement_covariance = (measurement_std**2) * np.eye(2)
    observations = [
        _observation(
            measurement,
            measurement_covariance,
            observe_position,
            observation_jacobian,
        )
        for measurement in measurements
    ]
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=True,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=True,
    )

    filtered_xy = np.array([state.mean()[:2] for state in filter_states])
    smoothed_xy = np.array([state.mean()[:2] for state in smoother_states])
    filter_rmse = np.sqrt(
        np.mean(np.sum((filtered_xy - truth[:, :2]) ** 2, axis=1))
    )
    smoother_rmse = np.sqrt(
        np.mean(np.sum((smoothed_xy - truth[:, :2]) ** 2, axis=1))
    )

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
        true_x,
        true_y,
        color=COLORS["truth"],
        linestyle="--",
        label="True trajectory",
        zorder=3,
    )
    axis.plot(
        filtered_xy[:, 0],
        filtered_xy[:, 1],
        color=COLORS["filter"],
        label=f"Filtered (RMSE {filter_rmse:.2f})",
        zorder=4,
    )
    axis.plot(
        smoothed_xy[:, 0],
        smoothed_xy[:, 1],
        color=COLORS["smoother"],
        label=f"RTS smoothed (RMSE {smoother_rmse:.2f})",
        zorder=5,
    )
    for index in range(5, len(smoother_states), 10):
        state = smoother_states[index]
        _add_covariance_ellipse(
            axis,
            state.mean()[:2],
            state.covariance()[:2, :2],
            COLORS["smoother"],
        )
    axis.scatter(
        true_x[0],
        true_y[0],
        marker="o",
        s=45,
        facecolor="white",
        edgecolor=COLORS["truth"],
        linewidth=1.5,
        zorder=6,
    )
    axis.scatter(
        true_x[-1],
        true_y[-1],
        marker="s",
        s=42,
        facecolor=COLORS["truth"],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    axis.set_title("Planar trajectory reconstruction")
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "planar-trajectory-tracking.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()
    generate_constant_velocity_figure()
    generate_planar_tracking_figure()


if __name__ == "__main__":
    main()

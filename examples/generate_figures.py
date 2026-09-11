"""Regenerate every figure shown in the examples documentation."""

from pathlib import Path

from examples.ballistic_tracking import run_ballistic_tracking
from examples.bearings_only import run_bearings_only_tracking
from examples.heading_coupled import run_heading_coupled_tracking
from examples.linear_tracking import run_planar_tracking
from examples.plotting import (
    plot_ballistic_tracking,
    plot_bearings_only_tracking,
    plot_heading_coupled_tracking,
    plot_inertia_estimation,
    plot_orbit_determination,
    plot_planar_tracking,
    plot_readme_inertia_summary,
    plot_rigid_body_tracking,
    run_and_plot_linear_example,
)
from examples.orbit_determination import run_orbit_determination
from examples.rigid_body import (
    run_inertia_estimation_rigid_body,
    run_known_inertia_rigid_body,
)
from examples.tdoa_emitter_localization import run_tdoa_emitter_localization
from examples.tdoa_plotting import plot_tdoa_emitter_localization


def main():
    for derivative_order, model_name in (
        (1, "constant-velocity"),
        (2, "constant-acceleration"),
    ):
        for dimension in (1, 2, 3):
            run_and_plot_linear_example(
                dimension,
                derivative_order,
                f"{model_name}-{dimension}d.png",
            )

    figure_directory = Path(__file__).parent / "figures"
    plot_planar_tracking(
        run_planar_tracking(),
        figure_directory / "planar-trajectory-tracking.png",
    )
    plot_bearings_only_tracking(
        run_bearings_only_tracking(),
        figure_directory / "bearings-only-tracking.png",
    )
    plot_ballistic_tracking(
        run_ballistic_tracking(),
        figure_directory / "ballistic-drag-estimation.png",
    )
    plot_orbit_determination(
        run_orbit_determination(),
        figure_directory / "orbit-determination.png",
    )
    plot_heading_coupled_tracking(
        run_heading_coupled_tracking(),
        figure_directory / "heading-coupled-tracking.png",
    )
    plot_tdoa_emitter_localization(
        run_tdoa_emitter_localization(),
        figure_directory / "tdoa-emitter-localization.png",
    )

    plot_rigid_body_tracking(
        run_known_inertia_rigid_body(),
        figure_directory / "rigid-body-known-inertia.png",
    )
    inertia_result = run_inertia_estimation_rigid_body()
    plot_inertia_estimation(
        inertia_result,
        figure_directory / "rigid-body-inertia-estimation.png",
    )
    plot_readme_inertia_summary(
        inertia_result,
        figure_directory / "readme-rigid-body-inference.png",
    )


if __name__ == "__main__":
    main()

"""Regenerate every figure shown in the examples documentation."""

from pathlib import Path

from examples.heading_coupled import run_heading_coupled_tracking
from examples.linear_tracking import run_planar_tracking
from examples.plotting import (
    plot_heading_coupled_tracking,
    plot_inertia_estimation,
    plot_planar_tracking,
    plot_readme_inertia_summary,
    plot_rigid_body_tracking,
    run_and_plot_linear_example,
)
from examples.rigid_body import (
    run_inertia_estimation_rigid_body,
    run_known_inertia_rigid_body,
)


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
    plot_heading_coupled_tracking(
        run_heading_coupled_tracking(),
        figure_directory / "heading-coupled-tracking.png",
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

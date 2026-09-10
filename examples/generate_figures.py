"""Regenerate every figure shown in the examples documentation."""

from pathlib import Path

from examples.plotting import (
    plot_inertia_estimation,
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
    plot_rigid_body_tracking(
        run_known_inertia_rigid_body(),
        figure_directory / "rigid-body-known-inertia.png",
    )
    plot_inertia_estimation(
        run_inertia_estimation_rigid_body(),
        figure_directory / "rigid-body-inertia-estimation.png",
    )


if __name__ == "__main__":
    main()

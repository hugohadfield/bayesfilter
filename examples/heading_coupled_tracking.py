from pathlib import Path

from examples.heading_coupled import run_heading_coupled_tracking
from examples.plotting import plot_heading_coupled_tracking


if __name__ == "__main__":
    result = run_heading_coupled_tracking()
    path = Path(__file__).parent / "figures" / "heading-coupled-tracking.png"
    plot_heading_coupled_tracking(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f}"
    )
    print(
        f"heading RMSE: {result['filtered_heading_rmse']:.3f} -> "
        f"{result['smoothed_heading_rmse']:.3f} rad"
    )
    print(f"saved {path}")

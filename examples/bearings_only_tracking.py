from pathlib import Path

from examples.bearings_only import run_bearings_only_tracking
from examples.plotting import plot_bearings_only_tracking


if __name__ == "__main__":
    result = run_bearings_only_tracking()
    path = Path(__file__).parent / "figures" / "bearings-only-tracking.png"
    plot_bearings_only_tracking(result, path)
    print(
        f"position RMSE: {result['filtered_position_rmse']:.3f} -> "
        f"{result['smoothed_position_rmse']:.3f}"
    )
    print(f"saved {path}")

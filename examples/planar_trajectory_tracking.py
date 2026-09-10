from pathlib import Path

from examples.linear_tracking import run_planar_tracking
from examples.plotting import plot_planar_tracking


if __name__ == "__main__":
    result = run_planar_tracking()
    path = Path(__file__).parent / "figures" / "planar-trajectory-tracking.png"
    plot_planar_tracking(result, path)
    print(
        f"filtered RMSE={result['filtered_rmse']:.3f}, "
        f"smoothed RMSE={result['smoothed_rmse']:.3f}"
    )
    print(f"saved {path}")

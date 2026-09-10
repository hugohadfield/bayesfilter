from pathlib import Path

from examples.plotting import plot_inertia_estimation
from examples.rigid_body import run_inertia_estimation_rigid_body


if __name__ == "__main__":
    result = run_inertia_estimation_rigid_body()
    path = Path(__file__).parent / "figures" / "rigid-body-inertia-estimation.png"
    plot_inertia_estimation(result, path)
    print(f"saved {path}")

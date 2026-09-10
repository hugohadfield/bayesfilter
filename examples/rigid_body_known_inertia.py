from pathlib import Path

from examples.plotting import plot_rigid_body_tracking
from examples.rigid_body import run_known_inertia_rigid_body


if __name__ == "__main__":
    result = run_known_inertia_rigid_body()
    path = Path(__file__).parent / "figures" / "rigid-body-known-inertia.png"
    plot_rigid_body_tracking(result, path)
    print(f"saved {path}")

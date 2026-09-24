# BayesFilter

BayesFilter is a Python library for Bayesian filtering and smoothing. This library provides tools for implementing Bayesian filters, Rauch-Tung-Striebel smoothers, and other related methods. The only dependency is NumPy.

![Rigid-body state and inertia estimation from noisy world-space point measurements](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/readme-rigid-body-inference.png)

## Installation

Install the released package from PyPI:

```bash
python -m pip install bayesfilter
```

For local development, install the package with its test dependencies:

```bash
python -m pip install -e '.[test]'
```

## Constant-velocity example

The state below contains one-dimensional position and velocity. Observations
measure position only.

```python
import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


def transition(state, delta_t_s):
    position, velocity = state
    return np.array([position + delta_t_s * velocity, velocity])


def transition_jacobian(_state, delta_t_s):
    return np.array([
        [1.0, delta_t_s],
        [0.0, 1.0],
    ])


def observe_position(state):
    return np.array([state[0]])


def observation_jacobian(_state):
    return np.array([[1.0, 0.0]])


transition_model = StateTransitionModel(
    transition_func=transition,
    transition_noise_covariance=np.diag([0.01, 0.001]),
    transition_jacobian_func=transition_jacobian,
)
initial_state = Gaussian(
    mean=np.array([0.0, 1.0]),
    covariance=np.diag([1.0, 0.25]),
)
bayes_filter = BayesianFilter(transition_model, initial_state)

rng = np.random.default_rng(0)
observation_times = np.arange(0.0, 10.5, 0.5)
position_measurements = observation_times + rng.normal(
    loc=0.0,
    scale=0.5,
    size=len(observation_times),
)
observations = [
    Observation(
        observation=np.array([position]),
        noise_covariance=np.array([[0.5**2]]),
        observation_func=observe_position,
        jacobian_func=observation_jacobian,
    )
    for position in position_measurements
]

filter_states, filter_times = bayes_filter.run(
    observations,
    observation_times,
    rate_hz=20.0,
    use_jacobian=True,
)

smoother_states = RTS(bayes_filter).apply(
    filter_states,
    filter_times,
    use_jacobian=True,
)
```

`BayesianFilter.run()` evaluates the model on its own fixed-rate time grid and
returns that grid as `filter_times`. Always pass `filter_times`, rather than the
original observation timestamps, to `RTS.apply()`. This matters whenever the
transition depends on `delta_t_s`.

As a convenience, filtering and smoothing can be run together:

```python
bayes_filter = BayesianFilter(transition_model, initial_state)
smoother_states = RTS(bayes_filter).smooth(
    observations,
    observation_times,
    rate_hz=20.0,
    use_jacobian=True,
)
```

Use the explicit `run()` followed by `apply()` form when the corresponding
filter timestamps are also needed.

![One-dimensional constant-velocity filtering and RTS smoothing](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-velocity-1d.png)

*Noisy position measurements are filtered forward and then refined by the RTS
smoother; velocity is inferred without being observed directly.*

## Jacobian and unscented modes

Set `use_jacobian=True` on filtering and smoothing calls to use the supplied
transition and observation Jacobians. Both Jacobian functions are required in
this mode.

Set `use_jacobian=False` to use unscented propagation. Jacobian functions may
then be omitted:

```python
filter_states, filter_times = bayes_filter.run(
    observations,
    observation_times,
    rate_hz=20.0,
    use_jacobian=False,
)
```

For a Jacobian transition with covariance `P` and transition matrix `F`, the
predicted covariance and state-to-prediction cross-covariance are

```text
P_pred = F @ P @ F.T + Q
cross_cov = P @ F.T
```

The RTS smoother gain is therefore

```text
G = cross_cov @ inv(P_pred)
```

The unscented path obtains the corresponding cross-covariance directly from
the propagated sigma points.

## Synchronous observations

When there is exactly one observation per transition interval, use
`run_synchronous()`:

```python
states = bayes_filter.run_synchronous(
    observations,
    times_s,
    use_jacobian=True,
)
```

For `N` timestamps this method consumes at most `N - 1` observations and
returns the initial state followed by each updated state. Use `run()` for
irregularly timed observations or when several observations may fall within a
single filter interval.

## Covariance conventions

All noise arguments are covariance matrices, not standard deviations. For a
scalar measurement with standard deviation `sigma`, use
`np.array([[sigma**2]])`.

The transition noise covariance is added once per prediction performed by the
filter. Observation covariances are added during their corresponding updates.

## Examples

The examples generate deterministic synthetic observations, run a Bayesian
filter and RTS smoother, print quantitative errors, and save Matplotlib figures
under `examples/figures/`.

Install Matplotlib separately when running the examples:

```bash
python -m pip install -e .
python -m pip install -r examples/requirements.txt
```

Matplotlib is not a BayesFilter runtime dependency.

### Running the examples

| Model | State | Command |
| --- | --- | --- |
| Constant velocity, 1D | `[p, v]` | `python -m examples.constant_velocity_1d` |
| Constant velocity, 2D | `[pₓ, pᵧ, vₓ, vᵧ]` | `python -m examples.constant_velocity_2d` |
| Constant velocity, 3D | `[p, v]`, with three-vectors | `python -m examples.constant_velocity_3d` |
| Planar trajectory tracking | `[pₓ, pᵧ, vₓ, vᵧ]` | `python -m examples.planar_trajectory_tracking` |
| Bearings-only target tracking | `[pₓ, pᵧ, vₓ, vᵧ]` | `python -m examples.bearings_only_tracking` |
| Black-box INS + GPS fusion | `[t_GL, ψ_GL, v_drift, ω_drift]` | `python -m examples.ins_gps_fusion` |
| TDOA emitter localization | `[pₓ, pᵧ, vₓ, vᵧ, b₂, …, b₅]` | `python -m examples.tdoa_emitter_localization` |
| Ballistic radar tracking and drag inference | `[x, h, vₓ, vₕ, log(c_d)]` | `python -m examples.ballistic_tracking` |
| Orbit determination from ground stations | `[rₓ, rᵧ, vₓ, vᵧ]` | `python -m examples.orbit_determination` |
| Heading-coupled tracking | `[pₓ, pᵧ, ψ, v, κ]` | `python -m examples.heading_coupled_tracking` |
| Battery charge and resistance estimation | `[q, vₚ, log(R₀), I]` | `python -m examples.battery_state_of_charge` |
| Known-correspondence landmark SLAM | `[pₓ, pᵧ, ψ, v, ω, ℓ₁, …, ℓ₈]` | `python -m examples.known_correspondence_slam` |
| Constant acceleration, 1D | `[p, v, a]` | `python -m examples.constant_acceleration_1d` |
| Constant acceleration, 2D | `[p, v, a]`, with two-vectors | `python -m examples.constant_acceleration_2d` |
| Constant acceleration, 3D | `[p, v, a]`, with three-vectors | `python -m examples.constant_acceleration_3d` |
| Rigid body, known inertia | `[p, v, ϕ, ω]` | `python -m examples.rigid_body_known_inertia` |
| Rigid body, inferred inertia | `[p, v, ϕ, ω, η₂, η₃]` | `python -m examples.rigid_body_inertia_estimation` |

Regenerate every figure, including the figure at the top of this README, with:

```bash
python -m examples.generate_figures
```

All examples use fixed random seeds. Their simulation functions can be
imported without importing Matplotlib.

### Constant-velocity tracking

For spatial dimension `d`, the state contains position and velocity:

```text
x = [p, v]
```

The transition and position observation matrices are

```text
F = [[I, dt I],
     [0,    I]]

H = [I, 0]
```

Process noise uses the discrete white-acceleration covariance

```text
Q = q [[dt³/3, dt²/2],
       [dt²/2,    dt]] ⊗ I.
```

| 1D | 2D | 3D |
| --- | --- | --- |
| ![Constant-velocity tracking in one dimension](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-velocity-1d.png) | ![Constant-velocity tracking in two dimensions](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-velocity-2d.png) | ![Constant-velocity tracking in three dimensions](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-velocity-3d.png) |

The same position-and-velocity state can reconstruct a curved planar
trajectory. The ellipses below show selected 95% smoothed position-confidence
regions.

![Planar trajectory reconstruction from noisy position observations](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/planar-trajectory-tracking.png)

### Bearings-only target tracking

This example tracks a curved target without observing range or position
directly. Two fixed sensors report noisy world-frame bearing unit vectors:

```text
x = [pₓ, pᵧ, vₓ, vᵧ]

hᵢ(x) = (p - sᵢ) / ‖p - sᵢ‖,
```

where `sᵢ` is the position of sensor `i`. Encoding bearing as
`[cos(bearing), sin(bearing)]` avoids a discontinuous scalar residual at
`±π`. The observation remains strongly nonlinear and supplies no information
in the instantaneous line-of-sight direction.

Sensor 1 reports at 10 Hz. Sensor 2 reports at 2 Hz but is unavailable from
5.5–11 seconds. During that interval, range is only weakly observable from
target motion, so the filtered uncertainty grows along the remaining bearing
geometry. When sensor 2 returns, triangulation rapidly reduces it. The RTS
smoother also uses those later measurements to reconstruct the dropout.

![Bearings-only target tracking with interrupted triangulation](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/bearings-only-tracking.png)

### Black-box INS + GPS fusion

This example treats the INS as an existing navigation system rather than
re-integrating raw accelerometer and gyroscope measurements. The INS provides
a smooth 6-DoF body pose in a drifting local frame `L`, while GPS provides
noisy global antenna positions in frame `G`.

The filter estimates the slowly wandering local-to-global correction

```text
x = [t_GL, psi_GL, v_drift, omega_drift],
```

where `t_GL` is three-dimensional translation, `psi_GL` is global yaw
alignment, and the remaining terms describe their slow drift. Roll and pitch
come directly from the gravity-aligned INS attitude.

For each local INS pose, the fused global pose is

```text
^G T_B = ^G T_L ^L T_B.
```

GPS observes a known antenna lever arm on the body:

```text
p_GPS = t_GL + Rz(psi_GL) (p_LB + R_LB r_antenna) + noise.
```

The synthetic INS is locally smooth but its frame translation and yaw wander
slowly. GPS arrives at 2 Hz with 0.6 m standard deviation and is completely
absent from 18--30 seconds. A fixed initial INS alignment therefore drifts
badly in global coordinates. Filtering continuously estimates the frame
wander; during the outage its uncertainty grows and the estimate follows the
learned drift dynamics. RTS smoothing then uses GPS measurements after the
outage to reconstruct a substantially better trajectory through the missing
interval.

The default deterministic example gives roughly 17 m position RMSE if the
initial INS alignment is never corrected, about 0.8 m with online filtering,
and about 0.3 m after RTS smoothing. Through the 12-second GPS outage, the
filtered trajectory is about 1.3 m RMSE and the smoothed trajectory about
0.46 m RMSE.

![Black-box INS and GPS fusion](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/ins-gps-fusion.png)

### TDOA emitter localization and clock calibration

Five radio receivers localize a moving emitter without knowing when any pulse
was transmitted. Subtracting the reference receiver's arrival time cancels
the unknown emission time:

```text
Δtᵢ₁ = (‖p - sᵢ‖ - ‖p - s₁‖) / c + bᵢ.
```

Here `sᵢ` is receiver position, `c` is the speed of light, and `bᵢ` is the
unknown clock offset of receiver `i` relative to receiver 1. The augmented
state

```text
x = [pₓ, pᵧ, vₓ, vᵧ, b₂, b₃, b₄, b₅]
```

therefore estimates the trajectory and calibrates four clocks at once. Each
TDOA constrains position to a hyperbola whose foci are the corresponding
receiver pair. Their changing intersections, combined with the motion model,
separate position from fixed clock bias.

All TDOAs share the same noisy reference timestamp, so their measurement noise
is correlated. The example constructs the complete covariance matrix rather
than treating the four differences as independent. Both Jacobian and unscented
filtering paths are implemented and tested.

![TDOA emitter localization with receiver clock calibration](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/tdoa-emitter-localization.png)

### Ballistic radar tracking and drag inference

This example tracks a ballistic object in a two-dimensional exponential
atmosphere while jointly estimating an unknown effective drag coefficient. The
augmented state is

```text
x = [rₓ, h, vₓ, vₕ, log(c_d)].
```

Using log drag keeps the inferred coefficient positive. Gravity and
aerodynamic deceleration follow

```text
ρ(h) / ρ₀ = exp(-h / H)
v̇ = [0, -g] - c_d exp(-h / H) ‖v‖ v.
```

A ground radar observes noisy slant range and elevation, but neither velocity
nor drag directly:

```text
z = [√(rₓ² + h²), atan2(h, rₓ)] + noise.
```

The coefficient becomes observable through accumulated curvature and
deceleration. RTS smoothing uses the complete radar arc to improve the early
trajectory and parameter estimate.

![Ballistic radar tracking with drag-coefficient inference](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/ballistic-drag-estimation.png)

### Orbit determination from rotating ground stations

The orbit example propagates a planar low-Earth satellite under two-body
gravity in an inertial frame:

```text
x = [r, v]
ṙ = v
v̇ = -μ r / ‖r‖³.
```

Eight equatorial stations rotate with Earth. A station contributes an
observation only while the satellite is above its local horizon. Each visible
station measures nonlinear range and range rate:

```text
ρ  = ‖r - sᵢ(t)‖
ρ̇ = (r - sᵢ(t)) · (v - ṡᵢ(t)) / ρ.
```

The unscented filter recovers position and velocity from the changing station
geometry, while the RTS pass reconstructs a consistent orbit over the full
90-minute tracking arc.

![Orbit determination from rotating ground stations](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/orbit-determination.png)

### Heading-coupled planar tracking

This example models a nonholonomic vehicle with state

```text
x = [pₓ, pᵧ, ψ, v, κ],
```

where `ψ` is its unwrapped world heading, `v` is signed longitudinal speed,
and `κ` is signed curvature. Heading rate is coupled to motion:

```text
ψ̇ = v κ.
```

The vehicle therefore cannot turn in place: both translation and heading rate
vanish when `v = 0`. Negative speed represents reversing and changes the sign
of the heading rate for fixed curvature. The discrete midpoint transition is

```text
α = ψ + ½ dt v κ

pₓ' = pₓ + dt v cos(α)
pᵧ' = pᵧ + dt v sin(α)
ψ'  = ψ + dt v κ.
```

The filter asynchronously fuses noisy 5 Hz global position fixes with noisy
20 Hz body-frame angular-rate measurements:

```text
z_position = [pₓ, pᵧ] + noise
z_gyro     = v κ + noise.
```

The deterministic trajectory drives forward, stops from 8–10 seconds, and
then reverses. The default example uses unscented propagation; analytic
Jacobians are also supplied and tested.

![Heading-coupled tracking from global position and local angular-rate measurements](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/heading-coupled-tracking.png)

### Battery state-of-charge and resistance estimation

This example uses a first-order Thevenin equivalent circuit to infer a
lithium-ion cell's state of charge and ohmic internal resistance. The state is

```text
x = [q, vₚ, log(R₀), I],
```

where `q` is fractional charge, `vₚ` is the voltage across a polarization RC
branch, and `I` is positive during discharge. Resistance is represented in log
space so its estimate remains positive. For cell capacity `Q`, polarization
resistance `R₁`, and time constant `τ`, the discrete transition is

```text
q'  = q - dt I / (3600 Q)
vₚ' = exp(-dt/τ) vₚ + R₁ (1 - exp(-dt/τ)) I.
```

The available sensors measure noisy terminal voltage and noisy load current:

```text
V_terminal = OCV(q) - vₚ - R₀ I
z = [V_terminal, I] + noise.
```

The nonlinear open-circuit-voltage curve makes charge observable, while the
repeating discharge, pulse-load, rest, and regenerative portions of the drive
cycle excite the instantaneous voltage drop needed to identify `R₀`. The
filter starts with deliberately biased charge and resistance estimates. RTS
smoothing then uses the entire one-hour experiment to reconstruct the early
charge state and resistance. The default uses unscented propagation, and
analytic Jacobians are included and tested for extended filtering and
smoothing as well.

![Battery state-of-charge and internal-resistance estimation](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/battery-state-of-charge.png)

### Known-correspondence landmark SLAM

This example jointly estimates a planar robot trajectory and the coordinates
of eight initially uncertain landmarks. Each observation carries a known
landmark identity, so the state has fixed dimension:

```text
x = [pₓ, pᵧ, ψ, v, ω, ℓ₁ₓ, ℓ₁ᵧ, …, ℓ₈ₓ, ℓ₈ᵧ].
```

The robot follows a heading-coupled coordinated-motion model. Wheel odometry
measures forward speed and yaw rate, while visible landmarks provide noisy
range and bearing. As in the bearings-only example, bearing is represented by
a local-frame unit vector to avoid a discontinuous residual at `±π`:

```text
ρᵢ = ‖ℓᵢ - p‖
uᵢ = R(-ψ) (ℓᵢ - p) / ρᵢ.
```

SLAM determines geometry only up to an arbitrary global translation and
rotation. The example therefore uses a tight prior on the initial robot pose
to define the world frame, while landmark priors begin more than a metre from
their true positions. The robot completes a loop through the map, repeatedly
changing which landmarks fall inside the sensor range and field of view.
Reobserving early landmarks closes the loop and sharpens both the trajectory
and map.

The default run uses analytic Jacobians, as in classical EKF-SLAM. Unscented
filtering and smoothing are also supported and covered by the tests.

![Known-correspondence SLAM with joint trajectory and landmark-map reconstruction](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/known-correspondence-slam.png)

### Constant-acceleration tracking

The state contains position, velocity, and acceleration:

```text
x = [p, v, a]
```

Its transition is

```text
F = [[I, dt I, ½ dt² I],
     [0,    I,     dt I],
     [0,    0,        I]].
```

Process noise uses the corresponding discrete white-jerk covariance. As in
the velocity examples, noisy position is the only observed state component.

| 1D | 2D | 3D |
| --- | --- | --- |
| ![Constant-acceleration tracking in one dimension](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-acceleration-1d.png) | ![Constant-acceleration tracking in two dimensions](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-acceleration-2d.png) | ![Constant-acceleration tracking in three dimensions](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/constant-acceleration-3d.png) |

The current transition-model API stores one fixed process covariance. These
examples therefore use fixed-cadence synchronous observations and construct
`Q` for that cadence.

### Rigid-body state and dynamics

The rigid-body state is

```text
x = [p, v, ϕ, ω],
```

where `p` and `v` are world-frame center position and velocity, `ω` is
body-frame angular velocity, and `ϕ ∈ R³` is a rotation bivector/vector. Its
equivalent representations are

```text
R = Exp([ϕ]×)
q = Exp(½ [0, ϕ])
ϕ = 2 vec(Log(q)).
```

Translation follows constant velocity. Rotation follows the torque-free Euler
equation for a body with inertia matrix `I`:

```text
I ω̇ + ω × (I ω) = 0.
```

Angular velocity is integrated with RK4. Attitude is advanced by composition
on SO(3):

```text
R_next = R Exp([dt ω_mid]×).
```

The state is then returned to bivector coordinates with `Log(R_next)`. The
examples remain inside one principal-log chart and do not cross the `π` branch
boundary. BayesFilter does not yet provide manifold-aware Gaussian means or
retractions, so this chart restriction is intentional.

### World-space point observations

The body carries four labeled canonical points:

```text
c₀ = 0,  c₁ = 0.75 e₁,  c₂ = 0.75 e₂,  c₃ = 0.75 e₃.
```

Their predicted world locations are

```text
hᵢ(x) = p + R(ϕ)cᵢ.
```

The origin marker observes translation directly. Subtracting it from the
other marker positions recovers scaled columns of the rotation matrix. The
filter residual therefore lives in ordinary Cartesian point space; it never
subtracts quaternions or rotation vectors directly.

#### Known fixed inertia

The known-inertia example uses `diag(1.0, 1.4, 1.8)` and the unscented filter
and smoother. Its figure shows position tracking, geodesic attitude error,
body angular velocity, and numerical conservation of torque-free invariants.

![Rigid-body tracking with a known inertia matrix](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/rigid-body-known-inertia.png)

#### Inferring inertia ratios in the state

Torque-free angular motion is unchanged if every principal moment is scaled
by the same constant. Absolute inertia scale is therefore unobservable from
these measurements. The augmented example fixes `I₁ = 1` and estimates the
two identifiable log ratios

```text
η₂ = log(I₂ / I₁),  η₃ = log(I₃ / I₁).
```

Exponentiation keeps the inferred ratios positive. The parameters have tiny
process noise and are inferred indirectly from the observed world-space point
trajectories—there are no direct angular-velocity or inertia measurements.
The default run uses 100 point-observation sets over four seconds, making both
the ratio convergence and posterior-uncertainty contraction visible.

![Rigid-body inertia-ratio inference](https://raw.githubusercontent.com/hugohadfield/bayesfilter/main/examples/figures/rigid-body-inertia-estimation.png)

To identify absolute inertia scale, an example would need a known applied
torque or another measurement that supplies scale information.

## Tests

The suite contains analytic linear-Gaussian regression tests and integration
tests covering filtering, smoothing, irregular observation timing, batched
observations, and both propagation modes.

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Pytest measures branch coverage for the `bayesfilter` package and enforces the
minimum configured in `pytest.ini`.

## Package layout

- `bayesfilter/distributions.py`: Gaussian distributions and sigma points.
- `bayesfilter/model.py`: State transition models.
- `bayesfilter/observation.py`: Observation models.
- `bayesfilter/filtering.py`: Bayesian prediction and update loops.
- `bayesfilter/smoothing.py`: RTS smoothing.
- `bayesfilter/unscented.py`: Unscented-transform utilities.
- `examples/`: Runnable tracking and rigid-body examples and their figures.
- `tests/`: Unit and integration tests.

## License

BayesFilter is distributed under the MIT License. See `LICENSE`.

# BayesFilter examples

These examples generate deterministic synthetic observations, run a Bayesian
filter and RTS smoother, print quantitative errors, and save a Matplotlib
figure under `examples/figures/`.

Install BayesFilter from the repository root, then install Matplotlib only for
the examples:

```bash
python -m pip install -e .
python -m pip install -r examples/requirements.txt
```

Matplotlib is not a BayesFilter runtime dependency.

## Running the examples

| Model | State | Command |
| --- | --- | --- |
| Constant velocity, 1D | `[p, v]` | `python -m examples.constant_velocity_1d` |
| Constant velocity, 2D | `[pₓ, pᵧ, vₓ, vᵧ]` | `python -m examples.constant_velocity_2d` |
| Constant velocity, 3D | `[p, v]`, with three-vectors | `python -m examples.constant_velocity_3d` |
| Constant acceleration, 1D | `[p, v, a]` | `python -m examples.constant_acceleration_1d` |
| Constant acceleration, 2D | `[p, v, a]`, with two-vectors | `python -m examples.constant_acceleration_2d` |
| Constant acceleration, 3D | `[p, v, a]`, with three-vectors | `python -m examples.constant_acceleration_3d` |
| Rigid body, known inertia | `[p, v, ϕ, ω]` | `python -m examples.rigid_body_known_inertia` |
| Rigid body, inferred inertia | `[p, v, ϕ, ω, η₂, η₃]` | `python -m examples.rigid_body_inertia_estimation` |

Regenerate every figure with:

```bash
python -m examples.generate_figures
```

All examples use a fixed random seed. Their simulation functions can also be
imported without importing Matplotlib.

## Constant-velocity tracking

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
| ![Constant-velocity tracking in one dimension](figures/constant-velocity-1d.png) | ![Constant-velocity tracking in two dimensions](figures/constant-velocity-2d.png) | ![Constant-velocity tracking in three dimensions](figures/constant-velocity-3d.png) |

## Constant-acceleration tracking

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
| ![Constant-acceleration tracking in one dimension](figures/constant-acceleration-1d.png) | ![Constant-acceleration tracking in two dimensions](figures/constant-acceleration-2d.png) | ![Constant-acceleration tracking in three dimensions](figures/constant-acceleration-3d.png) |

The current transition-model API stores one fixed process covariance. These
examples therefore use fixed-cadence synchronous observations and construct
`Q` for that cadence.

## Rigid-body state and dynamics

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
examples remain inside one principal-log chart and do not cross the `π`
branch boundary. BayesFilter does not yet provide manifold-aware Gaussian
means or retractions, so this chart restriction is intentional.

## World-space point observations

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

### Known fixed inertia

The known-inertia example uses `diag(1.0, 1.4, 1.8)` and the unscented filter
and smoother. Its figure shows position tracking, geodesic attitude error,
body angular velocity, and numerical conservation of torque-free invariants.

![Rigid-body tracking with a known inertia matrix](figures/rigid-body-known-inertia.png)

### Inferring inertia ratios in the state

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

![Rigid-body inertia-ratio inference](figures/rigid-body-inertia-estimation.png)

To identify absolute inertia scale, an example would need a known applied
torque or another measurement that supplies scale information.

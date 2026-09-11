# Changelog

Notable changes to BayesFilter are recorded here.

## 0.1.0 - 2026-09-10

### Fixed

- Correct the Jacobian-propagated state-to-prediction cross-covariance used by
  the Rauch-Tung-Striebel smoother.

### Added

- Allow filtering and RTS smoothing to select Jacobian or unscented
  propagation explicitly.
- Return the filter time grid and cover fixed-rate, irregular, synchronous,
  and batched observation timing.
- Add analytic regression and integration tests for filtering, smoothing,
  cross-covariance propagation, and unscented transforms.
- Add deterministic constant-velocity and constant-acceleration tracking
  examples in one, two, and three dimensions.
- Add planar trajectory, heading-coupled, and bearings-only tracking examples.
- Add three-dimensional rigid-body examples with world-space point
  observations, known inertia, and inferred principal-inertia ratios.
- Add reproducible Matplotlib figures and consolidated example documentation.

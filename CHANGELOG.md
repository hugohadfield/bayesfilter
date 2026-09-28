# Changelog

Notable changes to BayesFilter are recorded here.

## Unreleased

### Added

- Add sensor-fusion examples for phone gyrocompassing, raw-IMU gravity,
  radar/camera tracking, black-box INS/GPS alignment, and boat navigation with
  current and compass-bias inference.
- Add calibration examples for wrist-camera hand-eye transforms, camera/gyro
  rotation and timing, GNSS and IMU lever arms, vehicle wheel radius, and
  speedometer bias.
- Add physical-system examples for pendulum, bouncing-ball, tank-sloshing,
  active-suspension, building-thermal, vehicle coast-down, and wheel-slip
  parameter inference.
- Add examples for binary-packet channel tracking, quantization-noise
  covariance, Doppler-only localization, angles-only asteroid orbits,
  spacecraft-flyby mass inference, and position-only aircraft bank inference.
- Add a complete reproducible figure gallery and README catalog for every
  runnable application example.
- Add a sun-aided island-navigation example fusing sparse noisy speed-log,
  human landmark bearing/range, and Sun-vector observations while estimating
  heading, position, turn rate, and a steady ocean current.
- Add lithium-ion battery state-of-charge and internal-resistance estimation
  from noisy terminal-voltage and current measurements.
- Add known-correspondence landmark SLAM with noisy odometry, range/bearing
  observations, loop closure, and joint trajectory/map smoothing.
- Add moving-emitter TDOA localization with hyperbolic measurement geometry,
  correlated timing noise, and joint receiver clock-bias inference.
- Add ballistic radar tracking with log-space drag-coefficient inference.
- Add low-Earth orbit determination from rotating, visibility-limited ground
  stations using range and range-rate measurements.

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

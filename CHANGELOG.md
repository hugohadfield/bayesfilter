# Changelog

Notable changes to BayesFilter are recorded here.

## Unreleased

## 0.2.0 - 2026-09-28

### Added

- Add 30 deterministic end-to-end examples across nonlinear tracking and
  localization: ballistic radar tracking with drag inference, moving-emitter
  TDOA localization with receiver clock calibration, radar/camera fusion, and
  Doppler-only acoustic localization.
- Add navigation and mapping examples for local/global vehicle fusion,
  black-box INS/GPS alignment, Sun-aided island navigation, boat-current and
  compass-bias inference, known-correspondence landmark SLAM, and aircraft-bank
  inference from position tracks.
- Add space-estimation examples for low-Earth orbit determination from rotating
  ground stations, angles-only asteroid orbit determination, and planet-mass
  inference from a spacecraft flyby.
- Add inertial and calibration examples for raw-IMU gravity estimation, phone
  gyrocompassing, wrist-camera hand-eye calibration, camera/gyro spatial and
  temporal calibration, GNSS and IMU lever arms, and vehicle wheel-radius and
  speedometer calibration.
- Add physical-parameter examples for battery state of charge and resistance,
  house thermal properties, pendulum length and damping, bouncing-ball drag and
  restitution, tank fill level from sloshing, active-suspension road profile,
  vehicle coast-down parameters, and longitudinal wheel slip.
- Add focused measurement-model examples for quantization-noise covariance and
  continuous wireless link-margin inference from binary packet outcomes.
- Add integration tests for every new example, expanding the suite to cover
  asynchronous sensing, dropouts, nonlinear observability, calibration,
  parameter inference, filtering, and RTS smoothing.
- Add reproducible Matplotlib generation and a 42-figure gallery covering every
  runnable example without adding Matplotlib as a runtime dependency.

### Changed

- Reorganize the README example directory and walkthroughs into foundations,
  nonlinear tracking, navigation, orbital estimation, rigid-body estimation,
  calibration, physical parameter inference, and measurement-model lessons.

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

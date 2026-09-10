"""Bearings-only tracking from asynchronous fixed direction sensors."""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS

from examples.linear_tracking import (
    constant_velocity_matrix,
    white_acceleration_covariance,
)


SENSOR_POSITIONS = np.array([[-6.0, -4.0], [6.0, -4.0]])
SECOND_SENSOR_DROPOUT = (5.5, 11.0)


def observe_bearing_unit_vector(state, sensor_position):
    """Return the world-frame unit vector from a sensor to the target."""
    displacement = np.asarray(state)[:2] - np.asarray(sensor_position)
    return displacement / np.linalg.norm(displacement)


def bearing_unit_vector_jacobian(state, sensor_position):
    """Jacobian of a unit-vector bearing observation."""
    displacement = np.asarray(state)[:2] - np.asarray(sensor_position)
    distance = np.linalg.norm(displacement)
    direction = displacement / distance
    jacobian = np.zeros((2, 4))
    jacobian[:, :2] = (np.eye(2) - np.outer(direction, direction)) / distance
    return jacobian


def simulate_bearings_only_truth(times):
    """Generate a deterministic curved target trajectory and velocity."""
    times = np.asarray(times)
    phase = -1.1 + 0.20 * times
    position = np.column_stack((4.0 * np.cos(phase), 3.0 * np.sin(phase)))
    velocity = np.column_stack((-0.8 * np.sin(phase), 0.6 * np.cos(phase)))
    return np.column_stack((position, velocity))


def _bearing_measurement(truth_state, sensor_position, bearing_std_rad, rng):
    true_direction = observe_bearing_unit_vector(truth_state, sensor_position)
    true_angle = np.arctan2(true_direction[1], true_direction[0])
    measured_angle = true_angle + rng.normal(0.0, bearing_std_rad)
    return np.array([np.cos(measured_angle), np.sin(measured_angle)])


def _make_bearing_events(
    truth,
    times,
    rng,
    bearing_std_rad,
    first_sensor_interval_steps,
    second_sensor_interval_steps,
):
    filter_indices = np.arange(1, len(times))
    first_indices = filter_indices[
        (filter_indices - 1) % first_sensor_interval_steps == 0
    ]
    if first_indices[-1] != filter_indices[-1]:
        first_indices = np.append(first_indices, filter_indices[-1])
    second_indices = filter_indices[
        (filter_indices - 1) % second_sensor_interval_steps == 0
    ]
    second_times = times[second_indices]
    outside_dropout = (second_times < SECOND_SENSOR_DROPOUT[0]) | (
        second_times > SECOND_SENSOR_DROPOUT[1]
    )
    second_indices = second_indices[outside_dropout]

    event_records = []
    sensor_measurements = []
    sensor_times = []
    for sensor_index, indices in enumerate((first_indices, second_indices)):
        sensor_position = SENSOR_POSITIONS[sensor_index]

        def observation_func(state, position=sensor_position):
            return observe_bearing_unit_vector(state, position)

        def jacobian_func(state, position=sensor_position):
            return bearing_unit_vector_jacobian(state, position)

        measurements = np.array(
            [
                _bearing_measurement(
                    truth[index],
                    sensor_position,
                    bearing_std_rad,
                    rng,
                )
                for index in indices
            ]
        )
        measurement_times = times[indices]
        sensor_measurements.append(measurements)
        sensor_times.append(measurement_times)
        for time_s, measurement in zip(measurement_times, measurements):
            event_records.append(
                (
                    time_s,
                    sensor_index,
                    Observation(
                        measurement,
                        bearing_std_rad**2 * np.eye(2),
                        observation_func,
                        jacobian_func,
                    ),
                )
            )
    event_records.sort(key=lambda record: (record[0], record[1]))
    return {
        "observations": [record[2] for record in event_records],
        "event_times": np.array([record[0] for record in event_records]),
        "sensor_times": sensor_times,
        "sensor_measurements": sensor_measurements,
    }


def run_bearings_only_tracking(
    seed=501,
    duration_s=16.0,
    filter_rate_hz=20.0,
    first_sensor_rate_hz=10.0,
    second_sensor_rate_hz=2.0,
    use_jacobian=False,
):
    """Run bearings-only filtering and RTS smoothing with a sensor dropout."""
    rng = np.random.default_rng(seed)
    delta_t_s = 1.0 / filter_rate_hz
    times = np.arange(round(duration_s * filter_rate_hz) + 1) * delta_t_s
    truth = simulate_bearings_only_truth(times)

    first_interval = round(filter_rate_hz / first_sensor_rate_hz)
    second_interval = round(filter_rate_hz / second_sensor_rate_hz)
    if not np.isclose(first_interval * first_sensor_rate_hz, filter_rate_hz):
        raise ValueError("first_sensor_rate_hz must divide filter_rate_hz")
    if not np.isclose(second_interval * second_sensor_rate_hz, filter_rate_hz):
        raise ValueError("second_sensor_rate_hz must divide filter_rate_hz")

    bearing_std_rad = np.deg2rad(1.5)
    sensor_data = _make_bearing_events(
        truth,
        times,
        rng,
        bearing_std_rad,
        first_interval,
        second_interval,
    )

    def transition(state, step_s):
        return constant_velocity_matrix(2, step_s) @ state

    def transition_jacobian(_state, step_s):
        return constant_velocity_matrix(2, step_s)

    transition_model = StateTransitionModel(
        transition,
        white_acceleration_covariance(2, delta_t_s, spectral_density=0.12),
        transition_jacobian,
    )
    bayes_filter = BayesianFilter(
        transition_model,
        Gaussian(
            truth[0] + np.array([0.8, -0.6, 0.15, -0.15]),
            np.diag([1.2**2, 1.2**2, 0.45**2, 0.45**2]),
        ),
    )
    filter_states, filter_times = bayes_filter.run(
        sensor_data["observations"],
        sensor_data["event_times"],
        rate_hz=filter_rate_hz,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )
    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])
    aligned_truth = truth[1 : len(filter_times) + 1]

    def position_rmse(states):
        return np.sqrt(
            np.mean(np.sum((states[:, :2] - aligned_truth[:, :2]) ** 2, axis=1))
        )

    return {
        "times": np.asarray(filter_times),
        "truth": aligned_truth,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "sensor_positions": SENSOR_POSITIONS.copy(),
        "sensor_times": sensor_data["sensor_times"],
        "sensor_measurements": sensor_data["sensor_measurements"],
        "event_times": sensor_data["event_times"],
        "dropout_interval": SECOND_SENSOR_DROPOUT,
        "filtered_position_rmse": position_rmse(filtered),
        "smoothed_position_rmse": position_rmse(smoothed),
        "use_jacobian": use_jacobian,
        "bearing_std_rad": bearing_std_rad,
    }

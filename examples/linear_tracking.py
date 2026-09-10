"""Shared constant-velocity and constant-acceleration tracking examples."""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


def constant_velocity_matrix(dimension, delta_t_s):
    identity = np.eye(dimension)
    zeros = np.zeros_like(identity)
    return np.block(
        [
            [identity, delta_t_s * identity],
            [zeros, identity],
        ]
    )


def constant_acceleration_matrix(dimension, delta_t_s):
    identity = np.eye(dimension)
    zeros = np.zeros_like(identity)
    return np.block(
        [
            [
                identity,
                delta_t_s * identity,
                0.5 * delta_t_s**2 * identity,
            ],
            [zeros, identity, delta_t_s * identity],
            [zeros, zeros, identity],
        ]
    )


def white_acceleration_covariance(dimension, delta_t_s, spectral_density):
    time_covariance = np.array(
        [
            [delta_t_s**3 / 3.0, delta_t_s**2 / 2.0],
            [delta_t_s**2 / 2.0, delta_t_s],
        ]
    )
    return spectral_density * np.kron(time_covariance, np.eye(dimension))


def white_jerk_covariance(dimension, delta_t_s, spectral_density):
    time_covariance = np.array(
        [
            [delta_t_s**5 / 20.0, delta_t_s**4 / 8.0, delta_t_s**3 / 6.0],
            [delta_t_s**4 / 8.0, delta_t_s**3 / 3.0, delta_t_s**2 / 2.0],
            [delta_t_s**3 / 6.0, delta_t_s**2 / 2.0, delta_t_s],
        ]
    )
    return spectral_density * np.kron(time_covariance, np.eye(dimension))


def position_observation_matrix(dimension, derivative_order):
    state_dimension = dimension * (derivative_order + 1)
    matrix = np.zeros((dimension, state_dimension))
    matrix[:, :dimension] = np.eye(dimension)
    return matrix


def _truth(dimension, derivative_order, times):
    axis = np.arange(1, dimension + 1, dtype=float)
    phase = 0.55 * axis
    initial_position = 0.35 * (axis - 1.0)
    initial_velocity = 0.45 + 0.18 * axis

    if derivative_order == 1:
        amplitude = 0.22 + 0.05 * axis
        frequency = 0.45 + 0.07 * axis
        angles = times[:, None] * frequency + phase
        position = (
            initial_position
            + times[:, None] * initial_velocity
            + amplitude * np.sin(angles)
            - amplitude * np.sin(phase)
        )
        velocity = initial_velocity + amplitude * frequency * np.cos(angles)
        return np.concatenate((position, velocity), axis=1)

    if derivative_order == 2:
        initial_acceleration = 0.06 * (-1.0) ** axis
        amplitude = 0.05 + 0.012 * axis
        frequency = 0.35 + 0.04 * axis
        angles = times[:, None] * frequency + phase
        position = (
            initial_position
            + times[:, None] * initial_velocity
            + 0.5 * times[:, None] ** 2 * initial_acceleration
            + amplitude * (np.sin(angles) - np.sin(phase))
        )
        velocity = (
            initial_velocity
            + times[:, None] * initial_acceleration
            + amplitude * frequency * np.cos(angles)
        )
        acceleration = initial_acceleration - amplitude * frequency**2 * np.sin(angles)
        return np.concatenate((position, velocity, acceleration), axis=1)

    raise ValueError("derivative_order must be 1 or 2")


def run_linear_tracking(
    dimension,
    derivative_order,
    seed=0,
    num_steps=80,
    delta_t_s=0.15,
    use_jacobian=True,
):
    """Run one deterministic linear tracking and smoothing example."""
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2, or 3")
    if derivative_order not in (1, 2):
        raise ValueError("derivative_order must be 1 or 2")

    rng = np.random.default_rng(seed)
    times = np.arange(num_steps + 1, dtype=float) * delta_t_s
    truth = _truth(dimension, derivative_order, times)

    if derivative_order == 1:
        transition_matrix_func = constant_velocity_matrix
        process_covariance = white_acceleration_covariance(
            dimension,
            delta_t_s,
            spectral_density=0.16,
        )
    else:
        transition_matrix_func = constant_acceleration_matrix
        process_covariance = white_jerk_covariance(
            dimension,
            delta_t_s,
            spectral_density=0.045,
        )

    def transition(state, step_s):
        return transition_matrix_func(dimension, step_s) @ state

    def transition_jacobian(_state, step_s):
        return transition_matrix_func(dimension, step_s)

    observation_matrix = position_observation_matrix(
        dimension,
        derivative_order,
    )

    def observe_position(state):
        return observation_matrix @ state

    def observation_jacobian(_state):
        return observation_matrix

    measurement_std = 0.38
    measurements = truth[1:, :dimension] + rng.normal(
        0.0,
        measurement_std,
        (num_steps, dimension),
    )
    measurement_covariance = measurement_std**2 * np.eye(dimension)
    observations = [
        Observation(
            measurement,
            measurement_covariance,
            observe_position,
            observation_jacobian,
        )
        for measurement in measurements
    ]

    state_dimension = dimension * (derivative_order + 1)
    initial_offset = np.linspace(0.45, -0.25, state_dimension)
    initial_covariance_diagonal = np.concatenate(
        [
            np.full(dimension, 0.8),
            np.full(dimension, 0.45),
            *([np.full(dimension, 0.20)] if derivative_order == 2 else []),
        ]
    )
    model = StateTransitionModel(
        transition,
        process_covariance,
        transition_jacobian,
    )
    bayes_filter = BayesianFilter(
        model,
        Gaussian(
            truth[0] + initial_offset,
            np.diag(initial_covariance_diagonal),
        ),
    )
    filter_states = bayes_filter.run_synchronous(
        observations,
        times,
        use_jacobian=use_jacobian,
    )
    smoother_states = RTS(bayes_filter).apply(
        filter_states,
        times,
        use_jacobian=use_jacobian,
    )

    filtered = np.array([state.mean() for state in filter_states])
    smoothed = np.array([state.mean() for state in smoother_states])
    filtered_covariances = np.array([state.covariance() for state in filter_states])
    smoothed_covariances = np.array([state.covariance() for state in smoother_states])
    filtered_rmse = np.sqrt(
        np.mean((filtered[:, :dimension] - truth[:, :dimension]) ** 2)
    )
    smoothed_rmse = np.sqrt(
        np.mean((smoothed[:, :dimension] - truth[:, :dimension]) ** 2)
    )

    return {
        "dimension": dimension,
        "derivative_order": derivative_order,
        "times": times,
        "truth": truth,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": filtered_covariances,
        "smoothed_covariances": smoothed_covariances,
        "filtered_rmse": filtered_rmse,
        "smoothed_rmse": smoothed_rmse,
        "transition_matrix": transition_matrix_func(dimension, delta_t_s),
        "process_covariance": process_covariance,
        "observation_matrix": observation_matrix,
    }

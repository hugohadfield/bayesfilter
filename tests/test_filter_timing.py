import numpy as np
import pytest

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


def _scalar_filter(
    initial_mean=0.0,
    initial_variance=1.0,
    process_variance=0.2,
    drift_rate=0.0,
):
    def transition(state, delta_t_s):
        return state + drift_rate * delta_t_s

    def transition_jacobian(_state, _delta_t_s):
        return np.eye(1)

    return BayesianFilter(
        StateTransitionModel(
            transition,
            np.array([[process_variance]]),
            transition_jacobian,
        ),
        Gaussian(np.array([initial_mean]), np.array([[initial_variance]])),
    )


def _scalar_observation(value, variance=0.5):
    def observe(state):
        return state.copy()

    def observation_jacobian(_state):
        return np.eye(1)

    return Observation(
        np.array([value]),
        np.array([[variance]]),
        observe,
        observation_jacobian,
    )


def _scalar_update(mean, variance, observation, observation_variance):
    gain = variance / (variance + observation_variance)
    updated_mean = mean + gain * (observation - mean)
    updated_variance = variance - gain * (variance + observation_variance) * gain
    return updated_mean, updated_variance


@pytest.mark.parametrize(
    "use_jacobian",
    [True, False],
    ids=["jacobian", "unscented"],
)
def test_run_synchronous_matches_scalar_kalman_recursion(use_jacobian):
    process_variance = 0.2
    observation_variance = 0.5
    drift_rate = 0.4
    observation_values = [1.0, -0.25, 0.75]
    times = [0.0, 0.2, 0.9, 1.7]
    bayesian_filter = _scalar_filter(
        process_variance=process_variance,
        drift_rate=drift_rate,
    )
    observations = [
        _scalar_observation(value, observation_variance)
        for value in observation_values
    ]

    states = bayesian_filter.run_synchronous(
        observations,
        times,
        use_jacobian=use_jacobian,
    )

    expected = [(0.0, 1.0)]
    mean, variance = expected[0]
    for value, delta_t_s in zip(observation_values, np.diff(times)):
        mean += drift_rate * delta_t_s
        variance += process_variance
        mean, variance = _scalar_update(
            mean,
            variance,
            value,
            observation_variance,
        )
        expected.append((mean, variance))

    assert len(states) == len(times)
    for state, (expected_mean, expected_variance) in zip(states, expected):
        np.testing.assert_allclose(state.mean(), [expected_mean], atol=1e-7)
        np.testing.assert_allclose(
            state.covariance(),
            [[expected_variance]],
            atol=1e-7,
        )


@pytest.mark.parametrize(
    "use_jacobian",
    [True, False],
    ids=["jacobian", "unscented"],
)
def test_run_batches_irregular_observations_at_filter_ticks(use_jacobian):
    process_variance = 0.2
    observation_variance = 0.5
    observation_times = [0.0, 0.04, 0.09, 0.21]
    observation_values = [1.0, 2.0, -0.5, 0.75]
    bayesian_filter = _scalar_filter(process_variance=process_variance)
    observations = [
        _scalar_observation(value, observation_variance)
        for value in observation_values
    ]

    states, filter_times = bayesian_filter.run(
        observations,
        observation_times,
        rate_hz=10.0,
        use_jacobian=use_jacobian,
    )

    expected_times = np.array([0.0, 0.1, 0.2, 0.3])
    expected_batches = [[1.0], [2.0, -0.5], [], [0.75]]
    mean = 0.0
    variance = 1.0
    expected = []
    for batch in expected_batches:
        variance += process_variance
        for value in batch:
            mean, variance = _scalar_update(
                mean,
                variance,
                value,
                observation_variance,
            )
        expected.append((mean, variance))

    np.testing.assert_allclose(filter_times, expected_times)
    assert len(states) == len(expected_batches)
    for state, (expected_mean, expected_variance) in zip(states, expected):
        np.testing.assert_allclose(state.mean(), [expected_mean], atol=1e-7)
        np.testing.assert_allclose(
            state.covariance(),
            [[expected_variance]],
            atol=1e-7,
        )

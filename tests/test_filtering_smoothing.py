import numpy as np
import pytest

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation
from bayesfilter.smoothing import RTS


def _tracking_problem():
    rng = np.random.default_rng(0)

    def transition_func(state, _delta_t_s):
        return np.array([state[0]])

    def transition_jacobian_func(_state, _delta_t_s):
        return np.array([[1.0]])

    def observation_func(state):
        return np.array([np.sin(state[0]), np.cos(state[0])])

    def observation_jacobian_func(state):
        return np.array([[np.cos(state[0])], [-np.sin(state[0])]])

    transition_model = StateTransitionModel(
        transition_func,
        1e-8 * np.eye(1),
        transition_jacobian_func,
    )
    bayesian_filter = BayesianFilter(
        transition_model,
        Gaussian(np.array([0.0]), np.eye(1)),
    )

    true_state = np.array([-0.1])
    noise_std = 0.2
    observation_times = np.linspace(0, 2 * np.pi, 1000)
    observations = [
        Observation(
            observation_func(true_state) + rng.normal(0, noise_std, 2),
            noise_std * np.eye(2),
            observation_func,
            observation_jacobian_func,
        )
        for _ in observation_times
    ]
    return bayesian_filter, observations, observation_times, true_state


@pytest.mark.parametrize(
    "use_jacobian",
    [True, False],
    ids=["jacobian", "unscented"],
)
def test_filter_noisy_sin(use_jacobian):
    bayesian_filter, observations, observation_times, true_state = (
        _tracking_problem()
    )

    filter_states, filter_times = bayesian_filter.run(
        observations,
        observation_times,
        100.0,
        use_jacobian=use_jacobian,
    )

    assert len(filter_states) == len(filter_times)
    np.testing.assert_allclose(bayesian_filter.state.mean(), true_state, atol=1e-2)


@pytest.mark.parametrize(
    "use_jacobian",
    [True, False],
    ids=["jacobian", "unscented"],
)
def test_smoother_noisy_sin_uses_filter_times(use_jacobian):
    bayesian_filter, observations, observation_times, true_state = (
        _tracking_problem()
    )
    filter_states, filter_times = bayesian_filter.run(
        observations,
        observation_times,
        100.0,
        use_jacobian=use_jacobian,
    )

    smoother_states = RTS(bayesian_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )

    assert len(smoother_states) == len(filter_states) == len(filter_times)
    np.testing.assert_allclose(smoother_states[0].mean(), true_state, atol=1e-2)
    np.testing.assert_allclose(smoother_states[-1].mean(), true_state, atol=1e-2)


@pytest.mark.parametrize(
    "use_jacobian",
    [True, False],
    ids=["jacobian", "unscented"],
)
def test_smooth_matches_explicit_filter_then_apply(use_jacobian):
    manual_filter, observations, observation_times, _ = _tracking_problem()
    filter_states, filter_times = manual_filter.run(
        observations,
        observation_times,
        100.0,
        use_jacobian=use_jacobian,
    )
    expected_states = RTS(manual_filter).apply(
        filter_states,
        filter_times,
        use_jacobian=use_jacobian,
    )

    wrapper_filter, observations, observation_times, _ = _tracking_problem()
    actual_states = RTS(wrapper_filter).smooth(
        observations,
        observation_times,
        100.0,
        use_jacobian=use_jacobian,
    )

    assert len(actual_states) == len(expected_states)
    for actual, expected in zip(actual_states, expected_states):
        np.testing.assert_allclose(actual.mean(), expected.mean())
        np.testing.assert_allclose(actual.covariance(), expected.covariance())

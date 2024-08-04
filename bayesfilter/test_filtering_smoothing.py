
import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.observation import Observation
from bayesfilter.model import StateTransitionModel
from bayesfilter.smoothing import RTS
    

def setup_functions():

    def transition_func(x, delta_t_s):
        return np.array([x[0]])
    
    def transition_jacobian_func(x, delta_t_s):
        return np.array([[1.0]])

    def observation_func(x):
        return np.array([np.sin(x[0]), np.cos(x[0])])
    
    def observation_jacobian_func(x):
        return np.array([[np.cos(x[0])], [-np.sin(x[0])]])
    
    return transition_func, transition_jacobian_func, observation_func, observation_jacobian_func


def setup_filter_and_observations():
    # Set the random seed
    rng = np.random.default_rng(0)

    # Set up the functions
    transition_func, transition_jacobian_func, observation_func, observation_jacobian_func = setup_functions()

    # Set up the filter
    transition_model = StateTransitionModel(
        transition_func, 
        1e-8*np.eye(1),
        transition_jacobian_func
    )
    initial_state = Gaussian(np.array([0.0]), np.eye(1))
    filter = BayesianFilter(transition_model, initial_state)

    # Set up the observations
    true_state = np.array([-0.1])
    noise_std = 0.2
    observations = []
    for theta in np.linspace(0, 2*np.pi, 1000):
        observation = observation_func(true_state) + rng.normal(0, noise_std, 2)
        observations.append(Observation(observation, noise_std*np.eye(2), observation_func, observation_jacobian_func))
    return filter, observations, true_state
        

def test_filter_noisy_sin(use_jacobian=True):
    filter, observations, true_state = setup_filter_and_observations()
    
    filter.run(observations, np.linspace(0, 2*np.pi, 1000), 100.0, use_jacobian=use_jacobian)
    np.testing.assert_allclose(filter.state.mean(), true_state, atol=1e-2)


def test_smoother_noisy_sin(enable_debug_plots=False, use_jacobian=True):
    filter, observations, true_state = setup_filter_and_observations()
    
    filter_states, filter_times = filter.run(observations, np.linspace(0, 2*np.pi, 1000), 100.0, use_jacobian=use_jacobian)
    smoother = RTS(filter)
    smoother_states = smoother.apply(filter_states, np.linspace(0, 2*np.pi, 1000), use_jacobian=use_jacobian)
    assert len(smoother_states) == len(filter_states)
    assert len(smoother_states) == len(filter_times)

    if enable_debug_plots:
        import matplotlib.pyplot as plt
        plt.plot(filter_times, [state.mean()[0] for state in filter_states], label=f"{'Extended' if use_jacobian else 'Unscented'} Filter")
        plt.plot(filter_times, [state.mean()[0] for state in smoother_states], label=f"{'Extended' if use_jacobian else 'Unscented'} Smoother")
        plt.plot(filter_times, [true_state[0] for _ in filter_times], label="True value")
        plt.legend()
        plt.show()

    np.testing.assert_allclose(smoother_states[0].mean(), true_state, atol=1e-2)
    np.testing.assert_allclose(smoother_states[-1].mean(), true_state, atol=1e-2)


if __name__ == "__main__":
    test_filter_noisy_sin(True)
    test_filter_noisy_sin(False)
    test_smoother_noisy_sin(False, True)
    test_smoother_noisy_sin(False, False)
    print("All tests passed")

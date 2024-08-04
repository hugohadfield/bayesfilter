from typing import List, Tuple

import numpy as np

try:
    import tqdm
except ImportError:
    tqdm = None

from bayesfilter.distributions import Gaussian
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation

class BayesianFilter:
    def __init__(self, state_transition_model: StateTransitionModel, initial_state: Gaussian):
        """
        Set up the bayesian filter with the state transition model and initial state
        """
        self.transition_model = state_transition_model
        self.state = initial_state

    def predict(self, state: Gaussian, delta_t_s: float, use_jacobian: bool = True) -> Gaussian:
        """
        Predict the next state from a given state
        """
        # Predict the next state
        return self.transition_model.predict(state, delta_t_s, use_jacobian=use_jacobian)
    
    def update(self, observation: Observation, predicted_state: Gaussian, use_jacobian: bool = False) -> Gaussian:
        """
        Condition the state on a observation
        """
        # Predict the observation (with noise)
        predicted_obsurement, cross_covariance = observation.predict_with_cross_covariance(predicted_state, use_jacobian=use_jacobian)

        # Compute the Kalman gain
        kalman_gain = cross_covariance@np.linalg.inv(predicted_obsurement.noise_covariance)

        # Compute the residual
        residual = observation.observation - observation.observation_func(predicted_state.mean())

        # Compute the new state
        new_state = Gaussian(
            mean=predicted_state.mean() + kalman_gain@residual,
            covariance=predicted_state.covariance() - kalman_gain@predicted_obsurement.noise_covariance@kalman_gain.T
        )

        # Update the state
        self.state = new_state
        return new_state

    def set_state(self, state: Gaussian):
        """
        Directly set the filter state
        """
        self.state = state

    def run_synchronous(self, observations: List[Observation], times_s: List[float], use_jacobian=False) -> List[Gaussian]:
        """
        Run the BayesianFilter on a list of observations.
        Assumes that the observations are at a roughly fixed time interval.
        use_jacobian allows you to switch between an extended and unscented prediction and update
        """
        output_states = [self.state]  
        for i, observation in enumerate(observations[:len(times_s)-1]):
            delta_t_s = times_s[i+1] - times_s[i]
            predicted_state = self.predict(self.state, delta_t_s, use_jacobian=use_jacobian)
            self.update(observation, predicted_state, use_jacobian=use_jacobian)
            output_states.append(self.state)
        return output_states

    def run(
        self, 
        observations: List[Observation], 
        times_s: List[float], 
        rate_hz: float, 
        use_jacobian=False
    ) -> Tuple[List[Gaussian], List[float]]:
        """
        Run the bayesian filter on a list of observations at a fixed rate, start at the start of times_s and end at the 
        end of times_s. times_s referres to the observation times. use_jacobian allows you to switch
        between an extended linearisation and an unscented transform for the prediction
        """
        filter_states = []
        filter_times = []
        start_time = times_s[0]
        end_time = times_s[-1]
        current_obs_index = 0
        # Step through the times from start to finish
        # Use tqdm to keep track of progress if available
        print(f"Running BayesianFilter at {rate_hz} Hz with use_jacobian: {use_jacobian}", flush=True)
        delta_t_s = 1.0/rate_hz
        iterator = np.arange(start_time, end_time + delta_t_s, delta_t_s)
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator)
        for current_time in iterator:
            # Gets a list of all observations that have occured between previous time and curent_time
            current_obs = []
            if current_obs_index < len(times_s):
                next_obs_time = times_s[current_obs_index]
                while next_obs_time <= current_time:
                    current_obs.append(observations[current_obs_index])
                    current_obs_index += 1
                    if current_obs_index >= len(times_s):
                        break
                    next_obs_time = times_s[current_obs_index]
            
            # Predict the next state
            predicted_state = self.predict(self.state, delta_t_s, use_jacobian=use_jacobian)

            # Update the state with the observations, if we have any
            for observation in current_obs:
                predicted_state = self.update(observation, predicted_state, use_jacobian=use_jacobian)

            # Set the state
            self.set_state(predicted_state)

            # Append the state
            filter_states.append(self.state)
            filter_times.append(current_time)

        return filter_states, filter_times

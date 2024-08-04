from typing import List, Tuple

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.observation import Observation
from bayesfilter.filtering import BayesianFilter


class RTS:
    """
    General Rauch-Tung-Striebel smoother. 
    Equations from BAYESIAN FILTERING AND SMOOTHING by Simo Sarkka, what an absolute banger of a textbook.
    """
    def __init__(self, filter: BayesianFilter):
        self.filter = filter
        
    def apply(self, filter_states: List[Gaussian], time_list_s: List[float], use_jacobian=False):
        smoother_states = []
        # We start the smoother with the last filter state
        current_covariance = filter_states[-1].covariance()
        current_mean = filter_states[-1].mean()
        current_mean_s = current_mean
        current_covariance_s = current_covariance
        smoother_states.append(Gaussian(current_mean_s, current_covariance_s))
        delta_t_s = time_list_s[-1] - time_list_s[-2]

        # We iterate backwards through the states
        for i in range(len(filter_states)-2, -1, -1):
            # Get the data
            current_state_object = filter_states[i]
            current_mean = current_state_object.mean()
            current_covariance = current_state_object.covariance()
            delta_t_s = time_list_s[i+1] - time_list_s[i]

            # Predict
            pred_state, cross_covariance = self.filter.transition_model.predict_with_cross_covariance(
                current_state_object, delta_t_s, use_jacobian=use_jacobian
            )

            # Calculate the smoother gain
            G_k = cross_covariance@np.linalg.inv(pred_state.covariance())

            # Calculate the new mean and covariance
            smoothed_state = current_mean + G_k@(current_mean_s - pred_state.mean())
            smoothed_covariance = current_covariance + G_k@(current_covariance_s - pred_state.covariance())@G_k.T

            # Now update the current state
            current_mean_s = smoothed_state
            current_covariance_s = smoothed_covariance

            # Append the state
            smoother_states.append(Gaussian(smoothed_state, smoothed_covariance))

        smoother_states.reverse()
        return smoother_states
    
    def smooth(
        self, 
        observations: List[Observation], 
        times_s: List[float], 
        rate_hz: float = 1.0,
        use_jacobian = False,
    ) -> Tuple[List[Gaussian], List[float]]:
        """
        Smooths the observations using the general RTS algorithm
        """
        filter_states, filter_times = self.filter.run(observations, times_s, rate_hz, use_jacobian=use_jacobian)
        return self.apply(filter_states, filter_times, use_jacobian=use_jacobian)


from typing import Callable, Optional

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.utilities import propagate_covariance
from bayesfilter.unscented import (
    propagate_gaussian, propagate_gaussian_cross_cov
)

class Observation:
    """
    Represents an observation model
    """
    def __init__(
            self, 
            observation: np.ndarray, 
            noise_covariance: np.ndarray, 
            observation_func: Callable[[np.ndarray], np.ndarray],
            jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
        ):
        self.observation = observation.copy()
        self.noise_covariance = noise_covariance.copy()
        self.observation_func = observation_func
        self.dimension = len(noise_covariance)
        self.jacobian_func = jacobian_func
    
    def predict_no_noise_jacobian(self, state: Gaussian):
        """
        Predict an observation from a state without adding observation noise
        """
        # Predict observation
        pred_obs = self.observation_func(state.mean())
        # Predict covariance
        jac = self.get_jacobian(state.mean())
        pred_cov = propagate_covariance(
            jacobian=jac,
            covariance=state.covariance()
        )
        return Observation(pred_obs, pred_cov, self.observation_func)
    
    def predict_no_noise(self, state: Gaussian, use_jacobian: bool = False):
        """
        Predict an observation from a state without adding observation noise
        """
        if use_jacobian:
            return self.predict_no_noise_jacobian(state)
        # Unscented transform
        pred_mean, pred_cov = propagate_gaussian(
            state.mean(),
            state.covariance(),
            self.observation_func
        )
        return Observation(pred_mean, pred_cov, self.observation_func)

    def predict(self, state: Gaussian, use_jacobian: bool = False):
        """
        Predict an observation from a state
        """
        observation = self.predict_no_noise(state, use_jacobian)
        observation.noise_covariance += self.noise_covariance
        return observation
    
    def predict_with_cross_covariance(self, state: Gaussian, use_jacobian: bool = False):
        """
        Predict an observation from a state and also return the cross-covariance
        """
        if use_jacobian:
            predicted_obs = self.observation_func(state.mean())
            observation_jacobian = self.get_jacobian(state.mean())
            pred_cov = propagate_covariance(observation_jacobian, state.covariance())
            cross_covariance = state.covariance()@observation_jacobian.T
            return Observation(predicted_obs, pred_cov + self.noise_covariance, self.observation_func), cross_covariance
        
        pred_mean, pred_cov, cross_cov = propagate_gaussian_cross_cov(
            state.mean(),
            state.covariance(),
            self.observation_func
        )
        return Observation(pred_mean, pred_cov + self.noise_covariance, self.observation_func), cross_cov
    
    def get_jacobian(self, state: np.ndarray):
        if self.jacobian_func is None:
            raise ValueError("Observation Jacobian function not provided")
        return self.jacobian_func(state)
    

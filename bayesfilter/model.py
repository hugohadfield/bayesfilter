
from typing import Callable, Optional

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.utilities import propagate_covariance
from bayesfilter.unscented import (
    propagate_gaussian, propagate_gaussian_cross_cov
)


class StateTransitionModel:
    """
    Represents a state transition model
    """
    def __init__(
            self, 
            transition_func: Callable[[np.ndarray, float], np.ndarray],
            transition_noise_covariance: np.ndarray,
            transition_jacobian_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
        ):
        self.transition_func = transition_func
        self.dimension = len(transition_noise_covariance)
        self.transition_noise_covariance = transition_noise_covariance
        self.transition_jacobian_func = transition_jacobian_func

    def predict_no_noise_jacobian(self, state: Gaussian, delta_t_s: float):
        """
        Predict the next state without adding transition noise, uses the Jacobian of the
        state transition function
        """
        new_state = Gaussian(
            mean=self.transition_func(state.mean(), delta_t_s),
            covariance=propagate_covariance(
                jacobian=self.get_jacobian(state, delta_t_s),
                covariance=state.covariance()
            )
        )
        return new_state
    
    def predict_no_noise(self, state: Gaussian, delta_t_s: float, use_jacobian: bool = False):
        """
        Predict the next state without adding transition noise, uses the unscented transform
        """
        if use_jacobian:
            return self.predict_no_noise_jacobian(state, delta_t_s)
        # Unscented transform
        new_mean, new_covariance = propagate_gaussian(
            state.mean(),
            state.covariance(),
            lambda x: self.transition_func(x, delta_t_s)
        )
        return Gaussian(new_mean, new_covariance)
    
    def predict(self, state: Gaussian, delta_t_s: float, use_jacobian: bool = False):
        """
        Predict the next state
        """
        new_state = self.predict_no_noise(state, delta_t_s, use_jacobian)
        return Gaussian(new_state.mean(), new_state.covariance() + self.transition_noise_covariance)
    
    def predict_with_cross_covariance(self, state: Gaussian, delta_t_s: float, use_jacobian=False):
        """
        Predict the next state and return the cross-covariance too
        """
        if use_jacobian:
            pred_mean = self.transition_func(state.mean(), delta_t_s)
            transition_func_jac = self.get_jacobian(state, delta_t_s)
            pred_cov = propagate_covariance(transition_func_jac, state.covariance())
            cross_cov = pred_cov@transition_func_jac.T
            return Gaussian(pred_mean, pred_cov + self.transition_noise_covariance), cross_cov
        pred_mean, pred_cov, cross_cov = propagate_gaussian_cross_cov(
            state.mean(),
            state.covariance(),
            lambda x: self.transition_func(x, delta_t_s)
        )
        return Gaussian(pred_mean, pred_cov + self.transition_noise_covariance), cross_cov

    def get_jacobian(self, state: Gaussian, delta_t_s: float):
        if self.transition_jacobian_func is None:
            raise ValueError("Transition Jacobian function not provided")
        return self.transition_jacobian_func(state.mean(), delta_t_s)
    
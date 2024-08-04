
from typing import List

import numpy as np

DEFAULT_ALPHA = 1e-3
DEFAULT_BETA = 2.0


def compute_lambda(n: int, alpha: float = DEFAULT_ALPHA) -> int:
    """
    Compute the lambda parameter for the unscented transform.
    """
    kappa = 3 - n
    l = alpha*alpha*(n + kappa) - n
    return l


def compute_covariance_mean_weight(
        dimension: int, 
        alpha: float = DEFAULT_ALPHA, 
        beta: float = DEFAULT_BETA
    ) -> np.ndarray:
    """
    Compute the covariance weight for the sigma points representing the mean.
    """
    lamb = compute_lambda(dimension)
    return lamb / (dimension + lamb) + (1 - alpha*alpha + beta)


class Distribution:
    """
    Represents a generic probability distribution.
    """
    def dimension(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError
    
    def covariance(self):
        raise NotImplementedError
    
    def compute_sigma_points(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def from_samples(self, samples: List[np.ndarray]):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
    

class Gaussian(Distribution):
    """
    Represents a Gaussian distribution.
    """
    def __init__(self, mean, covariance, rng=None):
        self._mean = mean
        self._covariance = covariance
        self._rng = np.random.default_rng(rng)

    @property
    def rng(self):
        return self._rng

    def mean(self) -> np.ndarray:
        """
        Return the mean of the Gaussian distribution.
        """
        return self._mean
    
    def covariance(self) -> np.ndarray:
        """
        Return the covariance of the Gaussian distribution.
        """
        return self._covariance
    
    def dimension(self) -> int:
        """
        Return the dimension of the Gaussian distribution.
        """
        return len(self.mean())

    def sqrt_covariance(self) -> np.ndarray:
        """
        Return the square root of the covariance matrix.
        """
        if self.dimension() == 1:
            return np.array([[np.sqrt(self.covariance()[0, 0])]])
        return np.linalg.cholesky(self.covariance())
    
    def compute_sigma_points(self) -> np.ndarray:
        """
        Compute the sigma points for the Gaussian distribution.
        """
        mean = self.mean()
        sqrt_covariance = self.sqrt_covariance()
        n = self.dimension()
        lamb = compute_lambda(n)
        factor = np.sqrt(n + lamb)
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0, :] = mean
        counter = 1
        for i in range(n): # 2n + 1 sigma points
            sigma_points[counter, :] = mean + factor*sqrt_covariance[:, i]
            counter += 1
            sigma_points[counter, :] = mean - factor*sqrt_covariance[:, i]
            counter += 1
        return sigma_points
    
    def compute_weights(self) -> np.ndarray:
        """
        Compute the weights for the sigma points.
        """
        n = self.dimension()
        lamb = compute_lambda(n)
        weights = np.zeros(2*n + 1)
        # The first weight is for the mean
        weights[0] = lamb / (n + lamb) 
        for i in range(2*n):
            weights[i + 1] = 1.0 / (2.0*(n + lamb))
        return weights
    
    def from_sigma_points(self, sigma_points: np.ndarray, weights: np.ndarray):
        """
        Create a Gaussian distribution from the given sigma points.
        Equations coming from https://arxiv.org/pdf/2104.01958
        """
        new_n = sigma_points.shape[1]
        new_mean = np.zeros(new_n)
        for i in range(sigma_points.shape[0]):
            new_mean += weights[i] * sigma_points[i, :]
        new_covariance = np.zeros((new_n, new_n))
        for i in range(sigma_points.shape[0]):
            diff = sigma_points[i, :] - new_mean
            if i == 0:
                new_covariance += compute_covariance_mean_weight(self.dimension()) * np.outer(diff, diff)
            else:
                new_covariance += weights[i] * np.outer(diff, diff)
        return Gaussian(new_mean, new_covariance, self.rng)

    def sample(self) -> np.ndarray:
        """
        Sample from the Gaussian distribution.
        """
        return self.rng.multivariate_normal(self.mean(), self.covariance())
    
    def from_samples(self, samples: List[np.ndarray]):
        """
        Create a Gaussian distribution from the given samples.
        """
        sample_array = np.array(samples, dtype=np.float64)
        new_mean = np.mean(sample_array, axis=0)
        new_covariance = np.cov(sample_array, rowvar=False)
        return Gaussian(new_mean, new_covariance, self.rng)

    def __repr__(self):
        return f'Gaussian(mean={self.mean()}, covariance={self.covariance()})'
    
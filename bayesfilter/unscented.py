from typing import Callable, Tuple

import numpy as np

from bayesfilter.distributions import Distribution, Gaussian, compute_covariance_mean_weight


def unscented_transform(
    distribution: Distribution,
    non_linear_function: Callable,
) -> Distribution:
    """
    Propagate the distribution through the non-linear function using the unscented transform.
    """
    sigma_points = distribution.compute_sigma_points()
    transformed_sigma_points = np.array([non_linear_function(sp) for sp in sigma_points])
    weights = distribution.compute_weights()
    return distribution.from_sigma_points(transformed_sigma_points, weights)

def unscented_transform_cross_cov(
    distribution: Distribution,
    non_linear_function: Callable,
) -> Tuple[Distribution, np.ndarray]:
    """
    Propagate the distribution through the non-linear function using the unscented transform.
    Return the cross-covariance between the mean and the transformed sigma points.
    """
    sigma_points = distribution.compute_sigma_points()
    transformed_sigma_points = np.array([non_linear_function(sp) for sp in sigma_points])
    weights = distribution.compute_weights()
    transformed_distribution = distribution.from_sigma_points(transformed_sigma_points, weights)
    # Compute the cross-covariance
    cross_covariance = np.zeros((distribution.dimension(), transformed_sigma_points.shape[1]))
    for i in range(sigma_points.shape[0]):
        diff_x = sigma_points[i, :] - distribution.mean()
        diff_y = transformed_sigma_points[i, :] - transformed_distribution.mean()
        if i == 0:
            cross_covariance += compute_covariance_mean_weight(distribution.dimension()) * np.outer(diff_x, diff_y)
        else:
            cross_covariance += weights[i] * np.outer(diff_x, diff_y)
    return transformed_distribution, cross_covariance


def propagate_samples(
    distribution: Distribution,
    non_linear_function: Callable,
    num_samples: int,
) -> Distribution:
    """
    Sample a distribution and propagate the samples through the non-linear function.
    Reestimate the distribution from the transformed samples.
    """
    samples = [distribution.sample() for _ in range(num_samples)]
    transformed_samples = [non_linear_function(sample) for sample in samples]
    return distribution.from_samples(transformed_samples)


def propagate_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    non_linear_function: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a Gaussian through a function with the unscented transform.
    """
    gaussian = Gaussian(mean, covariance)
    new_gaussian = unscented_transform(gaussian, non_linear_function)
    return new_gaussian.mean(), new_gaussian.covariance()


def propagate_gaussian_cross_cov(
    mean: np.ndarray,
    covariance: np.ndarray,
    non_linear_function: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate a Gaussian through a function with the unscented transform.
    Return the cross-covariance between the mean and the transformed sigma points.
    """
    gaussian = Gaussian(mean, covariance)
    new_gaussian, cross_covariance = unscented_transform_cross_cov(gaussian, non_linear_function)
    return new_gaussian.mean(), new_gaussian.covariance(), cross_covariance

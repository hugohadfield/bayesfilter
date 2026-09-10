import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.unscented import unscented_transform


def test_unscented_transform_linear_func():
    mean = np.array([1.0, 2.0])
    covariance = np.eye(2)
    gaussian = Gaussian(mean, covariance)

    def linear_function(state):
        return np.array([2.0 * state[0], 3.0 * state[1]])

    jacobian = np.array([[2.0, 0.0], [0.0, 3.0]])
    transformed_gaussian = unscented_transform(gaussian, linear_function)

    np.testing.assert_allclose(transformed_gaussian.mean(), linear_function(mean))
    np.testing.assert_allclose(
        transformed_gaussian.covariance(),
        jacobian @ covariance @ jacobian.T,
    )


import numpy as np

def propagate_covariance(
    jacobian: np.ndarray,
    covariance: np.ndarray
) -> np.ndarray:
    """
    Propagate a covariance through a function
    """
    return np.dot(np.dot(jacobian, covariance), jacobian.T)

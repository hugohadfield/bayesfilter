import numpy as np
import pytest

from examples.linear_tracking import (
    constant_acceleration_matrix,
    constant_velocity_matrix,
    run_linear_tracking,
    run_planar_tracking,
    white_acceleration_covariance,
    white_jerk_covariance,
)


def test_constant_velocity_matrix_has_expected_block_structure():
    matrix = constant_velocity_matrix(2, 0.25)
    expected = np.array(
        [
            [1.0, 0.0, 0.25, 0.0],
            [0.0, 1.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected)


def test_constant_acceleration_matrix_has_expected_1d_form():
    matrix = constant_acceleration_matrix(1, 0.2)
    np.testing.assert_allclose(
        matrix,
        [[1.0, 0.2, 0.02], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]],
    )


@pytest.mark.parametrize(
    "covariance_func, derivative_order",
    [(white_acceleration_covariance, 1), (white_jerk_covariance, 2)],
)
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_process_covariance_is_symmetric_positive_semidefinite(
    covariance_func,
    derivative_order,
    dimension,
):
    covariance = covariance_func(dimension, 0.15, 0.2)
    expected_size = dimension * (derivative_order + 1)
    assert covariance.shape == (expected_size, expected_size)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


@pytest.mark.parametrize("derivative_order", [1, 2])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_linear_tracking_examples_are_finite_and_smoothing_improves_rmse(
    derivative_order,
    dimension,
):
    result = run_linear_tracking(
        dimension,
        derivative_order,
        seed=100 * derivative_order + dimension,
    )
    state_dimension = dimension * (derivative_order + 1)

    assert result["truth"].shape == (81, state_dimension)
    assert result["measurements"].shape == (80, dimension)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert result["smoothed_rmse"] < result["filtered_rmse"]

    for covariance in result["smoothed_covariances"]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_unscented_linear_example_matches_jacobian_example():
    jacobian_result = run_linear_tracking(2, 1, seed=102, use_jacobian=True)
    unscented_result = run_linear_tracking(2, 1, seed=102, use_jacobian=False)

    np.testing.assert_allclose(
        unscented_result["filtered"],
        jacobian_result["filtered"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        unscented_result["smoothed"],
        jacobian_result["smoothed"],
        atol=1e-6,
    )


def test_planar_tracking_example_reconstructs_curved_trajectory():
    result = run_planar_tracking()

    assert result["truth"].shape == (76, 4)
    assert result["measurements"].shape == (75, 2)
    assert result["smoothed_rmse"] < result["filtered_rmse"]
    assert np.ptp(result["truth"][:, 0]) > 6.0
    assert np.ptp(result["truth"][:, 1]) > 5.0
    for covariance in result["smoothed_covariances"]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10

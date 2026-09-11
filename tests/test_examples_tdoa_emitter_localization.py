import numpy as np
import pytest

from examples.tdoa_emitter_localization import (
    MICROSECONDS_PER_KM,
    RECEIVER_POSITIONS_KM,
    TRUE_CLOCK_BIASES_US,
    emitter_transition_jacobian,
    observe_tdoa,
    propagate_emitter_state,
    run_tdoa_emitter_localization,
    tdoa_jacobian,
    tdoa_noise_covariance,
)


def _finite_difference_jacobian(function, state, step=1e-6):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append(
            (function(state + offset) - function(state - offset)) / (2 * step)
        )
    return np.column_stack(columns)


def test_emission_time_cancels_from_tdoa():
    state = np.concatenate((np.array([1.0, 2.0, 0.1, -0.1]), np.zeros(4)))
    ranges = np.linalg.norm(RECEIVER_POSITIONS_KM - state[:2], axis=1)
    emission_time_us = 123456.0
    arrival_times_us = emission_time_us + MICROSECONDS_PER_KM * ranges

    np.testing.assert_allclose(
        arrival_times_us[1:] - arrival_times_us[0],
        observe_tdoa(state),
        atol=2e-11,
    )


def test_tdoa_noise_is_correlated_through_reference_receiver():
    covariance = tdoa_noise_covariance(0.2, 5)

    np.testing.assert_allclose(np.diag(covariance), 0.08)
    np.testing.assert_allclose(covariance[0, 1], 0.04)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_transition_and_tdoa_jacobians_match_finite_difference():
    state = np.concatenate(
        (np.array([1.5, -0.8, 0.09, 0.06]), TRUE_CLOCK_BIASES_US)
    )
    delta_t_s = 0.5

    np.testing.assert_allclose(
        emitter_transition_jacobian(state, delta_t_s),
        _finite_difference_jacobian(
            lambda value: propagate_emitter_state(value, delta_t_s),
            state,
        ),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        tdoa_jacobian(state),
        _finite_difference_jacobian(observe_tdoa, state),
        atol=2e-8,
    )


@pytest.mark.parametrize(
    "use_jacobian, maximum_smoothed_rmse",
    [(False, 0.10), (True, 0.10)],
    ids=["unscented", "jacobian"],
)
def test_tdoa_example_localizes_emitter_and_clock_biases(
    use_jacobian,
    maximum_smoothed_rmse,
):
    result = run_tdoa_emitter_localization(use_jacobian=use_jacobian)

    assert result["truth"].shape == (361, 8)
    assert result["measurements"].shape == (360, 4)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_position_rmse"] < maximum_smoothed_rmse
    np.testing.assert_allclose(
        result["filtered"][-1, 4:],
        result["true_clock_biases_us"],
        atol=0.08,
    )
    assert np.isfinite(result["smoothed"]).all()

    for covariance in result["smoothed_covariances"][::30]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

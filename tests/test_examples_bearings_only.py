import numpy as np
import pytest

from examples.bearings_only import (
    SECOND_SENSOR_DROPOUT,
    bearing_unit_vector_jacobian,
    observe_bearing_unit_vector,
    run_bearings_only_tracking,
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


def test_bearing_observation_is_unit_length_and_has_no_range_information():
    sensor = np.array([-3.0, 1.0])
    state = np.array([2.0, 4.0, 0.3, -0.2])
    direction = observe_bearing_unit_vector(state, sensor)
    jacobian = bearing_unit_vector_jacobian(state, sensor)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_allclose(jacobian[:, :2] @ direction, np.zeros(2), atol=1e-12)
    assert np.linalg.matrix_rank(jacobian) == 1


def test_bearing_jacobian_matches_finite_difference():
    sensor = np.array([4.0, -3.0])
    state = np.array([-1.0, 2.0, 0.3, -0.2])
    numerical = _finite_difference_jacobian(
        lambda value: observe_bearing_unit_vector(value, sensor),
        state,
    )
    np.testing.assert_allclose(
        bearing_unit_vector_jacobian(state, sensor),
        numerical,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "use_jacobian, maximum_smoothed_rmse",
    [(False, 0.20), (True, 0.35)],
    ids=["unscented", "jacobian"],
)
def test_bearings_only_example_recovers_dropout_with_rts(
    use_jacobian,
    maximum_smoothed_rmse,
):
    result = run_bearings_only_tracking(use_jacobian=use_jacobian)

    assert result["truth"].shape == (320, 4)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert [len(times) for times in result["sensor_times"]] == [161, 21]
    second_sensor_times = result["sensor_times"][1]
    assert not np.any(
        (second_sensor_times >= SECOND_SENSOR_DROPOUT[0])
        & (second_sensor_times <= SECOND_SENSOR_DROPOUT[1])
    )
    assert np.all(np.diff(result["event_times"]) >= 0.0)
    assert result["smoothed_position_rmse"] < result["filtered_position_rmse"]
    assert result["smoothed_position_rmse"] < maximum_smoothed_rmse
    assert np.isfinite(result["smoothed"]).all()

    times = result["times"]
    before_dropout = np.abs(times - 5.0).argmin()
    late_dropout = np.abs(times - 10.8).argmin()
    after_reacquisition = np.abs(times - 12.0).argmin()

    def major_position_std(covariance):
        return np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())

    filtered_std = np.array(
        [
            major_position_std(covariance)
            for covariance in result["filtered_covariances"]
        ]
    )
    assert filtered_std[late_dropout] > 8.0 * filtered_std[before_dropout]
    assert filtered_std[after_reacquisition] < 0.35 * filtered_std[late_dropout]
    for covariance in result["smoothed_covariances"][::20]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8

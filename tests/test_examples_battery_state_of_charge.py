import numpy as np

from examples.battery_state_of_charge import (
    battery_transition_jacobian,
    observe_voltage_and_current,
    open_circuit_voltage,
    propagate_battery_state,
    run_battery_state_of_charge,
    voltage_current_jacobian,
)


def _finite_difference_jacobian(function, state, step=1e-6):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append((function(state + offset) - function(state - offset)) / (2 * step))
    return np.column_stack(columns)


def test_open_circuit_voltage_is_physical_and_monotonic():
    charges = np.linspace(0.0, 1.0, 101)
    voltages = open_circuit_voltage(charges)

    assert 2.9 < voltages[0] < 3.3
    assert 4.0 < voltages[-1] < 4.3
    assert np.all(np.diff(voltages) > 0.0)


def test_battery_transition_jacobian_matches_finite_difference():
    state = np.array([0.72, 0.025, np.log(0.061), 1.4])
    delta_t_s = 5.0

    np.testing.assert_allclose(
        battery_transition_jacobian(state, delta_t_s),
        _finite_difference_jacobian(
            lambda candidate: propagate_battery_state(candidate, delta_t_s),
            state,
        ),
        atol=2e-9,
    )


def test_voltage_current_jacobian_matches_finite_difference():
    state = np.array([0.68, 0.018, np.log(0.059), 2.1])

    np.testing.assert_allclose(
        voltage_current_jacobian(state),
        _finite_difference_jacobian(observe_voltage_and_current, state),
        atol=2e-8,
    )


def test_battery_example_estimates_charge_and_resistance():
    result = run_battery_state_of_charge()

    assert result["truth"].shape == (721, 4)
    assert result["measurements"].shape == (720, 2)
    assert result["filtered"].shape == result["truth"].shape
    assert result["smoothed"].shape == result["truth"].shape
    assert result["smoothed_charge_rmse"] < result["filtered_charge_rmse"]
    assert result["smoothed_charge_rmse"] < 0.012

    true_resistance = result["true_resistance_ohm"]
    initial_error = abs(result["filtered_resistance_ohm"][0] - true_resistance)
    final_error = abs(result["filtered_resistance_ohm"][-1] - true_resistance)
    assert final_error < 0.05 * true_resistance
    assert final_error < 0.15 * initial_error
    assert np.isfinite(result["smoothed"]).all()

    for covariance in result["smoothed_covariances"][::45]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-9)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8


def test_battery_example_supports_jacobian_propagation():
    result = run_battery_state_of_charge(
        seed=802,
        duration_s=900.0,
        use_jacobian=True,
    )

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert result["smoothed_charge_rmse"] < 0.025

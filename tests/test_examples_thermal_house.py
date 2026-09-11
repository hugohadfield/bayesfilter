import numpy as np

from examples.thermal_house import (
    HEATER_POWER_KW,
    STEP_S,
    TRUE_CAPACITANCE_KWH_PER_K,
    TRUE_RESISTANCE_K_PER_KW,
    outdoor_temperature,
    propagate_thermal_state,
    run_house_thermal_estimation,
    simulate_house,
)


def test_exact_thermal_step_approaches_expected_equilibrium():
    state = np.array([18.0, np.log(2.0), np.log(8.0)])
    result = propagate_thermal_state(state, 3600.0, 5.0, 1.0)
    equilibrium = 5.0 + 2.0 * HEATER_POWER_KW
    expected_temperature = equilibrium + (18.0 - equilibrium) * np.exp(-1.0 / 16.0)
    assert np.isclose(result[0], expected_temperature)
    assert np.allclose(result[1:], state[1:])


def test_simulation_is_finite_and_heater_switches():
    times, truth, heater_commands = simulate_house()
    assert len(times) == len(truth)
    assert len(heater_commands) == len(times) - 1
    assert np.all(np.isfinite(truth))
    assert np.any(heater_commands == 0.0)
    assert np.any(heater_commands == 1.0)
    assert np.ptp(outdoor_temperature(times)) > 5.0


def test_filter_recovers_thermal_parameters():
    result = run_house_thermal_estimation()
    resistance = result["filtered_resistance_k_per_kw"][-1]
    capacitance = result["filtered_capacitance_kwh_per_k"][-1]
    assert abs(resistance - TRUE_RESISTANCE_K_PER_KW) < 0.25
    assert abs(capacitance - TRUE_CAPACITANCE_KWH_PER_K) < 1.0


def test_rts_improves_temperature_reconstruction():
    result = run_house_thermal_estimation()
    assert result["smoothed_temperature_rmse_c"] < result["filtered_temperature_rmse_c"]
    assert result["smoothed_temperature_rmse_c"] < 0.08


def test_default_run_is_deterministic():
    first = run_house_thermal_estimation()
    second = run_house_thermal_estimation()
    assert np.allclose(first["filtered"], second["filtered"])
    assert np.allclose(first["smoothed"], second["smoothed"])

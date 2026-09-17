import numpy as np
import pytest

from examples.planet_flyby_mass import (
    INITIAL_MU_FRACTION,
    JUPITER_MU_KM3_S2,
    JUPITER_RADIUS_KM,
    gravitational_acceleration,
    run_planet_flyby_mass_estimation,
    simulate_flyby_truth,
)


@pytest.fixture(scope="module")
def result():
    return run_planet_flyby_mass_estimation()


def test_gravity_points_toward_planet_and_scales_with_mass():
    position = np.array([4.0 * JUPITER_RADIUS_KM, 0.0])
    acceleration = gravitational_acceleration(position, JUPITER_MU_KM3_S2)
    lighter = gravitational_acceleration(position, 0.5 * JUPITER_MU_KM3_S2)
    assert acceleration[0] < 0.0
    assert np.isclose(acceleration[1], 0.0)
    assert np.allclose(lighter, 0.5 * acceleration)


def test_truth_executes_close_strongly_deflected_flyby():
    _times, truth = simulate_flyby_truth()
    closest_radius = np.min(np.linalg.norm(truth[:, :2], axis=1))
    incoming_velocity = truth[0, 2:4]
    outgoing_velocity = truth[-1, 2:4]
    cosine = incoming_velocity @ outgoing_velocity / (
        np.linalg.norm(incoming_velocity) * np.linalg.norm(outgoing_velocity)
    )
    deflection_deg = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    assert 3.5 * JUPITER_RADIUS_KM < closest_radius < 5.0 * JUPITER_RADIUS_KM
    assert deflection_deg > 60.0


def test_filter_recovers_planet_mass(result):
    final_ratio = result["final_mu_km3_s2"] / JUPITER_MU_KM3_S2
    assert abs(final_ratio - 1.0) < 0.01
    assert abs(final_ratio - 1.0) < abs(INITIAL_MU_FRACTION - 1.0) / 20.0


def test_estimating_mass_beats_holding_wrong_mass_fixed(result):
    assert (
        result["filtered_position_rmse_km"]
        < 0.20 * result["fixed_wrong_mass_position_rmse_km"]
    )


def test_rts_improves_trajectory_reconstruction(result):
    assert result["smoothed_position_rmse_km"] < result["filtered_position_rmse_km"]
    assert result["smoothed_position_rmse_km"] < 2000.0


def test_mass_uncertainty_contracts(result):
    assert result["final_log_mu_std"] < 0.01

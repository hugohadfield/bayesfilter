import numpy as np

from examples.asteroid_orbit import (
    OBSERVATION_GAP_END_DAY,
    OBSERVATION_GAP_START_DAY,
    TRUE_ECCENTRICITY,
    TRUE_SEMIMAJOR_AXIS_AU,
    earth_position,
    make_observations,
    orbital_elements,
    propagate_orbit,
    run_asteroid_orbit_estimation,
    state_from_orbital_elements,
)


def test_two_body_propagation_preserves_orbital_elements():
    state = state_from_orbital_elements(1.4, 0.22, np.deg2rad(35.0))
    propagated = propagate_orbit(state, 80.0)
    initial_a, initial_e = orbital_elements(state)
    final_a, final_e = orbital_elements(propagated)
    assert abs(final_a - initial_a) < 2e-8
    assert abs(final_e - initial_e) < 2e-8


def test_earth_orbit_stays_at_one_au():
    for time_days in (0.0, 30.0, 90.0, 180.0):
        assert np.isclose(np.linalg.norm(earth_position(time_days)), 1.0)


def test_observation_schedule_contains_the_configured_gap():
    times, measurements = make_observations()
    assert len(times) == len(measurements)
    assert not np.any(
        (times > OBSERVATION_GAP_START_DAY)
        & (times < OBSERVATION_GAP_END_DAY)
    )
    assert np.any(times < OBSERVATION_GAP_START_DAY)
    assert np.any(times > OBSERVATION_GAP_END_DAY)


def test_filter_recovers_orbital_elements_from_angles_only():
    result = run_asteroid_orbit_estimation()
    semimajor_axis, eccentricity = result["final_elements"]
    assert abs(semimajor_axis - TRUE_SEMIMAJOR_AXIS_AU) < 0.02
    assert abs(eccentricity - TRUE_ECCENTRICITY) < 0.01


def test_orbit_estimate_improves_substantially():
    result = run_asteroid_orbit_estimation()
    initial_a, initial_e = result["initial_elements"]
    final_a, final_e = result["final_elements"]
    assert abs(final_a - TRUE_SEMIMAJOR_AXIS_AU) < 0.1 * abs(
        initial_a - TRUE_SEMIMAJOR_AXIS_AU
    )
    assert abs(final_e - TRUE_ECCENTRICITY) < 0.1 * abs(
        initial_e - TRUE_ECCENTRICITY
    )

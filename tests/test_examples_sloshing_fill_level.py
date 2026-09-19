import numpy as np

from examples.sloshing_fill_level import (
    DEFAULT_DURATION_S,
    DEPTH_INDEX,
    FILL_RATE_INDEX,
    GRAVITY_M_S2,
    SHAKE_INTERVAL_S,
    SLOSH_WAVENUMBER_PER_M,
    TRUE_FINAL_DEPTH_M,
    TRUE_INITIAL_DEPTH_M,
    commanded_tank_acceleration,
    propagate_slosh_state,
    run_sloshing_fill_level,
    slosh_frequency_hz,
    slosh_frequency_sensitivity_hz_per_m,
)


def test_slosh_frequency_increases_with_depth_and_saturates():
    depths = np.array([0.03, 0.08, 0.15, 0.22, 1.0])
    frequencies = slosh_frequency_hz(depths)

    assert np.all(np.diff(frequencies) > 0.0)

    deep_water_limit_hz = (
        np.sqrt(GRAVITY_M_S2 * SLOSH_WAVENUMBER_PER_M)
        / (2.0 * np.pi)
    )
    assert frequencies[-1] < deep_water_limit_hz
    np.testing.assert_allclose(
        frequencies[-1],
        deep_water_limit_hz,
        rtol=2e-4,
    )


def test_frequency_sensitivity_collapses_as_tank_gets_deep():
    shallow = slosh_frequency_sensitivity_hz_per_m(0.03)
    deep = slosh_frequency_sensitivity_hz_per_m(0.22)

    assert shallow > 30.0 * deep


def test_unexcited_state_fills_linearly_without_creating_slosh():
    fill_rate = 0.0015
    state = np.array([0.0, 0.0, 0.05, fill_rate, 0.0])

    propagated = propagate_slosh_state(state, 2.0)

    np.testing.assert_allclose(propagated[0:2], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(
        propagated[DEPTH_INDEX],
        0.05 + 2.0 * fill_rate,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        propagated[FILL_RATE_INDEX],
        fill_rate,
        atol=1e-12,
    )


def test_shake_schedule_is_periodic_and_zero_between_events():
    assert commanded_tank_acceleration(8.5) != 0.0
    assert commanded_tank_acceleration(8.5 + SHAKE_INTERVAL_S) != 0.0
    np.testing.assert_allclose(
        commanded_tank_acceleration(13.0),
        0.0,
        atol=1e-12,
    )


def test_sloshing_recovers_fill_level_and_rate_without_direct_level_sensor():
    result = run_sloshing_fill_level()

    assert result["filtered_depth_rmse_m"] < 0.03
    assert result["smoothed_depth_rmse_m"] < 0.015
    assert (
        result["smoothed_depth_rmse_m"]
        < 0.6 * result["filtered_depth_rmse_m"]
    )

    assert abs(
        result["final_filtered_depth_m"] - TRUE_FINAL_DEPTH_M
    ) < 0.03

    true_fill_rate = (
        TRUE_FINAL_DEPTH_M - TRUE_INITIAL_DEPTH_M
    ) / DEFAULT_DURATION_S
    assert abs(
        result["final_filtered_fill_rate_m_s"] - true_fill_rate
    ) < 3e-4

    # The first shake is much more informative than late high-fill shakes:
    # the fundamental frequency curve has already started flattening.
    assert (
        result["frequency_sensitivity_hz_per_m"][0]
        > 30.0 * result["frequency_sensitivity_hz_per_m"][-1]
    )

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.all(result["filtered_depth_std_m"] >= 0.0)
    assert np.all(result["smoothed_depth_std_m"] >= 0.0)

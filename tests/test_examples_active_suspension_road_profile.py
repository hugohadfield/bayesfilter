import numpy as np

from examples.active_suspension_road_profile import (
    ACTUATOR_FORCE_INDEX,
    LOG_DAMPING_INDEX,
    LOG_STIFFNESS_INDEX,
    ROAD_HEIGHT_INDEX,
    TRUE_SUSPENSION_DAMPING_N_S_M,
    TRUE_SUSPENSION_STIFFNESS_N_M,
    active_actuator_force_n,
    observe_suspension,
    road_profile_m,
    run_active_suspension_road_profile,
    suspension_derivative,
)


def test_quarter_car_derivative_has_stationary_slow_parameters():
    state = np.array(
        [
            0.015,
            0.25,
            -0.004,
            -0.10,
            0.006,
            0.02,
            np.log(18000.0),
            np.log(1800.0),
            250.0,
        ]
    )

    derivative = suspension_derivative(state)

    np.testing.assert_allclose(
        derivative[[LOG_STIFFNESS_INDEX, LOG_DAMPING_INDEX]],
        np.zeros(2),
    )
    np.testing.assert_allclose(
        derivative[ACTUATOR_FORCE_INDEX],
        0.0,
    )


def test_sensor_model_does_not_directly_measure_road_height():
    state = np.array(
        [
            0.010,
            0.2,
            0.002,
            -0.1,
            0.005,
            0.0,
            np.log(18000.0),
            np.log(1800.0),
            100.0,
        ]
    )

    measurement = observe_suspension(state)

    assert measurement.shape == (4,)
    assert not np.any(np.isclose(measurement, state[ROAD_HEIGHT_INDEX]))


def test_road_profile_contains_positive_and_negative_features():
    times = np.linspace(0.0, 12.0, 1201)
    road = np.array([road_profile_m(time_s) for time_s in times])

    assert np.max(road) > 0.03
    assert np.min(road) < -0.02


def test_active_excitation_is_nonzero_only_in_excitation_windows():
    assert abs(active_actuator_force_n(3.6)) < 1e-9
    assert abs(active_actuator_force_n(3.9)) > 100.0
    assert abs(active_actuator_force_n(0.5)) < 1.0


def test_active_suspension_recovers_road_and_slow_parameters():
    result = run_active_suspension_road_profile()

    assert result["filtered_road_rmse_m"] < 0.007
    assert result["smoothed_road_rmse_m"] < 0.004
    assert (
        result["smoothed_road_rmse_m"]
        < 0.75 * result["filtered_road_rmse_m"]
    )

    assert result["filtered_road_correlation"] > 0.70
    assert result["smoothed_road_correlation"] > 0.90

    assert abs(
        result["final_filtered_stiffness_n_m"]
        - TRUE_SUSPENSION_STIFFNESS_N_M
    ) < 700.0

    assert abs(
        result["final_filtered_damping_n_s_m"]
        - TRUE_SUSPENSION_DAMPING_N_S_M
    ) < 120.0

    # The filter starts substantially wrong, so successful convergence is
    # meaningful rather than just preserving a good prior.
    assert (
        abs(
            result["initial_stiffness_guess_n_m"]
            - TRUE_SUSPENSION_STIFFNESS_N_M
        )
        > 7000.0
    )
    assert (
        abs(
            result["initial_damping_guess_n_s_m"]
            - TRUE_SUSPENSION_DAMPING_N_S_M
        )
        > 800.0
    )

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.all(result["filtered_road_std_m"] >= 0.0)


def test_slow_parameters_are_positive_by_construction():
    result = run_active_suspension_road_profile()

    assert np.all(result["filtered_stiffness_n_m"] > 0.0)
    assert np.all(result["smoothed_stiffness_n_m"] > 0.0)
    assert np.all(result["filtered_damping_n_s_m"] > 0.0)
    assert np.all(result["smoothed_damping_n_s_m"] > 0.0)

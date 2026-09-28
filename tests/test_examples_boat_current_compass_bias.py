import numpy as np

from examples.boat_current_compass_bias import (
    COMPASS_BIAS_INDEX,
    CURRENT_X_INDEX,
    CURRENT_Y_INDEX,
    HEADING_INDEX,
    TRUE_COMPASS_BIAS_RAD,
    WATER_SPEED_INDEX,
    YAW_RATE_INDEX,
    boat_transition_jacobian,
    navigation_without_gps_jacobian,
    observe_navigation_without_gps,
    propagate_boat_state,
    run_boat_current_compass_bias,
    true_current_m_s,
)


def test_boat_transition_matches_kinematics():
    state = np.array(
        [
            10.0,
            -3.0,
            np.deg2rad(30.0),
            2.5,
            np.deg2rad(5.0),
            0.4,
            -0.2,
            np.deg2rad(8.0),
        ]
    )

    propagated = propagate_boat_state(state, 0.2)

    np.testing.assert_allclose(
        propagated[0],
        10.0 + 0.2 * (2.5 * np.cos(np.deg2rad(30.0)) + 0.4),
    )
    np.testing.assert_allclose(
        propagated[1],
        -3.0 + 0.2 * (2.5 * np.sin(np.deg2rad(30.0)) - 0.2),
    )
    np.testing.assert_allclose(
        propagated[HEADING_INDEX],
        np.deg2rad(31.0),
    )


def test_transition_jacobian_matches_finite_difference():
    state = np.array(
        [
            2.0,
            -1.0,
            0.7,
            2.6,
            0.08,
            0.3,
            -0.2,
            0.1,
        ]
    )
    delta_t_s = 0.1
    analytic = boat_transition_jacobian(state, delta_t_s)

    epsilon = 1e-6
    numerical = np.empty_like(analytic)
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = epsilon
        numerical[:, index] = (
            propagate_boat_state(state + offset, delta_t_s)
            - propagate_boat_state(state - offset, delta_t_s)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic,
        numerical,
        atol=2e-7,
        rtol=2e-6,
    )


def test_compass_unit_vector_observation_wraps_continuously():
    state_a = np.zeros(8)
    state_b = np.zeros(8)

    state_a[HEADING_INDEX] = np.pi - 1e-6
    state_b[HEADING_INDEX] = -np.pi + 1e-6

    observation_a = observe_navigation_without_gps(state_a)[2:]
    observation_b = observe_navigation_without_gps(state_b)[2:]

    assert np.linalg.norm(observation_a - observation_b) < 3e-6


def test_compass_observation_jacobian_matches_finite_difference():
    state = np.array(
        [
            1.0,
            2.0,
            0.5,
            2.4,
            -0.04,
            0.4,
            -0.3,
            0.12,
        ]
    )
    analytic = navigation_without_gps_jacobian(state)

    epsilon = 1e-6
    numerical = np.empty_like(analytic)
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = epsilon
        numerical[:, index] = (
            observe_navigation_without_gps(state + offset)
            - observe_navigation_without_gps(state - offset)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic,
        numerical,
        atol=2e-7,
        rtol=2e-6,
    )


def test_true_current_varies_slowly():
    current_start = true_current_m_s(0.0)
    current_one_second = true_current_m_s(1.0)
    current_late = true_current_m_s(45.0)

    assert np.linalg.norm(current_one_second - current_start) < 0.02
    assert np.linalg.norm(current_late - current_start) > 0.05


def test_boat_infers_current_and_compass_bias_after_heading_excitation():
    result = run_boat_current_compass_bias()

    assert result["filtered_position_rmse_m"] < 0.70
    assert result["smoothed_position_rmse_m"] < 0.35

    assert result["filtered_current_rmse_m_s"] < 0.25
    assert result["smoothed_current_rmse_m_s"] < 0.06

    assert result["filtered_bias_rmse_deg"] < 4.0
    assert result["smoothed_bias_rmse_deg"] < 0.50

    assert abs(
        result["final_filtered_bias_deg"]
        - result["true_compass_bias_deg"]
    ) < 0.70

    np.testing.assert_allclose(
        result["final_filtered_current_m_s"],
        result["true_final_current_m_s"],
        atol=0.06,
    )

    # During the initial straight segment, current and compass bias remain
    # strongly confounded. Deliberate heading changes break that ambiguity.
    index_10s = np.argmin(np.abs(result["times"] - 10.0))
    index_34s = np.argmin(np.abs(result["times"] - 34.0))

    bias_error_10s_deg = abs(
        np.rad2deg(
            result["filtered"][index_10s, COMPASS_BIAS_INDEX]
            - TRUE_COMPASS_BIAS_RAD
        )
    )
    bias_error_34s_deg = abs(
        np.rad2deg(
            result["filtered"][index_34s, COMPASS_BIAS_INDEX]
            - TRUE_COMPASS_BIAS_RAD
        )
    )

    assert bias_error_10s_deg > 2.0
    assert bias_error_34s_deg < 1.5

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()

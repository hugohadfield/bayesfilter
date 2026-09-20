import numpy as np

from examples.pendulum_parameter_inference import (
    GRAVITY_M_S2,
    TRUE_DAMPING_PER_S,
    TRUE_LENGTH_M,
    LOG_DAMPING_INDEX,
    LOG_LENGTH_INDEX,
    pendulum_derivative,
    propagate_pendulum_state,
    run_pendulum_parameter_inference,
)


def test_pendulum_derivative_matches_expected_acceleration():
    state = np.array(
        [
            np.deg2rad(30.0),
            0.4,
            np.log(0.8),
            np.log(0.12),
        ]
    )
    derivative = pendulum_derivative(state)

    expected = (
        -(GRAVITY_M_S2 / 0.8) * np.sin(np.deg2rad(30.0))
        - 0.12 * 0.4
    )

    np.testing.assert_allclose(derivative[0], 0.4)
    np.testing.assert_allclose(derivative[1], expected)
    np.testing.assert_allclose(derivative[2:], np.zeros(2))


def test_stationary_parameters_remain_constant_under_dynamics():
    state = np.array(
        [
            np.deg2rad(20.0),
            -0.3,
            np.log(0.9),
            np.log(0.08),
        ]
    )

    propagated = propagate_pendulum_state(state, 0.25)

    np.testing.assert_allclose(
        propagated[LOG_LENGTH_INDEX],
        state[LOG_LENGTH_INDEX],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        propagated[LOG_DAMPING_INDEX],
        state[LOG_DAMPING_INDEX],
        atol=1e-12,
    )


def test_pendulum_tracks_fast_motion_and_infers_slow_parameters():
    result = run_pendulum_parameter_inference()

    assert result["filtered_angle_rmse_deg"] < 0.6
    assert result["smoothed_angle_rmse_deg"] < 0.35
    assert result["filtered_angular_rate_rmse_rad_s"] < 0.08
    assert result["smoothed_angular_rate_rmse_rad_s"] < 0.05

    assert abs(
        result["final_filtered_length_m"] - TRUE_LENGTH_M
    ) < 0.01
    assert abs(
        result["final_filtered_damping_per_s"]
        - TRUE_DAMPING_PER_S
    ) < 0.01

    # Length is identified quickly from oscillation period.
    index_2s = np.argmin(np.abs(result["times"] - 2.0))
    assert abs(
        result["filtered_length_m"][index_2s] - TRUE_LENGTH_M
    ) < 0.02

    # RTS should make the stationary parameter histories essentially flat and
    # close to their final values by using the whole trajectory.
    smoothed_length_rmse = np.sqrt(
        np.mean(
            (
                result["smoothed_length_m"]
                - TRUE_LENGTH_M
            )
            ** 2
        )
    )
    smoothed_damping_rmse = np.sqrt(
        np.mean(
            (
                result["smoothed_damping_per_s"]
                - TRUE_DAMPING_PER_S
            )
            ** 2
        )
    )
    assert smoothed_length_rmse < 0.002
    assert smoothed_damping_rmse < 0.005

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.all(result["filtered_length_std_m"] > 0.0)
    assert np.all(result["filtered_damping_std_per_s"] > 0.0)


def test_length_and_damping_are_positive_by_construction():
    result = run_pendulum_parameter_inference()

    assert np.all(result["filtered_length_m"] > 0.0)
    assert np.all(result["smoothed_length_m"] > 0.0)
    assert np.all(result["filtered_damping_per_s"] > 0.0)
    assert np.all(result["smoothed_damping_per_s"] > 0.0)

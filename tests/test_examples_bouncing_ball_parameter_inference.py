import numpy as np

from examples.bouncing_ball_parameter_inference import (
    HEIGHT_INDEX,
    LOG_DRAG_INDEX,
    LOGIT_RESTITUTION_INDEX,
    TRUE_DRAG_PER_M,
    TRUE_RESTITUTION,
    VELOCITY_INDEX,
    bouncing_ball_derivative,
    logit,
    propagate_bouncing_ball,
    run_bouncing_ball_parameter_inference,
    sigmoid,
)


def test_logit_sigmoid_round_trip():
    probabilities = np.array([0.2, 0.55, 0.78, 0.95])
    np.testing.assert_allclose(
        sigmoid(logit(probabilities)),
        probabilities,
        atol=1e-12,
    )


def test_airborne_dynamics_apply_gravity_and_quadratic_drag():
    state = np.array(
        [
            3.0,
            -4.0,
            np.log(0.05),
            logit(0.8),
        ]
    )
    derivative = bouncing_ball_derivative(state)

    np.testing.assert_allclose(derivative[HEIGHT_INDEX], -4.0)
    np.testing.assert_allclose(
        derivative[VELOCITY_INDEX],
        -9.80665 + 0.05 * 16.0,
    )
    np.testing.assert_allclose(
        derivative[[LOG_DRAG_INDEX, LOGIT_RESTITUTION_INDEX]],
        np.zeros(2),
    )


def test_floor_impact_reverses_velocity_and_applies_restitution():
    state = np.array(
        [
            0.01,
            -3.0,
            np.log(TRUE_DRAG_PER_M),
            logit(TRUE_RESTITUTION),
        ]
    )

    propagated = propagate_bouncing_ball(state, 0.02)

    assert propagated[HEIGHT_INDEX] >= 0.0
    assert propagated[VELOCITY_INDEX] > 0.0
    assert propagated[VELOCITY_INDEX] < 3.0


def test_bouncing_ball_tracks_motion_and_infers_physics():
    result = run_bouncing_ball_parameter_inference()

    assert len(result["bounce_indices"]) >= 4

    assert result["filtered_height_rmse_m"] < 0.035
    assert result["smoothed_height_rmse_m"] < 0.020
    assert result["filtered_velocity_rmse_m_s"] < 0.65
    assert result["smoothed_velocity_rmse_m_s"] < 0.55

    assert abs(
        result["final_filtered_drag_per_m"] - TRUE_DRAG_PER_M
    ) < 0.006
    assert abs(
        result["final_filtered_restitution"] - TRUE_RESTITUTION
    ) < 0.025

    # Restitution is not observable before the first impact, then moves sharply
    # toward the truth immediately after that impact.
    first_bounce = result["bounce_indices"][0]
    before = result["filtered_restitution"][first_bounce - 2]
    after = result["filtered_restitution"][first_bounce + 5]
    assert abs(before - 0.55) < 0.03
    assert abs(after - TRUE_RESTITUTION) < 0.06

    # Drag is learned from the first airborne arc, before restitution becomes
    # observable at the floor.
    half_second = np.argmin(np.abs(result["times"] - 0.5))
    one_second = np.argmin(np.abs(result["times"] - 1.0))
    assert abs(
        result["filtered_drag_per_m"][one_second] - TRUE_DRAG_PER_M
    ) < abs(
        result["filtered_drag_per_m"][half_second] - TRUE_DRAG_PER_M
    )

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    assert np.all(result["filtered_drag_per_m"] > 0.0)
    assert np.all(
        (result["filtered_restitution"] > 0.0)
        & (result["filtered_restitution"] < 1.0)
    )

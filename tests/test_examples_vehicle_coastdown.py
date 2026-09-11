import numpy as np

from examples.vehicle_coastdown import (
    DRAG_AREA,
    ROLLING_RESISTANCE,
    SPEED,
    TRUE_DRAG_AREA_M2,
    TRUE_ROLLING_RESISTANCE,
    coastdown_acceleration,
    propagate_coastdown_state,
    run_vehicle_coastdown,
    simulate_coastdown_truth,
)


def test_drag_deceleration_increases_with_speed():
    low_speed_state = np.array(
        [10.0, TRUE_ROLLING_RESISTANCE, TRUE_DRAG_AREA_M2]
    )
    high_speed_state = np.array(
        [35.0, TRUE_ROLLING_RESISTANCE, TRUE_DRAG_AREA_M2]
    )
    assert coastdown_acceleration(high_speed_state) < coastdown_acceleration(
        low_speed_state
    ) < 0.0


def test_transition_preserves_physical_parameters():
    state = np.array([30.0, 0.014, 0.70])
    propagated = propagate_coastdown_state(state, 3.0)
    assert propagated[SPEED] < state[SPEED]
    assert np.allclose(
        propagated[[ROLLING_RESISTANCE, DRAG_AREA]],
        state[[ROLLING_RESISTANCE, DRAG_AREA]],
    )


def test_truth_spans_wide_speed_range():
    _times, truth = simulate_coastdown_truth()
    assert truth[0, SPEED] > 35.0
    assert truth[-1, SPEED] < 8.0


def test_filter_recovers_drag_and_rolling_resistance():
    result = run_vehicle_coastdown()
    assert (
        abs(result["final_rolling_resistance"] - TRUE_ROLLING_RESISTANCE)
        < 4e-4
    )
    assert abs(result["final_drag_area_m2"] - TRUE_DRAG_AREA_M2) < 0.03


def test_rts_improves_speed_reconstruction():
    result = run_vehicle_coastdown()
    assert result["smoothed_speed_rmse_mps"] < result["filtered_speed_rmse_mps"]

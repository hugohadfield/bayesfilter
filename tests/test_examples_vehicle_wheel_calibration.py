import numpy as np

from examples.vehicle_wheel_calibration import (
    GPS_OUTAGE_END_S,
    GPS_OUTAGE_START_S,
    TRUE_SPEEDOMETER_BIAS_MPS,
    TRUE_WHEEL_RADIUS_M,
    LOG_WHEEL_RADIUS,
    SPEEDOMETER_BIAS,
    propagate_vehicle_state,
    run_vehicle_wheel_calibration,
)


def test_vehicle_transition_constant_acceleration():
    state = np.array([10.0, 4.0, 2.0, np.log(0.32), 0.5])
    propagated = propagate_vehicle_state(state, 3.0)
    np.testing.assert_allclose(
        propagated,
        np.array([31.0, 10.0, 2.0, np.log(0.32), 0.5]),
    )


def test_default_run_recovers_calibration_parameters():
    result = run_vehicle_wheel_calibration()
    assert abs(result["final_wheel_radius_m"] - TRUE_WHEEL_RADIUS_M) < 0.005
    assert abs(result["final_speedometer_bias_mps"] - TRUE_SPEEDOMETER_BIAS_MPS) < 0.08


def test_rts_improves_speed_reconstruction():
    result = run_vehicle_wheel_calibration()
    assert result["smoothed_speed_rmse_mps"] < result["filtered_speed_rmse_mps"]


def test_gps_outage_is_present():
    result = run_vehicle_wheel_calibration()
    assert not np.any(
        (result["gps_times"] >= GPS_OUTAGE_START_S)
        & (result["gps_times"] <= GPS_OUTAGE_END_S)
    )
    assert np.any(result["wheel_times"] > GPS_OUTAGE_START_S)
    assert np.any(result["wheel_times"] < GPS_OUTAGE_END_S)


def test_calibration_is_stable_late_in_drive():
    result = run_vehicle_wheel_calibration()
    smoothed = result["smoothed"]
    second_half = slice(len(smoothed) // 2, None)
    mean_radius = np.mean(np.exp(smoothed[second_half, LOG_WHEEL_RADIUS]))
    mean_bias = np.mean(smoothed[second_half, SPEEDOMETER_BIAS])
    assert abs(mean_radius - TRUE_WHEEL_RADIUS_M) < 0.005
    assert abs(mean_bias - TRUE_SPEEDOMETER_BIAS_MPS) < 0.08

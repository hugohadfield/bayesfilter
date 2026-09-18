import numpy as np
import pytest

from examples.doppler_source_localization import (
    EMITTED_FREQUENCY,
    FREQUENCY_STD_HZ,
    INITIAL_RECEIVER_POSITION_M,
    SPEED_OF_SOUND_M_S,
    STRAIGHT_SEGMENT_END_S,
    TRUE_EMITTED_FREQUENCY_HZ,
    TRUE_SOURCE_POSITION_M,
    TRUE_SOURCE_STATE,
    doppler_jacobian,
    mirror_source_across_initial_road,
    observe_doppler,
    run_doppler_source_localization,
    simulate_doppler_measurements,
)


def _finite_difference_jacobian(function, state, step=1e-5):
    columns = []
    for index in range(len(state)):
        offset = np.zeros_like(state)
        offset[index] = step
        columns.append(
            (function(state + offset) - function(state - offset))
            / (2.0 * step)
        )
    return np.column_stack(columns)


def test_doppler_shift_has_expected_approaching_and_receding_sign():
    source = np.array([100.0, 0.0, 700.0])
    receiver_position = np.array([0.0, 0.0])

    approaching = observe_doppler(
        source,
        receiver_position,
        np.array([15.0, 0.0]),
    )[0]
    receding = observe_doppler(
        source,
        receiver_position,
        np.array([-15.0, 0.0]),
    )[0]

    assert approaching > source[EMITTED_FREQUENCY]
    assert receding < source[EMITTED_FREQUENCY]
    np.testing.assert_allclose(
        approaching,
        source[EMITTED_FREQUENCY]
        * (1.0 + 15.0 / SPEED_OF_SOUND_M_S),
    )


def test_doppler_jacobian_matches_finite_difference():
    state = np.array([320.0, 180.0, 718.0])
    receiver_position = np.array([-250.0, -80.0])
    receiver_velocity = np.array([18.0, 9.0])

    numerical = _finite_difference_jacobian(
        lambda value: observe_doppler(
            value,
            receiver_position,
            receiver_velocity,
        ),
        state,
    )
    np.testing.assert_allclose(
        doppler_jacobian(
            state,
            receiver_position,
            receiver_velocity,
        ),
        numerical,
        atol=2e-7,
        rtol=2e-6,
    )


def test_straight_road_has_exact_mirror_source_ambiguity():
    data = simulate_doppler_measurements(seed=3)
    mirror_state = mirror_source_across_initial_road()
    straight = data["times"] <= STRAIGHT_SEGMENT_END_S

    true_frequency = data["expected_frequency_hz"][straight]
    mirror_frequency = data["mirror_frequency_hz"][straight]
    np.testing.assert_allclose(
        true_frequency,
        mirror_frequency,
        atol=1e-12,
        rtol=0.0,
    )

    road_y = INITIAL_RECEIVER_POSITION_M[1]
    np.testing.assert_allclose(
        mirror_state[1],
        2.0 * road_y - TRUE_SOURCE_POSITION_M[1],
    )


def test_receiver_turn_breaks_mirror_source_ambiguity():
    data = simulate_doppler_measurements(seed=3)
    after_turn = data["times"] >= 40.0
    separation_hz = np.abs(
        data["expected_frequency_hz"][after_turn]
        - data["mirror_frequency_hz"][after_turn]
    )

    assert separation_hz.max() > 50.0
    assert separation_hz.mean() > 10.0


def test_only_scalar_frequency_is_measured():
    data = simulate_doppler_measurements(seed=7)

    assert data["measurements_hz"].ndim == 1
    assert data["measurements_hz"].shape == data["times"].shape
    assert FREQUENCY_STD_HZ < np.ptp(data["expected_frequency_hz"])


@pytest.mark.parametrize(
    "use_jacobian",
    [False, True],
    ids=["unscented", "jacobian"],
)
def test_doppler_only_example_recovers_source_and_emitted_frequency(
    use_jacobian,
):
    result = run_doppler_source_localization(use_jacobian=use_jacobian)

    assert result["observation_rate_hz"] == 2.0

    # The straight segment remains badly localized despite many precise
    # frequency measurements because the mirror ambiguity has not been broken.
    assert result["straight_end_position_error_m"] > 100.0
    assert result["straight_end_position_std_m"] > 15.0

    # Receiver maneuvers make the source geometry observable.
    assert result["post_turn_position_error_m"] < 20.0
    assert result["post_turn_position_std_m"] < 5.0
    assert result["final_position_error_m"] < 2.0
    assert abs(result["final_frequency_error_hz"]) < 0.5

    # RTS can use the later maneuver to reconstruct where the static source
    # must have been even during the earlier ambiguous interval.
    assert result["smoothed_position_rmse_m"] < 2.0
    assert result["smoothed_position_rmse_m"] < result["filtered_position_rmse_m"]
    assert result["initial_smoothed_position_error_m"] < 2.0

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()
    for covariance in result["smoothed_covariances"][::30]:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-6

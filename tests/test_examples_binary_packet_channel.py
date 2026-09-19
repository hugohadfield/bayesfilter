import numpy as np

from examples.binary_packet_channel import (
    SUCCESS_TRANSITION_WIDTH_DB,
    packet_success_jacobian,
    packet_success_probability,
    run_binary_packet_channel,
    simulate_packet_link,
)


def test_packet_success_probability_is_monotonic_and_centered():
    margins = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    probabilities = packet_success_probability(margins)

    assert np.all(np.diff(probabilities) > 0.0)
    np.testing.assert_allclose(probabilities[2], 0.5, atol=1e-12)
    assert probabilities[0] < 1e-3
    assert probabilities[-1] > 1.0 - 1e-3


def test_packet_success_jacobian_matches_finite_difference():
    state = np.array([0.7])
    epsilon = 1e-6

    plus = packet_success_probability(state[0] + epsilon)
    minus = packet_success_probability(state[0] - epsilon)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        packet_success_jacobian(state)[0, 0],
        finite_difference,
        rtol=1e-6,
        atol=1e-9,
    )


def test_binary_packet_simulation_is_reproducible():
    first = simulate_packet_link(seed=7)
    second = simulate_packet_link(seed=7)

    np.testing.assert_array_equal(
        first["packet_received"],
        second["packet_received"],
    )
    np.testing.assert_allclose(
        first["true_margin_db"],
        second["true_margin_db"],
    )


def test_binary_packets_recover_continuous_channel_quality():
    result = run_binary_packet_channel(seed=7, use_jacobian=True)

    assert 0.6 < result["packet_success_rate"] < 0.8

    # A continuous hidden dB-scale channel state is recovered from binary
    # packet successes/failures alone.
    assert result["filtered_margin_rmse_db"] < 1.7
    assert result["smoothed_margin_rmse_db"] < 1.4
    assert (
        result["smoothed_margin_rmse_db"]
        < result["filtered_margin_rmse_db"]
    )

    # Binary observations are most informative around p=0.5. Once the link is
    # saturated into almost-all-successes/failures, only a loose bound remains.
    assert (
        result["threshold_region_mean_std_db"]
        < 0.5 * result["saturated_region_mean_std_db"]
    )

    assert np.isfinite(result["filtered_margin_db"]).all()
    assert np.isfinite(result["smoothed_margin_db"]).all()
    assert np.all(result["filtered_std_db"] > 0.0)
    assert np.all(result["smoothed_std_db"] > 0.0)


def test_transition_width_sets_half_success_slope():
    expected = 0.25 / SUCCESS_TRANSITION_WIDTH_DB
    np.testing.assert_allclose(
        packet_success_jacobian(np.array([0.0]))[0, 0],
        expected,
        atol=1e-12,
    )

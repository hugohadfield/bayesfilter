import numpy as np

from examples.quantization_noise import (
    ENCODER_RESOLUTION_DEG,
    ENCODER_RESOLUTION_RAD,
    quantization_standard_deviation,
    run_quantization_noise_comparison,
    simulate_quantized_encoder,
)


def test_uniform_quantization_standard_deviation_formula():
    expected = ENCODER_RESOLUTION_RAD / np.sqrt(12.0)

    np.testing.assert_allclose(
        quantization_standard_deviation(
            ENCODER_RESOLUTION_RAD
        ),
        expected,
    )


def test_quantization_error_stays_inside_half_bin():
    data = simulate_quantized_encoder()

    assert np.all(
        np.abs(data["quantization_error"])
        <= 0.5 * ENCODER_RESOLUTION_RAD + 1e-12
    )


def test_empirical_quantization_noise_matches_uniform_bin_model():
    result = run_quantization_noise_comparison()

    empirical_std_deg = np.rad2deg(
        result["empirical_quantization_std_rad"]
    )
    expected_std_deg = (
        ENCODER_RESOLUTION_DEG / np.sqrt(12.0)
    )

    assert abs(
        empirical_std_deg - expected_std_deg
    ) < 0.002


def test_using_resolution_as_sigma_is_too_conservative():
    result = run_quantization_noise_comparison()

    # If R = Delta^2, the filter claims one standard deviation equals an
    # entire quantization bin. The resulting normalized innovations are much
    # smaller than a well-specified scalar model should produce.
    assert result["wrong"]["mean_nis"] < 0.30

    # The uniform-bin approximation R = Delta^2 / 12 is much closer to a
    # statistically consistent scalar innovation sequence.
    assert 0.75 < result["uniform"]["mean_nis"] < 1.30

    assert (
        abs(result["uniform"]["mean_nis"] - 1.0)
        < abs(result["wrong"]["mean_nis"] - 1.0)
    )


def test_correct_quantization_variance_does_not_hurt_tracking():
    result = run_quantization_noise_comparison()

    assert (
        result["uniform"]["angle_rmse_rad"]
        < result["wrong"]["angle_rmse_rad"]
    )

    assert (
        result["uniform"]["rate_rmse_rad_s"]
        < result["wrong"]["rate_rmse_rad_s"]
    )

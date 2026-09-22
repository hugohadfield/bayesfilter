import numpy as np

from examples.rigid_body import rotation_distance
from examples.wrist_camera_hand_eye_calibration import (
    DETECTION_DROPOUTS_S,
    TRUE_BASE_TO_OBJECT_ROTATION_VECTOR,
    TRUE_BASE_TO_OBJECT_TRANSLATION_M,
    TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR,
    TRUE_WRIST_TO_CAMERA_TRANSLATION_M,
    compose_pose,
    invert_pose,
    relative_pose,
    run_wrist_camera_calibration,
    simulate_wrist_camera_calibration,
    wrist_pose_at_time,
)


def test_pose_inverse_round_trip():
    translation = np.array([0.3, -0.2, 0.5])
    rotation = np.array([0.2, -0.4, 0.1])

    inverse_translation, inverse_rotation = invert_pose(
        translation,
        rotation,
    )
    identity_translation, identity_rotation = compose_pose(
        translation,
        rotation,
        inverse_translation,
        inverse_rotation,
    )

    np.testing.assert_allclose(
        identity_translation,
        np.zeros(3),
        atol=1e-10,
    )
    assert np.linalg.norm(identity_rotation) < 1e-10


def test_hand_eye_transform_chain_is_consistent():
    wrist_translation, wrist_rotation = wrist_pose_at_time(2.4)

    camera_to_object_translation, camera_to_object_rotation = relative_pose(
        wrist_translation,
        wrist_rotation,
        TRUE_WRIST_TO_CAMERA_TRANSLATION_M,
        TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR,
        TRUE_BASE_TO_OBJECT_TRANSLATION_M,
        TRUE_BASE_TO_OBJECT_ROTATION_VECTOR,
    )

    base_to_camera_translation, base_to_camera_rotation = compose_pose(
        wrist_translation,
        wrist_rotation,
        TRUE_WRIST_TO_CAMERA_TRANSLATION_M,
        TRUE_WRIST_TO_CAMERA_ROTATION_VECTOR,
    )
    reconstructed_object_translation, reconstructed_object_rotation = (
        compose_pose(
            base_to_camera_translation,
            base_to_camera_rotation,
            camera_to_object_translation,
            camera_to_object_rotation,
        )
    )

    np.testing.assert_allclose(
        reconstructed_object_translation,
        TRUE_BASE_TO_OBJECT_TRANSLATION_M,
        atol=1e-10,
    )
    assert (
        rotation_distance(
            reconstructed_object_rotation,
            TRUE_BASE_TO_OBJECT_ROTATION_VECTOR,
        )
        < 1e-10
    )


def test_trajectory_has_multi_axis_rotation_excitation():
    times = np.linspace(0.0, 12.0, 241)
    rotations = np.array(
        [wrist_pose_at_time(time_s)[1] for time_s in times]
    )
    span = np.ptp(rotations, axis=0)

    assert np.all(span > 0.8)


def test_detections_are_intermittent_and_include_long_dropouts():
    data = simulate_wrist_camera_calibration()

    assert 0.30 < (
        len(data["detection_indices"])
        / (len(data["times"]) - 1)
    ) < 0.70

    for start_s, end_s in DETECTION_DROPOUTS_S:
        assert not np.any(
            (data["detection_times"] >= start_s)
            & (data["detection_times"] <= end_s)
        )


def test_iterative_ukf_rts_restarts_from_previous_smoothed_start():
    result = run_wrist_camera_calibration()

    assert result["num_iterations"] == 3
    assert len(result["iteration_runs"]) == 3

    for index in range(1, result["num_iterations"]):
        np.testing.assert_allclose(
            result["iteration_runs"][index]["initial_mean"],
            result["iteration_runs"][index - 1]["smoothed_initial_mean"],
            atol=1e-12,
        )

    # Every forward pass deliberately restarts with the original broad P0,
    # rather than reusing the previous posterior covariance.
    for run in result["iteration_runs"]:
        np.testing.assert_allclose(
            run["initial_covariance"],
            result["initial_covariance"],
            atol=0.0,
        )


def test_iterative_ukf_rts_improves_the_nonlinear_calibration_start():
    result = run_wrist_camera_calibration()

    translation_errors = result[
        "iteration_camera_translation_error_m"
    ]
    rotation_errors = result[
        "iteration_camera_rotation_error_rad"
    ]

    assert translation_errors[-1] < translation_errors[0]
    assert rotation_errors[-1] < rotation_errors[0]


def test_wrist_camera_extrinsics_converge_from_bad_prior():
    result = run_wrist_camera_calibration()

    initial_translation_error_m = np.linalg.norm(
        result["initial_mean"][:3]
        - result["true_state"][:3]
    )
    initial_rotation_error_rad = rotation_distance(
        result["initial_mean"][3:6],
        result["true_state"][3:6],
    )

    assert initial_translation_error_m > 0.015
    assert np.rad2deg(initial_rotation_error_rad) > 4.0

    assert result["filtered_camera_translation_error_m"][-1] < 0.0015
    assert np.rad2deg(
        result["filtered_camera_rotation_error_rad"][-1]
    ) < 0.15

    assert result["filtered_object_translation_error_m"][-1] < 0.0015
    assert np.rad2deg(
        result["filtered_object_rotation_error_rad"][-1]
    ) < 0.15

    # Because the calibration is stationary, smoothing can use later
    # detections to infer what the extrinsics already were near the start.
    assert (
        result["smoothed_camera_translation_error_m"][0]
        < 0.25
        * result["filtered_camera_translation_error_m"][0]
    )
    assert (
        result["smoothed_camera_rotation_error_rad"][0]
        < 0.10
        * result["filtered_camera_rotation_error_rad"][0]
    )

    assert np.isfinite(result["filtered"]).all()
    assert np.isfinite(result["smoothed"]).all()

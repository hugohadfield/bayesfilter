"""Regenerate every figure shown in the examples documentation."""

from pathlib import Path

from examples.active_suspension_road_profile import (
    plot_active_suspension_road_profile,
    run_active_suspension_road_profile,
)
from examples.additional_plotting import (
    plot_house_thermal_estimation,
    plot_vehicle_coastdown,
    plot_vehicle_wheel_calibration,
    plot_wheel_slip_estimation,
)
from examples.aircraft_bank_inference import (
    plot_aircraft_bank_inference,
    run_aircraft_bank_inference,
)
from examples.asteroid_orbit import plot_asteroid_orbit, run_asteroid_orbit_estimation
from examples.ballistic_tracking import run_ballistic_tracking
from examples.battery_plotting import plot_battery_state_of_charge
from examples.battery_state_of_charge import run_battery_state_of_charge
from examples.bearings_only import run_bearings_only_tracking
from examples.binary_packet_channel import (
    plot_binary_packet_channel,
    run_binary_packet_channel,
)
from examples.boat_current_compass_bias import (
    plot_boat_current_compass_bias,
    run_boat_current_compass_bias,
)
from examples.boat_navigation import run_boat_navigation
from examples.boat_plotting import plot_boat_navigation
from examples.bouncing_ball_parameter_inference import (
    plot_bouncing_ball_parameter_inference,
    run_bouncing_ball_parameter_inference,
)
from examples.camera_gyro_calibration import (
    plot_camera_gyro_calibration,
    run_camera_gyro_calibration,
)
from examples.doppler_source_localization import (
    plot_doppler_source_localization,
    run_doppler_source_localization,
)
from examples.gnss_lever_arm_calibration import (
    plot_gnss_lever_arm_calibration,
    run_gnss_lever_arm_calibration,
)
from examples.gravity_vector_from_imu import (
    plot_gravity_vector_from_imu,
    run_gravity_vector_from_imu,
)
from examples.heading_coupled import run_heading_coupled_tracking
from examples.imu_lever_arm_calibration import (
    plot_imu_lever_arm_calibration,
    run_imu_lever_arm_calibration,
)
from examples.ins_gps_fusion import plot_ins_gps_fusion, run_ins_gps_fusion
from examples.known_correspondence_slam import run_known_correspondence_slam
from examples.linear_tracking import run_planar_tracking
from examples.local_global_fusion import (
    plot_local_global_fusion,
    run_local_global_fusion,
)
from examples.orbit_determination import run_orbit_determination
from examples.pendulum_parameter_inference import (
    plot_pendulum_parameter_inference,
    run_pendulum_parameter_inference,
)
from examples.phone_gyrocompass import plot_phone_gyrocompass, run_phone_gyrocompass
from examples.planet_flyby_mass import (
    plot_planet_flyby_mass_estimation,
    run_planet_flyby_mass_estimation,
)
from examples.plotting import (
    plot_ballistic_tracking,
    plot_bearings_only_tracking,
    plot_heading_coupled_tracking,
    plot_inertia_estimation,
    plot_orbit_determination,
    plot_planar_tracking,
    plot_readme_inertia_summary,
    plot_rigid_body_tracking,
    run_and_plot_linear_example,
)
from examples.quantization_noise import (
    plot_quantization_noise_comparison,
    run_quantization_noise_comparison,
)
from examples.radar_camera_fusion import (
    plot_radar_camera_fusion,
    run_radar_camera_fusion,
)
from examples.rigid_body import (
    run_inertia_estimation_rigid_body,
    run_known_inertia_rigid_body,
)
from examples.slam_plotting import plot_known_correspondence_slam
from examples.sloshing_fill_level import (
    plot_sloshing_fill_level,
    run_sloshing_fill_level,
)
from examples.tdoa_emitter_localization import run_tdoa_emitter_localization
from examples.tdoa_plotting import plot_tdoa_emitter_localization
from examples.thermal_house import run_house_thermal_estimation
from examples.vehicle_coastdown import run_vehicle_coastdown
from examples.vehicle_wheel_calibration import run_vehicle_wheel_calibration
from examples.wheel_slip_estimation import run_wheel_slip_estimation
from examples.wrist_camera_hand_eye_calibration import (
    plot_wrist_camera_calibration,
    run_wrist_camera_calibration,
)


def _render(plotter, result, output_path):
    """Render one plot and close plotters that return a live figure."""
    figure = plotter(result, output_path)
    if figure is not None:
        import matplotlib.pyplot as plt

        plt.close(figure)


def main():
    for derivative_order, model_name in (
        (1, "constant-velocity"),
        (2, "constant-acceleration"),
    ):
        for dimension in (1, 2, 3):
            run_and_plot_linear_example(
                dimension,
                derivative_order,
                f"{model_name}-{dimension}d.png",
            )

    figure_directory = Path(__file__).parent / "figures"
    jobs = (
        (plot_planar_tracking, run_planar_tracking, "planar-trajectory-tracking.png"),
        (plot_bearings_only_tracking, run_bearings_only_tracking, "bearings-only-tracking.png"),
        (plot_radar_camera_fusion, run_radar_camera_fusion, "radar-camera-fusion.png"),
        (plot_tdoa_emitter_localization, run_tdoa_emitter_localization, "tdoa-emitter-localization.png"),
        (plot_ballistic_tracking, run_ballistic_tracking, "ballistic-drag-estimation.png"),
        (plot_orbit_determination, run_orbit_determination, "orbit-determination.png"),
        (plot_heading_coupled_tracking, run_heading_coupled_tracking, "heading-coupled-tracking.png"),
        (plot_gravity_vector_from_imu, run_gravity_vector_from_imu, "gravity-vector-from-imu.png"),
        (plot_battery_state_of_charge, run_battery_state_of_charge, "battery-state-of-charge.png"),
        (plot_phone_gyrocompass, run_phone_gyrocompass, "phone-gyrocompass.png"),
        (plot_known_correspondence_slam, run_known_correspondence_slam, "known-correspondence-slam.png"),
        (plot_wrist_camera_calibration, run_wrist_camera_calibration, "wrist-camera-hand-eye-calibration.png"),
        (plot_boat_navigation, run_boat_navigation, "boat-navigation.png"),
        (plot_house_thermal_estimation, run_house_thermal_estimation, "house-thermal-estimation.png"),
        (plot_vehicle_wheel_calibration, run_vehicle_wheel_calibration, "vehicle-wheel-calibration.png"),
        (plot_wheel_slip_estimation, run_wheel_slip_estimation, "wheel-slip-estimation.png"),
        (plot_vehicle_coastdown, run_vehicle_coastdown, "vehicle-coastdown.png"),
        (plot_asteroid_orbit, run_asteroid_orbit_estimation, "asteroid-orbit.png"),
        (plot_planet_flyby_mass_estimation, run_planet_flyby_mass_estimation, "planet-flyby-mass.png"),
        (plot_camera_gyro_calibration, run_camera_gyro_calibration, "camera-gyro-calibration.png"),
        (plot_local_global_fusion, run_local_global_fusion, "local-global-fusion.png"),
        (plot_aircraft_bank_inference, run_aircraft_bank_inference, "aircraft-bank-inference.png"),
        (plot_gnss_lever_arm_calibration, run_gnss_lever_arm_calibration, "gnss-lever-arm-calibration.png"),
        (plot_imu_lever_arm_calibration, run_imu_lever_arm_calibration, "imu-lever-arm-calibration.png"),
        (plot_doppler_source_localization, run_doppler_source_localization, "doppler-source-localization.png"),
        (plot_binary_packet_channel, run_binary_packet_channel, "binary-packet-channel.png"),
        (plot_sloshing_fill_level, run_sloshing_fill_level, "sloshing-fill-level.png"),
        (plot_pendulum_parameter_inference, run_pendulum_parameter_inference, "pendulum-parameter-inference.png"),
        (plot_bouncing_ball_parameter_inference, run_bouncing_ball_parameter_inference, "bouncing-ball-parameter-inference.png"),
        (plot_active_suspension_road_profile, run_active_suspension_road_profile, "active-suspension-road-profile.png"),
        (plot_boat_current_compass_bias, run_boat_current_compass_bias, "boat-current-compass-bias.png"),
        (plot_quantization_noise_comparison, run_quantization_noise_comparison, "quantization-noise-comparison.png"),
        (plot_ins_gps_fusion, run_ins_gps_fusion, "ins-gps-fusion.png"),
    )
    for plotter, runner, filename in jobs:
        print(f"generating {filename}", flush=True)
        _render(plotter, runner(), figure_directory / filename)

    plot_rigid_body_tracking(
        run_known_inertia_rigid_body(),
        figure_directory / "rigid-body-known-inertia.png",
    )
    inertia_result = run_inertia_estimation_rigid_body()
    plot_inertia_estimation(
        inertia_result,
        figure_directory / "rigid-body-inertia-estimation.png",
    )
    plot_readme_inertia_summary(
        inertia_result,
        figure_directory / "readme-rigid-body-inference.png",
    )


if __name__ == "__main__":
    main()

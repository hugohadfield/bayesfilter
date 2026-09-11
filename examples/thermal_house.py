"""Infer house insulation and thermal mass from temperature measurements.

A one-zone RC thermal model is driven by known outdoor temperature and a known
heater command. The hidden state is

    [indoor_temperature, log(thermal_resistance), log(thermal_capacitance)].

Only indoor temperature is measured. The filter therefore performs both state
estimation and online physical-parameter identification. Thermal resistance and
capacitance are represented in log space to keep them positive.

The heater's rated power is treated as known. With only indoor temperature,
outdoor temperature, and an on/off heater command, an unknown heater gain cannot
be identified separately from R and C: the dynamics expose only combinations of
those parameters.
"""

import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.observation import Observation


TRUE_RESISTANCE_K_PER_KW = 2.8
TRUE_CAPACITANCE_KWH_PER_K = 10.5
HEATER_POWER_KW = 5.2
STEP_S = 5.0 * 60.0
DURATION_S = 5.0 * 24.0 * 60.0 * 60.0
OBSERVATION_INTERVAL_STEPS = 3
TEMPERATURE_STD_C = 0.15


def outdoor_temperature(time_s):
    """Synthetic five-day outdoor-temperature profile in degrees Celsius."""
    time_h = np.asarray(time_s) / 3600.0
    daily_cycle = 4.5 * np.sin(2.0 * np.pi * (time_h - 9.0) / 24.0)
    weather_front = 0.5 * np.sin(2.0 * np.pi * time_h / (3.0 * 24.0))
    return 7.0 + daily_cycle + weather_front


def setpoint_temperature(time_s):
    """Return a simple occupied/unoccupied thermostat schedule."""
    hour = (np.asarray(time_s) / 3600.0) % 24.0
    occupied = ((hour >= 6.0) & (hour < 9.0)) | ((hour >= 17.0) & (hour < 23.0))
    return np.where(occupied, 21.0, 17.0)


def propagate_thermal_state(state, delta_t_s, outdoor_c, heater_command):
    """Propagate ``[T_inside, log(R), log(C)]`` through a one-zone RC model."""
    indoor_c, log_resistance, log_capacitance = np.asarray(state)
    resistance = np.exp(log_resistance)
    capacitance = np.exp(log_capacitance)
    delta_t_h = delta_t_s / 3600.0

    decay = np.exp(-delta_t_h / (resistance * capacitance))
    equilibrium_c = outdoor_c + resistance * HEATER_POWER_KW * heater_command
    next_indoor_c = equilibrium_c + (indoor_c - equilibrium_c) * decay
    return np.array([next_indoor_c, log_resistance, log_capacitance])


def process_noise_covariance(delta_t_s):
    """Small random-walk noise for temperature and slowly varying parameters."""
    scale = max(delta_t_s / STEP_S, 1e-9)
    return np.diag(
        [
            0.02**2 * scale,
            0.0001**2 * scale,
            0.0001**2 * scale,
        ]
    )


def make_transition_model(outdoor_c, heater_command, delta_t_s):
    """Create the time-varying thermal transition for one interval."""
    return StateTransitionModel(
        lambda state, dt: propagate_thermal_state(
            state,
            dt,
            outdoor_c,
            heater_command,
        ),
        process_noise_covariance(delta_t_s),
    )


def observe_indoor_temperature(state):
    """Observe indoor temperature only."""
    return np.array([state[0]])


def simulate_house():
    """Simulate thermostat-controlled heating for five days."""
    times = np.arange(0.0, DURATION_S + STEP_S, STEP_S)
    truth = np.empty((len(times), 3))
    truth[0] = np.array(
        [
            19.2,
            np.log(TRUE_RESISTANCE_K_PER_KW),
            np.log(TRUE_CAPACITANCE_KWH_PER_K),
        ]
    )

    heater_commands = np.zeros(len(times) - 1)
    heater_on = False
    for index in range(len(times) - 1):
        setpoint_c = float(setpoint_temperature(times[index]))
        indoor_c = truth[index, 0]
        if heater_on and indoor_c > setpoint_c + 0.25:
            heater_on = False
        elif not heater_on and indoor_c < setpoint_c - 0.25:
            heater_on = True

        heater_commands[index] = float(heater_on)
        midpoint_outdoor_c = float(outdoor_temperature(times[index] + 0.5 * STEP_S))
        truth[index + 1] = propagate_thermal_state(
            truth[index],
            STEP_S,
            midpoint_outdoor_c,
            heater_commands[index],
        )

    return times, truth, heater_commands


def _rts_smooth_time_varying(filtered_states, transition_models):
    """Apply RTS smoothing when each interval has a different known input."""
    smoothed_states = list(filtered_states)
    current_smoothed = filtered_states[-1]

    for index in range(len(filtered_states) - 2, -1, -1):
        predicted_state, cross_covariance = transition_models[
            index
        ].predict_with_cross_covariance(
            filtered_states[index],
            STEP_S,
            use_jacobian=False,
        )
        smoother_gain = cross_covariance @ np.linalg.inv(
            predicted_state.covariance()
        )
        smoothed_mean = filtered_states[index].mean() + smoother_gain @ (
            current_smoothed.mean() - predicted_state.mean()
        )
        smoothed_covariance = filtered_states[index].covariance() + smoother_gain @ (
            current_smoothed.covariance() - predicted_state.covariance()
        ) @ smoother_gain.T
        current_smoothed = Gaussian(smoothed_mean, smoothed_covariance)
        smoothed_states[index] = current_smoothed

    return smoothed_states


def run_house_thermal_estimation(seed=701):
    """Infer house R and C from sparse noisy indoor-temperature measurements."""
    rng = np.random.default_rng(seed)
    times, truth, heater_commands = simulate_house()

    observation_indices = np.arange(
        OBSERVATION_INTERVAL_STEPS,
        len(times),
        OBSERVATION_INTERVAL_STEPS,
    )
    measurements = truth[observation_indices, 0] + rng.normal(
        0.0,
        TEMPERATURE_STD_C,
        len(observation_indices),
    )
    measurement_by_index = {
        int(index): float(measurement)
        for index, measurement in zip(observation_indices, measurements)
    }

    initial_state = Gaussian(
        np.array([18.0, np.log(1.6), np.log(5.5)]),
        np.diag([1.0**2, 0.55**2, 0.55**2]),
    )
    initial_model = make_transition_model(
        float(outdoor_temperature(0.5 * STEP_S)),
        heater_commands[0],
        STEP_S,
    )
    bayes_filter = BayesianFilter(initial_model, initial_state)

    filtered_states = [initial_state]
    transition_models = []
    for index in range(len(times) - 1):
        transition_model = make_transition_model(
            float(outdoor_temperature(times[index] + 0.5 * STEP_S)),
            heater_commands[index],
            STEP_S,
        )
        bayes_filter.transition_model = transition_model
        predicted_state = bayes_filter.predict(
            filtered_states[-1],
            STEP_S,
            use_jacobian=False,
        )

        if index + 1 in measurement_by_index:
            observation = Observation(
                np.array([measurement_by_index[index + 1]]),
                np.array([[TEMPERATURE_STD_C**2]]),
                observe_indoor_temperature,
            )
            predicted_state = bayes_filter.update(
                observation,
                predicted_state,
                use_jacobian=False,
            )

        bayes_filter.set_state(predicted_state)
        filtered_states.append(predicted_state)
        transition_models.append(transition_model)

    smoothed_states = _rts_smooth_time_varying(
        filtered_states,
        transition_models,
    )
    filtered = np.asarray([state.mean() for state in filtered_states])
    smoothed = np.asarray([state.mean() for state in smoothed_states])

    def temperature_rmse(states):
        return float(np.sqrt(np.mean((states[:, 0] - truth[:, 0]) ** 2)))

    return {
        "times": times,
        "truth": truth,
        "heater_commands": heater_commands,
        "outdoor_temperature_c": outdoor_temperature(times),
        "setpoint_temperature_c": setpoint_temperature(times),
        "observation_indices": observation_indices,
        "measurements": measurements,
        "filtered": filtered,
        "smoothed": smoothed,
        "filtered_covariances": np.asarray(
            [state.covariance() for state in filtered_states]
        ),
        "smoothed_covariances": np.asarray(
            [state.covariance() for state in smoothed_states]
        ),
        "filtered_temperature_rmse_c": temperature_rmse(filtered),
        "smoothed_temperature_rmse_c": temperature_rmse(smoothed),
        "filtered_resistance_k_per_kw": np.exp(filtered[:, 1]),
        "filtered_capacitance_kwh_per_k": np.exp(filtered[:, 2]),
        "smoothed_resistance_k_per_kw": np.exp(smoothed[:, 1]),
        "smoothed_capacitance_kwh_per_k": np.exp(smoothed[:, 2]),
    }


def _print_summary(result):
    print("Five-day house thermal-parameter estimation")
    print(
        f"  true R={TRUE_RESISTANCE_K_PER_KW:.2f} K/kW, "
        f"final estimate={result['filtered_resistance_k_per_kw'][-1]:.2f} K/kW"
    )
    print(
        f"  true C={TRUE_CAPACITANCE_KWH_PER_K:.2f} kWh/K, "
        f"final estimate={result['filtered_capacitance_kwh_per_k'][-1]:.2f} kWh/K"
    )
    print(
        f"  indoor-temperature RMSE: "
        f"filter {result['filtered_temperature_rmse_c']:.3f} C, "
        f"RTS {result['smoothed_temperature_rmse_c']:.3f} C"
    )


if __name__ == "__main__":
    _print_summary(run_house_thermal_estimation())

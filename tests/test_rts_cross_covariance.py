import numpy as np

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.model import StateTransitionModel
from bayesfilter.smoothing import RTS


def _linear_transition_model(transition_matrix, process_covariance):
    def transition(state, _delta_t_s):
        return transition_matrix @ state

    def transition_jacobian(_state, _delta_t_s):
        return transition_matrix

    return StateTransitionModel(
        transition,
        process_covariance,
        transition_jacobian,
    )


def test_jacobian_cross_covariance_is_unchanged_for_identity_transition():
    """Control: the suspected bug is invisible when F is the identity."""
    transition_matrix = np.eye(2)
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    model = _linear_transition_model(transition_matrix, 0.1 * np.eye(2))

    _, cross_covariance = model.predict_with_cross_covariance(
        Gaussian(np.array([1.0, 0.5]), covariance),
        delta_t_s=1.0,
        use_jacobian=True,
    )

    np.testing.assert_allclose(
        cross_covariance,
        covariance @ transition_matrix.T,
    )


def test_jacobian_cross_covariance_matches_linear_gaussian_result():
    """For y = F x + w, Cov[x, y] must equal P F.T."""
    transition_matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    process_covariance = np.array([[0.3, 0.05], [0.05, 0.2]])
    state = Gaussian(np.array([1.0, 0.5]), covariance)
    model = _linear_transition_model(transition_matrix, process_covariance)

    predicted_state, cross_covariance = model.predict_with_cross_covariance(
        state,
        delta_t_s=2.0,
        use_jacobian=True,
    )

    expected_predicted_covariance = (
        transition_matrix @ covariance @ transition_matrix.T
        + process_covariance
    )
    expected_cross_covariance = covariance @ transition_matrix.T

    np.testing.assert_allclose(
        predicted_state.covariance(),
        expected_predicted_covariance,
    )
    np.testing.assert_allclose(cross_covariance, expected_cross_covariance)


def test_unscented_cross_covariance_matches_linear_gaussian_result():
    """The unscented branch is a useful independent control for a linear model."""
    transition_matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    process_covariance = np.array([[0.3, 0.05], [0.05, 0.2]])
    state = Gaussian(np.array([1.0, 0.5]), covariance)
    model = _linear_transition_model(transition_matrix, process_covariance)

    predicted_state, cross_covariance = model.predict_with_cross_covariance(
        state,
        delta_t_s=2.0,
        use_jacobian=False,
    )

    np.testing.assert_allclose(
        predicted_state.covariance(),
        transition_matrix @ covariance @ transition_matrix.T
        + process_covariance,
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        cross_covariance,
        covariance @ transition_matrix.T,
        rtol=1e-7,
        atol=1e-7,
    )


def test_jacobian_rts_step_matches_closed_form_linear_smoother():
    """A one-step RTS result must use G = P F.T (F P F.T + Q)^-1."""
    transition_matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    process_covariance = np.array([[0.3, 0.05], [0.05, 0.2]])
    current_mean = np.array([1.0, 0.5])
    current_covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    next_mean = np.array([2.4, 0.2])
    next_covariance = np.array([[0.5, 0.1], [0.1, 0.4]])

    model = _linear_transition_model(transition_matrix, process_covariance)
    bayesian_filter = BayesianFilter(
        model,
        Gaussian(current_mean, current_covariance),
    )
    filter_states = [
        Gaussian(current_mean, current_covariance),
        Gaussian(next_mean, next_covariance),
    ]

    smoothed_states = RTS(bayesian_filter).apply(
        filter_states,
        time_list_s=[0.0, 2.0],
        use_jacobian=True,
    )

    predicted_mean = transition_matrix @ current_mean
    predicted_covariance = (
        transition_matrix @ current_covariance @ transition_matrix.T
        + process_covariance
    )
    smoother_gain = (
        current_covariance
        @ transition_matrix.T
        @ np.linalg.inv(predicted_covariance)
    )
    expected_mean = (
        current_mean + smoother_gain @ (next_mean - predicted_mean)
    )
    expected_covariance = (
        current_covariance
        + smoother_gain
        @ (next_covariance - predicted_covariance)
        @ smoother_gain.T
    )

    np.testing.assert_allclose(smoothed_states[0].mean(), expected_mean)
    np.testing.assert_allclose(
        smoothed_states[0].covariance(),
        expected_covariance,
    )
    np.testing.assert_allclose(smoothed_states[1].mean(), next_mean)
    np.testing.assert_allclose(smoothed_states[1].covariance(), next_covariance)

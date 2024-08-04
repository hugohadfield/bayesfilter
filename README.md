# BayesFilter

BayesFilter is a Python library for Bayesian filtering and smoothing. This library provides tools for implementing Bayesian filters, Rauch-Tung-Striebel smoothers, and other related methods. The only dependency is NumPy.

## Installation

To install BayesFilter, just use `pip`:

```bash
pip install bayesfilter
```

## Usage

### Basic Structure

The library consists of several modules, each responsible for different parts of the Bayesian filtering and smoothing process:

- `distributions.py`: Defines the distribution classes, including the Gaussian distribution used for the filters.
- `filtering.py`: Implements the BayesianFilter class, which runs the filtering process.
- `model.py`: Contains the StateTransitionModel class for state transitions.
- `observation.py`: Defines the Observation class for observation models.
- `smoothing.py`: Implements the RTS (Rauch-Tung-Striebel) smoother.
- `unscented.py`: Provides functions for the unscented transform.
- `utilities.py`: Contains utility functions used throughout the library.
- `test_filtering_smoothing.py`: Contains tests for filtering and smoothing.

### Example

Here is a basic example of how to set up and run a Bayesian filter with the provided library:

1. **Setup Functions**:

```python
def setup_functions():
    def transition_func(x, delta_t_s):
        return np.array([x[0]])
    
    def transition_jacobian_func(x, delta_t_s):
        return np.array([[1.0]])

    def observation_func(x):
        return np.array([np.sin(x[0]), np.cos(x[0])])
    
    def observation_jacobian_func(x):
        return np.array([[np.cos(x[0])], [-np.sin(x[0])]])
    
    return transition_func, transition_jacobian_func, observation_func, observation_jacobian_func
```

2. **Setup Filter and Observations**:

```python
def setup_filter_and_observations():
    rng = np.random.default_rng(0)
    transition_func, transition_jacobian_func, observation_func, observation_jacobian_func = setup_functions()

    transition_model = StateTransitionModel(
        transition_func, 
        1e-8*np.eye(1),
        transition_jacobian_func
    )
    initial_state = Gaussian(np.array([0.0]), np.eye(1))
    filter = BayesianFilter(transition_model, initial_state)

    true_state = np.array([-0.1])
    noise_std = 0.2
    observations = []
    for theta in np.linspace(0, 2*np.pi, 1000):
        observation = observation_func(true_state) + rng.normal(0, noise_std, 2)
        observations.append(Observation(observation, noise_std*np.eye(2), observation_func, observation_jacobian_func))
    return filter, observations, true_state
```

3. **Run Filter**:

```python
def test_filter_noisy_sin(use_jacobian=True):
    filter, observations, true_state = setup_filter_and_observations()
    filter.run(observations, np.linspace(0, 2*np.pi, 1000), 100.0, use_jacobian=use_jacobian)
    np.testing.assert_allclose(filter.state.mean(), true_state, atol=1e-2)
```

### Tests

The library includes a set of tests to ensure the functionality of the filtering and smoothing algorithms. These can be run as follows:

```bash
python test_filtering_smoothing.py
```

## Documentation

### `distributions.py`

Defines the Gaussian distribution class used for state representation and propagation.

### `filtering.py`

Implements the `BayesianFilter` class, responsible for running the filtering process with predict and update steps.

### `model.py`

Contains the `StateTransitionModel` class, representing the state transition model.

### `observation.py`

Defines the `Observation` class, representing the observation model.

### `smoothing.py`

Implements the `RTS` class for Rauch-Tung-Striebel smoothing.

### `unscented.py`

Provides functions for the unscented transform, including `unscented_transform` and `propagate_gaussian`.

### `utilities.py`

Contains utility functions like `propagate_covariance`.

### `test_filtering_smoothing.py`

Includes tests for filtering and smoothing to validate the implementation.

## Author

Hugo Hadfield

## License

This project is licensed under the MIT License.
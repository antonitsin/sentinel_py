# Causal Survey Analyzer

A Python package for analyzing survey responses to estimate causal effects of treatments in experimental and quasi-experimental designs.

## Overview

The `causal_survey_analyzer` package supports:
- A/B tests
- Difference-in-differences (DiD)
- Pre/post analysis
- Synthetic control
- Synthetic DiD

The package validates raw response-to-score encoding, checks for data abnormalities, performs population correction when population data is provided, and visualizes treatment effects per question as lifts with confidence intervals.

## Installation

```bash
pip install causal_survey_analyzer
```

Or install from source:

```bash
git clone https://github.com/antonitsin/causal_survey_analyzer.git
cd causal_survey_analyzer
pip install -e .
```

## Features

- **Data Validation**: Ensures raw responses are correctly encoded into scores and detects data abnormalities.
- **Causal Effect Estimation**: Supports A/B tests, DiD, pre/post, synthetic control, and synthetic DiD to estimate treatment effects.
- **Population Correction**: Adjusts estimates to reflect a target population using post-stratification weights.
- **Visualization**: Displays treatment effects as lifts with 90% CIs, color-coded by significance, with support for covariate breakdowns.
- **Flexibility**: Handles Likert or binary scores and accommodates user-level covariates.

## Usage Examples

### A/B Test Example

```python
import pandas as pd
import matplotlib.pyplot as plt
from causal_survey_analyzer import (
    ResponseValidator,
    DataValidator,
    PostStratificationWeightCalculator,
    CausalEstimator,
    EffectVisualizer
)

# Sample A/B Test Data
ab_data = pd.DataFrame({
    'userid': [1, 1, 2, 2],
    'question': ['q1', 'q2', 'q1', 'q2'],
    'raw_response': ['Agree', 'Disagree', 'Strongly Agree', 'Agree'],
    'score': [4, 2, 5, 4],
    'treatment': [1, 1, 0, 0],
    'age': [25, 25, 30, 30],
    'gender': ['M', 'M', 'F', 'F']
})

# Population data for weighting
pop_data = pd.DataFrame({
    'age': [25, 30],
    'gender': ['M', 'F'],
    'proportion': [0.5, 0.5]
})

# Validate encoding
encoding_map = {'q1': {'Strongly Agree': 5, 'Agree': 4}, 
                'q2': {'Agree': 4, 'Disagree': 2}}
response_validator = ResponseValidator()
mapping = response_validator.validate_encoding(ab_data, 'question', 'raw_response', 'score', encoding_map)

# Check for data issues
data_validator = DataValidator()
issues = data_validator.check_abnormalities(ab_data, 'ab', ['question', 'score', 'treatment'], 'userid', 'treatment')
if issues:
    print("Data issues found:", issues)
else:
    print("No data issues found.")

# Apply population weights
weight_calculator = PostStratificationWeightCalculator()
ab_data['weights'] = weight_calculator.compute_weights(ab_data, pop_data, ['age', 'gender'])

# Run analysis
estimator = CausalEstimator()
results = estimator.run_analysis(
    ab_data, 'ab', 'question', 'score', 'treatment', 
    covariates=['age'], weights_col='weights'
)

# Visualize results
visualizer = EffectVisualizer()
fig = visualizer.visualize_effects(results, 'ab', breakdown_col='gender', data=ab_data)
plt.show()
```

### Difference-in-Differences Example

```python
# DiD Example
did_data = pd.DataFrame({
    'userid': [1, 1, 2, 2],
    'question': ['q1', 'q1', 'q1', 'q1'],
    'time': [0, 1, 0, 1],
    'raw_response': ['Neutral', 'Agree', 'Agree', 'Strongly Agree'],
    'score': [3, 4, 4, 5],
    'treatment': [1, 1, 0, 0],
    'age': [25, 25, 30, 30],
    'gender': ['M', 'M', 'F', 'F']
})

# Run DiD analysis
estimator = CausalEstimator()
results_did = estimator.run_analysis(
    did_data, 'did', 'question', 'score', 'treatment', 'time', 
    covariates=['age']
)

# Visualize DiD results
visualizer = EffectVisualizer()
fig_did = visualizer.visualize_effects(results_did, 'did')
plt.show()
```

## API Documentation

### Data Validation

#### `ResponseValidator.validate_encoding(data, question_col, raw_response_col, score_col, encoding_map=None)`

Verifies that raw responses are correctly mapped to scores for each question.

**Parameters:**
- `data` (pd.DataFrame): Input data.
- `question_col` (str): Column name for question identifier.
- `raw_response_col` (str): Column name for raw responses.
- `score_col` (str): Column name for scores.
- `encoding_map` (dict, optional): Dictionary per question mapping raw responses to scores. If None, infers mapping from data.

**Returns:** 
- dict of validated mappings per question.

#### `DataValidator.check_abnormalities(data, design, required_cols, userid_col, treatment_col, time_col=None)`

Detects potential issues in the data.

**Parameters:**
- `data` (pd.DataFrame): Input data.
- `design` (str): One of 'ab', 'did', 'prepost', 'synth_control', 'synth_did'.
- `required_cols` (list): Columns expected in data.
- `userid_col` (str): Column name for user ID.
- `treatment_col` (str): Column name for treatment indicator.
- `time_col` (str, optional): Column name for time (required for DiD, pre/post, synthetic designs).

**Returns:**
- dict summarizing issues (e.g., missing values, duplicates, inconsistent treatment).

### Analysis

#### `CausalEstimator.run_analysis(data, design, question_col, outcome_col, treatment_col, time_col=None, covariates=None, weights_col=None, synth_pre_periods=None)`

Runs causal effect estimation for the specified design.

**Parameters:**
- `data` (pd.DataFrame): Input data.
- `design` (str): One of 'ab', 'did', 'prepost', 'synth_control', 'synth_did'.
- `question_col` (str): Column name for question identifier.
- `outcome_col` (str): Column name for score.
- `treatment_col` (str): Column name for treatment indicator.
- `time_col` (str, optional): Column name for time (required for DiD, pre/post, synthetic designs).
- `covariates` (list, optional): Covariate column names for adjustment.
- `weights_col` (str, optional): Column name for pre-computed weights.
- `synth_pre_periods` (list, optional): Time points considered pre-treatment for synthetic designs.

**Returns:**
- dict mapping questions to estimation results.

### Population Correction

#### `PostStratificationWeightCalculator.compute_weights(sample_data, population_data, variables)`

Computes post-stratification weights based on population data.

**Parameters:**
- `sample_data` (pd.DataFrame): Survey data.
- `population_data` (pd.DataFrame): Population data with proportions for stratification variables.
- `variables` (list): Stratification variables (e.g., ['age', 'gender']).

**Returns:**
- pd.Series of weights.

### Visualization

#### `EffectVisualizer.visualize_effects(results, design, breakdown_col=None, data=None, confidence_level=0.9)`

Visualizes treatment effects as lifts with confidence intervals.

**Parameters:**
- `results` (dict): Output from CausalEstimator.run_analysis.
- `design` (str): Analysis design (e.g., 'ab', 'did').
- `breakdown_col` (str, optional): Column for subgroup analysis (e.g., gender).
- `data` (pd.DataFrame, optional): Required if breakdown_col is provided.
- `confidence_level` (float, default=0.9): Confidence level for intervals (e.g., 0.9 for 90% CI).

**Returns:**
- matplotlib.figure.Figure: Plot of treatment effects.

## Data Requirements

The package expects data in pandas DataFrame format, structured as follows based on the experimental design:

### Common Columns (All Designs)
- `userid`: Unique user identifier.
- `question`: Question identifier (e.g., q1, q2).
- `raw_response`: Original response (e.g., text like "Strongly Agree" or 1).
- `score`: Encoded score (Likert scale treated as continuous, or binary 0/1).
- `treatment`: Binary indicator (1 = treatment, 0 = control; for pre/post, control may be absent).
- Covariates: age, gender, etc. as needed.
- For DiD, Synthetic DiD, or Pre/Post: `time` (e.g., 0 = pre, 1 = post).

### Population Data (Optional)
Format: pandas DataFrame with proportions for stratification variables (must include a 'proportion' column).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Testing and Validation

The package includes utilities for generating synthetic data with known treatment effects to validate estimator performance:

### Data Generators

The `examples/data_generators.py` module provides functions to generate synthetic data for each estimator type:

```python
from causal_survey_analyzer.examples.data_generators import (
    generate_ab_test_data,
    generate_did_data,
    generate_synthetic_control_data,
    test_estimator_accuracy
)

# Generate A/B test data with known effects
ab_data, true_params = generate_ab_test_data(
    n_treatment=1000,  # 1,000 users in treatment group 
    n_control=1000,    # 1,000 users in control group
    questions=['Q1', 'Q2', 'Q3'],  # Survey questions
    baselines=[3.5, 4.0, 2.8],     # Baseline scores for each question
    treatment_effects=[0.2, 0.5, 0.0],  # True treatment effects
    noise_scale=5.0    # Random noise standard deviation
)

# Generate DiD data with known effects
did_data, true_params = generate_did_data(
    n_treatment=1000,
    n_control=1000,
    pre_periods=14,    # 14 days pre-treatment
    post_periods=14,   # 14 days post-treatment
    questions=['Q1', 'Q2', 'Q3'],
    baselines=[3.5, 4.0, 2.8],
    treatment_effects=[0.2, 0.5, 0.0]
)

# Generate Synthetic Control data with known effects
sc_data, true_params = generate_synthetic_control_data(
    n_treatment_units=5,   # 5 treated units (e.g., regions)
    n_control_units=20,    # 20 control units
    n_users_per_unit=200,  # 200 users per unit on average
    pre_periods=14,
    post_periods=14,
    questions=['Q1', 'Q2', 'Q3'],
    baselines=[3.5, 4.0, 2.8],
    treatment_effects=[0.2, 0.5, 0.0]
)

# Evaluate estimator accuracy
true_effects = true_params['treatment_effects']
estimated_effects = {...}  # From estimator output
metrics = test_estimator_accuracy(true_effects, estimated_effects)
print(f"Mean Absolute Error: {metrics['mae']}")
print(f"Root Mean Squared Error: {metrics['rmse']}")
```

### Estimator Testing

The package includes a test script (`examples/test_estimators.py`) that evaluates the accuracy of each estimator type using synthetic data with known treatment effects:

```bash
# Run the test script
python examples/test_estimators.py
```

This script:
1. Generates synthetic data for each estimator type (A/B test, DiD, Synthetic Control)
2. Runs the corresponding estimator on the data
3. Compares estimated effects to the true effects
4. Calculates accuracy metrics (MAE, RMSE, bias)
5. Displays comparison tables and visualizations
6. Saves summary charts to PNG files

The test script helps validate estimator performance and can be used to ensure the package is working correctly after changes or updates. 
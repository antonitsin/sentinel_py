#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Causal Survey Analyzer: Quickstart Tutorial
# 
# This tutorial demonstrates how to use the `causal_survey_analyzer` package to analyze survey data and estimate causal effects using different experimental designs.

# ## Installation and Setup
# 
# First, let's install the package if you haven't already.
#
# ```
# pip install causal_survey_analyzer
# ```
# 
# Or install from source:
#
# ```
# git clone https://github.com/antonitsin/causal_survey_analyzer.git
# cd causal_survey_analyzer
# pip install -e .
# ```
#
# Now, let's import the necessary modules:

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without display
import matplotlib.pyplot as plt
from causal_survey_analyzer import ResponseValidator, CausalEstimator, EffectVisualizer
from causal_survey_analyzer.estimator import ABTestEstimator, DIDEstimator, SyntheticControlEstimator

# For generating sample data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.data_generators import generate_ab_test_data, generate_did_data, generate_synthetic_control_data

# Create output directory for plots if it doesn't exist
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plots_dir, exist_ok=True)

# ## Part 1: A/B Test Analysis
# 
# In this section, we'll demonstrate how to analyze data from an A/B test using synthetic data with known effects.

# Generate sample A/B test data
np.random.seed(42)  # For reproducibility
ab_data, true_params = generate_ab_test_data(
    n_treatment=500,
    n_control=500,
    questions=None,  # Use default questions
    baselines=None,  # Use default baselines
    treatment_effects=None,  # Use default treatment effects
    response_rate=0.8,
    noise_scale=5.0
)

print("A/B Test Data Sample:")
print(ab_data.head())

# ### Data Validation
#
# Before analyzing the data, we should validate it to ensure it meets our expectations.

# Extract one question for simplicity in this tutorial
question = 'product_satisfaction'
ab_data_filtered = ab_data[ab_data['question'] == question].copy()

# Create a validator to check the data
validator = ResponseValidator(ab_data_filtered)
validation_result = validator.validate_data(
    user_id_col='userid',
    treatment_col='treatment',
    response_col='likert_score',
    treatment_val=1,
    control_val=0,
    min_response=1,
    max_response=5
)

print("\nData Validation Result:", validation_result)

# ### Running A/B Test Analysis

# If validation passes, proceed with analysis
if validation_result:
    # Create an A/B Test estimator
    ab_estimator = ABTestEstimator(
        data=ab_data_filtered,
        user_id_col='userid',
        treatment_col='treatment',
        response_col='likert_score',
        treatment_val=1,
        control_val=0
    )
    
    # Run the analysis
    results = ab_estimator.run_analysis(alpha=0.05)
    
    # Print results
    print("\nA/B Test Results:")
    print(f"Treatment Effect: {results['effect']:.4f}")
    print(f"Standard Error: {results['std_error']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Confidence Interval: [{results['conf_int'][0]:.4f}, {results['conf_int'][1]:.4f}]")
    print(f"True Treatment Effect for {question}: {true_params['treatment_effects'][question]}")
    
    # Create a visualizer
    visualizer = EffectVisualizer(ab_estimator)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    visualizer.plot_effect()
    plt.title('A/B Test Effect Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ab_test_effect.png'))
    plt.close()
    print(f"Plot saved to {os.path.join(plots_dir, 'ab_test_effect.png')}")

# ## Part 2: Difference-in-Differences (DiD) Analysis
# 
# In this section, we'll demonstrate how to analyze data using the Difference-in-Differences approach.
# DiD is a statistical technique that attempts to mimic an experimental research design using observational data.
# It calculates the effect of a treatment by comparing the changes in outcomes over time between a treatment group and a control group.

# Generate sample DiD data
np.random.seed(42)  # For reproducibility
did_data, true_params = generate_did_data(
    n_treatment=250,
    n_control=250,
    pre_periods=5,
    post_periods=5,
    questions=None,  # Use default questions
    baselines=None,  # Use default baselines
    treatment_effects=None,  # Use default treatment effects
    time_trends=None,  # Use default time trends
    response_rate_per_period=0.7,
    noise_scale=5.0
)

print("\nDiD Data Sample:")
print(did_data.head())

# ### Validating DiD Data

# Extract one question for simplicity
question = 'app_usability'
did_data_filtered = did_data[did_data['question'] == question].copy()

# Create a validator to check the data
validator = ResponseValidator(did_data_filtered)
validation_result = validator.validate_did_data(
    user_id_col='userid',
    treatment_col='treatment',
    time_col='time',
    response_col='likert_score',
    treatment_val=1,
    control_val=0,
    pre_val=0,
    post_val=1,
    min_response=1,
    max_response=5
)

print("\nDiD Data Validation Result:", validation_result)

# ### Running DiD Analysis

# If validation passes, proceed with analysis
if validation_result:
    # Create a DiD estimator
    did_estimator = DIDEstimator(
        data=did_data_filtered,
        user_id_col='userid',
        treatment_col='treatment',
        time_col='time',
        response_col='likert_score',
        treatment_val=1,
        control_val=0,
        pre_val=0,
        post_val=1
    )
    
    # Run the analysis
    results = did_estimator.run_analysis(alpha=0.05)
    
    # Print results
    print("\nDiD Analysis Results:")
    print(f"Treatment Effect: {results['effect']:.4f}")
    print(f"Standard Error: {results['std_error']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Confidence Interval: [{results['conf_int'][0]:.4f}, {results['conf_int'][1]:.4f}]")
    print(f"True Treatment Effect for {question}: {true_params['treatment_effects'][question]}")
    
    # Create a visualizer
    visualizer = EffectVisualizer(did_estimator)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    visualizer.plot_did_trends()
    plt.title('DiD Trends Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'did_trends.png'))
    plt.close()
    print(f"Plot saved to {os.path.join(plots_dir, 'did_trends.png')}")

# ## Part 3: Synthetic Control Analysis
# 
# In this section, we'll demonstrate how to analyze data using the Synthetic Control approach.
# The synthetic control method creates a weighted combination of control units to serve as a better counterfactual for the treated unit.
# It's particularly useful when you have a single treated unit and multiple control units.

# Generate sample synthetic control data
np.random.seed(42)  # For reproducibility
sc_data, true_params = generate_synthetic_control_data(
    n_treatment_units=1,
    n_control_units=10,
    n_users_per_unit=100,
    pre_periods=10,
    post_periods=10,
    questions=None,  # Use default questions
    baselines=None,  # Use default baselines
    treatment_effects=None,  # Use default treatment effects
    unit_trends=None,  # Random trends will be generated
    noise_scale=3.0,
    daily_response_probability=0.3
)

print("\nSynthetic Control Data Sample:")
print(sc_data.head())

# ### Preparing Data for Synthetic Control Analysis

# Extract one question for simplicity
question = 'overall_satisfaction'
sc_data_filtered = sc_data[sc_data['question'] == question].copy()

# Group by unit and time, taking the mean response
sc_agg = sc_data_filtered.groupby(['unitid', 'time'])['likert_score'].mean().reset_index()

# ### Running Synthetic Control Analysis

# Create a synthetic control estimator
sc_estimator = SyntheticControlEstimator(
    data=sc_agg,
    unit_col='unitid',
    time_col='time',
    response_col='likert_score',
    treated_unit=1,  # First unit is treated
    treatment_time=10  # Treatment occurs halfway
)

# Run the analysis
results = sc_estimator.run_analysis()

# Print results
print("\nSynthetic Control Results:")
print(f"Average Treatment Effect: {results['effect']:.4f}")
print(f"RMSE of pre-treatment fit: {results['pre_rmse']:.4f}")
print(f"True Treatment Effect for {question}: {true_params['treatment_effects'][question]}")

# Plot the results
plt.figure(figsize=(12, 7))
sc_estimator.plot_synthetic_control()
plt.title('Synthetic Control Analysis')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'synthetic_control.png'))
plt.close()
print(f"Plot saved to {os.path.join(plots_dir, 'synthetic_control.png')}")

# Save all data for reference
ab_data.to_csv(os.path.join(plots_dir, 'ab_test_data.csv'), index=False)
did_data.to_csv(os.path.join(plots_dir, 'did_data.csv'), index=False)
sc_data.to_csv(os.path.join(plots_dir, 'synthetic_control_data.csv'), index=False)
print("All data saved to CSV files in the plots directory")

# ## Using Real Data
#
# In a real-world scenario, you would import your own data instead of generating synthetic data. Here's how you might load your data:
#
# ```python
# your_data = pd.read_csv('path_to_your_data.csv')
#
# # Proceed with validation and analysis as shown above
# validator = ResponseValidator(your_data)
# # ... rest of the analysis
# ```

# ## Conclusion
# 
# In this tutorial, we've demonstrated how to use the `causal_survey_analyzer` package to analyze data using three different causal inference approaches:
# 
# 1. A/B Testing - For comparing outcomes between randomly assigned treatment and control groups
# 2. Difference-in-Differences - For estimating the effect of a treatment by comparing changes over time
# 3. Synthetic Control - For creating a weighted combination of control units as a counterfactual
# 
# These methods allow you to estimate causal effects from survey data under different experimental and quasi-experimental designs. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_survey_analyzer import (
    DataValidator,
    CausalEstimator,
    EffectVisualizer
)
from data_generators import (
    generate_ab_test_data,
    generate_did_data,
    generate_synthetic_control_data,
    test_estimator_accuracy
)

def run_ab_test_evaluation(n_treatment=1000, n_control=1000, noise_scale=5.0, seed=42):
    """
    Test the A/B test estimator with synthetic data.
    
    Parameters:
    -----------
    n_treatment : int
        Number of users in treatment group
    n_control : int
        Number of users in control group
    noise_scale : float
        Standard deviation of random noise added to scores
    seed : int
        Random seed for reproducibility
    """
    print("\n===== A/B Test Estimator Evaluation =====")
    
    # Generate synthetic data
    print(f"Generating synthetic A/B test data with {n_treatment} treatment users, {n_control} control users...")
    ab_data, true_params = generate_ab_test_data(
        n_treatment=n_treatment, 
        n_control=n_control,
        noise_scale=noise_scale,
        seed=seed
    )
    
    # Print data stats
    print(f"Generated {len(ab_data)} observations, {ab_data['userid'].nunique()} unique users")
    print(f"Treatment users: {ab_data[ab_data['treatment']==1]['userid'].nunique()}")
    print(f"Control users: {ab_data[ab_data['treatment']==0]['userid'].nunique()}")
    
    # Create estimator
    estimator = CausalEstimator()
    
    # Run A/B test analysis
    print("Running A/B test analysis...")
    results = estimator.run_analysis(
        ab_data, 'ab', 'question', 'score', 'treatment', 
        covariates=['age', 'gender', 'region']
    )
    
    # Extract true and estimated effects
    true_effects = true_params['treatment_effects']
    estimated_effects = {q: result.params['treatment'] for q, result in results.items()}
    
    # Calculate accuracy metrics
    metrics = test_estimator_accuracy(true_effects, estimated_effects)
    
    # Print comparison of true vs estimated effects
    print("\nTrue vs. Estimated Treatment Effects:")
    comparison = []
    for question in true_effects:
        if question in estimated_effects:
            true_effect = true_effects[question]
            est_effect = estimated_effects[question]
            p_value = results[question].pvalues['treatment']
            error = est_effect - true_effect
            error_percent = (error / true_effect) * 100 if true_effect != 0 else float('inf')
            
            comparison.append({
                'Question': question,
                'True Effect': true_effect,
                'Estimated Effect': round(est_effect, 2),
                'Error': round(error, 2),
                'Error %': f"{error_percent:.1f}%",
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    # Display comparison as DataFrame
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Print accuracy metrics
    print("\nAccuracy Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return results, true_params, metrics


def run_did_evaluation(n_treatment=1000, n_control=1000, pre_periods=14, post_periods=14, 
                       noise_scale=5.0, seed=42):
    """
    Test the DiD estimator with synthetic data.
    
    Parameters:
    -----------
    n_treatment : int
        Number of users in treatment group
    n_control : int
        Number of users in control group
    pre_periods : int
        Number of pre-treatment time periods
    post_periods : int
        Number of post-treatment time periods
    noise_scale : float
        Standard deviation of random noise added to scores
    seed : int
        Random seed for reproducibility
    """
    print("\n===== Difference-in-Differences Estimator Evaluation =====")
    
    # Generate synthetic data
    print(f"Generating synthetic DiD data with {n_treatment} treatment users, {n_control} control users...")
    did_data, true_params = generate_did_data(
        n_treatment=n_treatment,
        n_control=n_control,
        pre_periods=pre_periods,
        post_periods=post_periods,
        noise_scale=noise_scale,
        seed=seed
    )
    
    # Print data stats
    print(f"Generated {len(did_data)} observations, {did_data['userid'].nunique()} unique users")
    print(f"Treatment users: {did_data[did_data['treatment']==1]['userid'].nunique()}")
    print(f"Control users: {did_data[did_data['treatment']==0]['userid'].nunique()}")
    print(f"Pre-treatment periods: {did_data[did_data['time']==0].shape[0]} observations")
    print(f"Post-treatment periods: {did_data[did_data['time']==1].shape[0]} observations")
    
    # Create estimator
    estimator = CausalEstimator()
    
    # Run DiD analysis
    print("Running Difference-in-Differences analysis...")
    results = estimator.run_analysis(
        did_data, 'did', 'question', 'score', 'treatment', 'time',
        covariates=['age', 'gender', 'activity']
    )
    
    # Extract true and estimated effects
    true_effects = true_params['treatment_effects']
    estimated_effects = {q: result.params['treatment_time'] for q, result in results.items()}
    
    # Calculate accuracy metrics
    metrics = test_estimator_accuracy(true_effects, estimated_effects)
    
    # Print comparison of true vs estimated effects
    print("\nTrue vs. Estimated Treatment Effects:")
    comparison = []
    for question in true_effects:
        if question in estimated_effects:
            true_effect = true_effects[question]
            est_effect = estimated_effects[question]
            p_value = results[question].pvalues['treatment_time']
            error = est_effect - true_effect
            error_percent = (error / true_effect) * 100 if true_effect != 0 else float('inf')
            
            comparison.append({
                'Question': question,
                'True Effect': true_effect,
                'Estimated Effect': round(est_effect, 2),
                'Error': round(error, 2),
                'Error %': f"{error_percent:.1f}%",
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    # Display comparison as DataFrame
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Print accuracy metrics
    print("\nAccuracy Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot true effect vs estimated effect
    fig, ax = plt.subplots(figsize=(10, 6))
    questions = list(true_effects.keys())
    true_vals = [true_effects[q] for q in questions]
    est_vals = [estimated_effects[q] for q in questions]
    
    ax.scatter(true_vals, est_vals, s=80, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(true_vals), min(est_vals))
    max_val = max(max(true_vals), max(est_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels
    for i, q in enumerate(questions):
        ax.annotate(q, (true_vals[i], est_vals[i]), fontsize=10)
    
    ax.set_xlabel('True Effect', fontsize=12)
    ax.set_ylabel('Estimated Effect', fontsize=12)
    ax.set_title('DiD Estimator: True vs Estimated Effects', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('did_estimator_evaluation.png')
    print("Evaluation plot saved to 'did_estimator_evaluation.png'")
    
    return results, true_params, metrics


def run_synthetic_control_evaluation(n_treatment_units=5, n_control_units=20, 
                                   n_users_per_unit=200, pre_periods=14, 
                                   post_periods=14, noise_scale=3.0, seed=42):
    """
    Test the Synthetic Control estimator with synthetic data.
    
    Parameters:
    -----------
    n_treatment_units : int
        Number of treated units (e.g., regions, stores)
    n_control_units : int
        Number of control units
    n_users_per_unit : int
        Average number of users per unit
    pre_periods : int
        Number of pre-treatment time periods
    post_periods : int
        Number of post-treatment time periods
    noise_scale : float
        Standard deviation of random noise added to scores
    seed : int
        Random seed for reproducibility
    """
    print("\n===== Synthetic Control Estimator Evaluation =====")
    
    # Generate synthetic data
    print(f"Generating synthetic control data with {n_treatment_units} treatment units, {n_control_units} control units...")
    synth_data, true_params = generate_synthetic_control_data(
        n_treatment_units=n_treatment_units,
        n_control_units=n_control_units,
        n_users_per_unit=n_users_per_unit,
        pre_periods=pre_periods,
        post_periods=post_periods,
        noise_scale=noise_scale,
        seed=seed
    )
    
    # Print data stats
    print(f"Generated {len(synth_data)} observations, {synth_data['userid'].nunique()} unique users")
    print(f"Treatment units: {synth_data[synth_data['treatment']==1]['unitid'].nunique()}")
    print(f"Control units: {synth_data[synth_data['treatment']==0]['unitid'].nunique()}")
    
    # Create estimator
    estimator = CausalEstimator()
    
    # For synthetic control, we need to identify pre_periods
    pre_periods_indices = true_params['pre_periods']
    
    # Run Synthetic Control analysis
    print("Running Synthetic Control analysis...")
    results = estimator.run_analysis(
        synth_data, 'synth_control', 'question', 'score', 'treatment', 'time',
        covariates=['age', 'gender', 'education'],
        synth_pre_periods=pre_periods_indices,
        userid_col='unitid'  # Important: For synthetic control, we use unit IDs as the identifier
    )
    
    # Extract true and estimated effects
    true_effects = true_params['treatment_effects']
    estimated_effects = {q: result['avg_effect'] for q, result in results.items()}
    
    # Calculate accuracy metrics
    metrics = test_estimator_accuracy(true_effects, estimated_effects)
    
    # Print comparison of true vs estimated effects
    print("\nTrue vs. Estimated Treatment Effects:")
    comparison = []
    for question in true_effects:
        if question in estimated_effects:
            true_effect = true_effects[question]
            est_effect = estimated_effects[question]
            error = est_effect - true_effect
            error_percent = (error / true_effect) * 100 if true_effect != 0 else float('inf')
            
            comparison.append({
                'Question': question,
                'True Effect': true_effect,
                'Estimated Effect': round(est_effect, 2),
                'Error': round(error, 2),
                'Error %': f"{error_percent:.1f}%"
            })
    
    # Display comparison as DataFrame
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Print accuracy metrics
    print("\nAccuracy Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return results, true_params, metrics


def main():
    """Run evaluations for all estimator types"""
    # Smaller defaults for quick testing, increase for more realistic tests
    ab_results, ab_params, ab_metrics = run_ab_test_evaluation(
        n_treatment=1000, 
        n_control=1000
    )
    
    did_results, did_params, did_metrics = run_did_evaluation(
        n_treatment=1000, 
        n_control=1000,
        pre_periods=10,
        post_periods=10
    )
    
    synth_results, synth_params, synth_metrics = run_synthetic_control_evaluation(
        n_treatment_units=5,
        n_control_units=20,
        n_users_per_unit=100,
        pre_periods=10,
        post_periods=10
    )
    
    # Print overall summary
    print("\n===== Overall Estimator Accuracy Summary =====")
    summary = {
        'A/B Test': ab_metrics,
        'DiD': did_metrics,
        'Synthetic Control': synth_metrics
    }
    
    # Compare RMSE and Mean Error across methods
    methods = []
    mae_values = []
    rmse_values = []
    bias_values = []
    
    for method, metrics in summary.items():
        methods.append(method)
        mae_values.append(metrics['mae'])
        rmse_values.append(metrics['rmse'])
        bias_values.append(metrics['mean_error'])
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Method': methods,
        'MAE': mae_values,
        'RMSE': rmse_values,
        'Bias': bias_values
    })
    
    print(summary_df.to_string(index=False))
    
    # Create a bar chart comparing methods
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, mae_values, width, label='MAE', color='blue', alpha=0.7)
    ax.bar(x, rmse_values, width, label='RMSE', color='green', alpha=0.7)
    ax.bar(x + width, np.abs(bias_values), width, label='|Bias|', color='red', alpha=0.7)
    
    ax.set_xlabel('Estimation Method')
    ax.set_ylabel('Error Metric')
    ax.set_title('Comparison of Estimator Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('estimator_comparison.png')
    print("Comparison chart saved to 'estimator_comparison.png'")


if __name__ == "__main__":
    main() 
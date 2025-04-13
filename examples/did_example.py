import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from causal_survey_analyzer import (
    DataValidator,
    CausalEstimator,
    EffectVisualizer
)

def main():
    # Create a realistic dataset for DiD analysis
    # ~20k users (10k treatment, 10k control), 2 weeks pre/post data
    np.random.seed(42)  # For reproducibility
    
    # Sample size parameters
    n_users_per_group = 10000
    n_users = n_users_per_group * 2
    
    # Define time periods
    start_date = datetime(2023, 1, 1)
    pre_dates = [(start_date + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(14)]
    post_dates = [(start_date + timedelta(days=d+14)).strftime('%Y-%m-%d') for d in range(14)]
    all_dates = pre_dates + post_dates
    
    # For the model, we need numeric time periods
    # pre = 0, post = 1 for DiD analysis
    pre_indicator = 0
    post_indicator = 1
    
    # Define user demographics
    age_groups = {
        '18-24': 0.18,
        '25-34': 0.26,
        '35-44': 0.22,
        '45-54': 0.16,
        '55-64': 0.12,
        '65+': 0.06
    }
    
    gender_groups = {
        'Male': 0.49,
        'Female': 0.50,
        'Other': 0.01
    }
    
    # Activity levels affect response patterns
    activity_levels = {
        'Low': 0.25,
        'Medium': 0.50,
        'High': 0.25
    }
    
    # Define survey questions
    questions = [
        'app_usability',
        'feature_satisfaction',
        'customer_support',
        'performance_rating',
        'overall_satisfaction'
    ]
    
    # Define baseline scores and treatment effects for each question
    question_baselines = {
        'app_usability': 70,
        'feature_satisfaction': 65,
        'customer_support': 62,
        'performance_rating': 68,
        'overall_satisfaction': 72
    }
    
    # Realistic treatment effects around 5 points
    treatment_effects = {
        'app_usability': 5.2,
        'feature_satisfaction': 4.8,
        'customer_support': 5.5,
        'performance_rating': 6.2,
        'overall_satisfaction': 7.0
    }
    
    # Define demographic effects
    age_effects = {
        '18-24': 2,
        '25-34': 3,
        '35-44': 0,
        '45-54': -1,
        '55-64': -2,
        '65+': -3
    }
    
    gender_effects = {
        'Male': 1,
        'Female': -1,
        'Other': 0
    }
    
    # Natural time trends that would happen anyway (without treatment)
    time_trends = {
        'app_usability': 1.0,  # Natural 1-point increase over time
        'feature_satisfaction': 0.5,
        'customer_support': -0.5,  # Natural decline over time
        'performance_rating': 0.8,
        'overall_satisfaction': 0.3
    }
    
    # Activity levels affect response frequency
    response_probabilities = {
        'Low': 0.5,     # Low activity users respond to 50% of surveys
        'Medium': 0.7,  # Medium activity users respond to 70% of surveys
        'High': 0.9     # High activity users respond to 90% of surveys
    }
    
    # Generate user data
    print("Generating user data...")
    users = []
    for i in range(1, n_users + 1):
        # Assign treatment (first half of users are treated)
        treatment = 1 if i <= n_users_per_group else 0
        
        # Generate demographics
        age = np.random.choice(list(age_groups.keys()), p=list(age_groups.values()))
        gender = np.random.choice(list(gender_groups.keys()), p=list(gender_groups.values()))
        activity = np.random.choice(list(activity_levels.keys()), p=list(activity_levels.values()))
        
        # Add user-specific baseline variation (some users naturally give higher/lower ratings)
        user_baseline_effect = np.random.normal(0, 8)
        
        users.append({
            'userid': i,
            'treatment': treatment,
            'age': age,
            'gender': gender,
            'activity': activity,
            'baseline_effect': user_baseline_effect
        })
    
    # Create user DataFrame
    users_df = pd.DataFrame(users)
    
    # Generate survey responses
    print("Generating survey responses...")
    responses = []
    
    # Add noise to simulate real-world data
    noise_scale = 5.0
    
    for user in users:
        # Determine which dates this user responds to based on activity level
        response_prob = response_probabilities[user['activity']]
        
        # Users typically don't respond to every survey in real life
        user_dates = [date for date in all_dates if np.random.random() < response_prob]
        
        for date in user_dates:
            # Determine if pre or post period
            is_post = date in post_dates
            time_indicator = post_indicator if is_post else pre_indicator
            
            # Add a random day effect (some days have naturally higher/lower scores)
            day_effect = np.random.normal(0, 2)
            
            for question in questions:
                baseline = question_baselines[question]
                
                # Base score with user and demographic effects
                score = baseline + user['baseline_effect']
                score += age_effects[user['age']]
                score += gender_effects[user['gender']]
                score += day_effect
                
                # Add time trend (natural change over time)
                if is_post:
                    score += time_trends[question]
                
                # Add treatment effect for treatment group in post period
                if user['treatment'] == 1 and is_post:
                    score += treatment_effects[question]
                
                # Add noise
                score += np.random.normal(0, noise_scale)
                
                # Ensure score is within valid range (0-100)
                score = max(0, min(100, score))
                
                # Round to nearest integer for realistic scores
                score_rounded = round(score)
                
                # Map to Likert scale for display purposes
                if score_rounded <= 20:
                    likert_response = 'Very Dissatisfied'
                    likert_score = 1
                elif score_rounded <= 40:
                    likert_response = 'Dissatisfied'
                    likert_score = 2
                elif score_rounded <= 60:
                    likert_response = 'Neutral'
                    likert_score = 3
                elif score_rounded <= 80:
                    likert_response = 'Satisfied'
                    likert_score = 4
                else:
                    likert_response = 'Very Satisfied'
                    likert_score = 5
                
                # Add to responses
                responses.append({
                    'userid': user['userid'],
                    'question': question,
                    'date': date,
                    'time': time_indicator,  # Binary pre/post indicator for DiD
                    'raw_response': likert_response,
                    'likert_score': likert_score,
                    'score': score_rounded,
                    'treatment': user['treatment'],
                    'age': user['age'],
                    'gender': user['gender'],
                    'activity': user['activity']
                })
    
    # Create response DataFrame
    did_data = pd.DataFrame(responses)
    
    # Print dataset statistics
    print("\nDiD Dataset Statistics:")
    print(f"Total observations: {len(did_data)}")
    print(f"Unique users: {did_data['userid'].nunique()}")
    print(f"Treatment group users: {did_data[did_data['treatment']==1]['userid'].nunique()}")
    print(f"Control group users: {did_data[did_data['treatment']==0]['userid'].nunique()}")
    print(f"Pre-treatment observations: {len(did_data[did_data['time']==pre_indicator])}")
    print(f"Post-treatment observations: {len(did_data[did_data['time']==post_indicator])}")
    print(f"Questions: {', '.join(questions)}")
    
    # Verify that we have data for both treatment groups in both periods
    counts = did_data.groupby(['treatment', 'time']).size().reset_index(name='count')
    print("\nObservation counts by treatment and time:")
    print(counts)
    
    # Sample of data
    print("\nSample data:")
    print(did_data[['userid', 'question', 'time', 'score', 'treatment', 'age', 'gender']].head())
    
    # Check for data quality issues
    data_validator = DataValidator()
    issues = data_validator.check_abnormalities(
        did_data, 'did', ['question', 'score', 'treatment', 'time'],
        'userid', 'treatment', 'time'
    )
    
    if issues:
        print("\nData issues found:")
        for issue_type, details in issues.items():
            print(f"  - {issue_type}: {details}")
    else:
        print("\nNo data issues found. Data is ready for DiD analysis.")
    
    # Run DiD analysis
    print("\nRunning Difference-in-Differences analysis...")
    estimator = CausalEstimator()
    results = estimator.run_analysis(
        did_data, 'did', 'question', 'score', 'treatment', 'time',
        covariates=['age', 'gender', 'activity']
    )
    
    # Create results summary
    print("\nResults Summary:")
    summary_data = []
    
    for question, result in results.items():
        # Extract the treatment*time interaction coefficient
        effect = result.params['treatment_time']
        p_value = result.pvalues['treatment_time']
        ci_low, ci_high = result.conf_int(alpha=0.1).loc['treatment_time']
        
        # Format question name for display
        question_display = question.replace('_', ' ').title()
        
        # Calculate percentage improvement
        baseline = question_baselines[question]
        percent_improvement = (effect / baseline) * 100
        
        # Calculate true treatment effect (for comparison)
        true_effect = treatment_effects[question]
        
        # Add to summary data
        summary_data.append({
            'Question': question_display,
            'Baseline': baseline,
            'True Effect': true_effect,
            'Estimated Effect': round(effect, 2),
            'Difference': round(effect - true_effect, 2),
            'Percent Change': f"{percent_improvement:.2f}%",
            'p-value': p_value,
            '90% CI': f"({ci_low:.2f}, {ci_high:.2f})",
            'Significant': 'Yes' if p_value < 0.1 else 'No'
        })
        
        # Print individual results
        print(f"\nQuestion: {question_display}")
        print(f"True treatment effect: {true_effect:.2f} points")
        print(f"Estimated treatment effect: {effect:.2f} points (SE: {result.bse['treatment_time']:.2f})")
        print(f"Estimation error: {effect - true_effect:.2f} points")
        print(f"Percent change: {percent_improvement:.2f}%")
        print(f"p-value: {p_value:.4f}" + (" (Significant)" if p_value < 0.1 else ""))
        print(f"90% Confidence Interval: ({ci_low:.2f}, {ci_high:.2f})")
    
    # Create and print summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print("\nOverall DiD Analysis Summary:")
    print(summary_df[['Question', 'True Effect', 'Estimated Effect', 'Difference', 'Percent Change', 'Significant']].to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualizer = EffectVisualizer()
    
    # Overall effects plot
    fig = visualizer.visualize_effects(results, 'did')
    plt.tight_layout()
    plt.savefig('did_results.png')
    print("Overall DiD results plot saved to 'did_results.png'")
    
    # Breakdown by demographics
    fig_gender = visualizer.visualize_effects(results, 'did', breakdown_col='gender', data=did_data)
    plt.tight_layout()
    plt.savefig('did_results_by_gender.png')
    print("Gender breakdown plot saved to 'did_results_by_gender.png'")
    
    fig_age = visualizer.visualize_effects(results, 'did', breakdown_col='age', data=did_data)
    plt.tight_layout()
    plt.savefig('did_results_by_age.png')
    print("Age breakdown plot saved to 'did_results_by_age.png'")
    
    # Create mean score plots by group and time
    print("\nCreating trend plots by treatment group...")
    
    # Calculate mean scores by question, treatment, and time
    mean_scores = did_data.groupby(['question', 'treatment', 'time'])['score'].mean().reset_index()
    
    # Plot trends for each question
    for question in questions:
        question_data = mean_scores[mean_scores['question'] == question]
        
        # Format for plotting
        question_display = question.replace('_', ' ').title()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot treatment group
        treatment_data = question_data[question_data['treatment'] == 1]
        ax.plot([pre_indicator, post_indicator], treatment_data['score'], 'b-o', linewidth=2, markersize=8, label='Treatment Group')
        
        # Plot control group
        control_data = question_data[question_data['treatment'] == 0]
        ax.plot([pre_indicator, post_indicator], control_data['score'], 'r--x', linewidth=2, markersize=8, label='Control Group')
        
        # Calculate the treatment effect
        pre_diff = treatment_data[treatment_data['time'] == pre_indicator]['score'].values[0] - \
                   control_data[control_data['time'] == pre_indicator]['score'].values[0]
                   
        post_diff = treatment_data[treatment_data['time'] == post_indicator]['score'].values[0] - \
                    control_data[control_data['time'] == post_indicator]['score'].values[0]
                    
        did_effect = post_diff - pre_diff
        
        # Add DiD effect annotation
        ax.text(0.5, 0.05, f'DiD Effect: {did_effect:.2f} points', 
                transform=ax.transAxes, fontsize=12, ha='center',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add styling
        ax.set_title(f'{question_display}: Treatment vs Control Group Over Time', fontsize=14)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Average Score (0-100)', fontsize=12)
        ax.set_xticks([pre_indicator, post_indicator])
        ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'did_trend_{question}.png')
        print(f"Trend plot saved to did_trend_{question}.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
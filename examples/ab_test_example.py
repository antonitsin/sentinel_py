import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from causal_survey_analyzer import (
    ResponseValidator,
    DataValidator,
    PostStratificationWeightCalculator,
    CausalEstimator,
    EffectVisualizer
)

def main():
    # Create realistic A/B Test data with ~20k users (10k treatment, 10k control)
    np.random.seed(42)  # For reproducibility
    
    # Define sample size
    n_users_per_group = 10000
    n_users = n_users_per_group * 2
    
    # Define demographics for stratification
    age_groups = {
        '18-24': 0.15,
        '25-34': 0.25,
        '35-44': 0.20,
        '45-54': 0.18,
        '55-64': 0.12,
        '65+': 0.10
    }
    
    gender_groups = {
        'Male': 0.48,
        'Female': 0.51,
        'Other': 0.01
    }
    
    region_groups = {
        'North': 0.25,
        'South': 0.30,
        'East': 0.20,
        'West': 0.25
    }
    
    # Generate user data with demographics
    user_data = []
    for i in range(1, n_users + 1):
        # Assign treatment - first half of users are in treatment group
        treatment = 1 if i <= n_users_per_group else 0
        
        # Generate demographics
        age = np.random.choice(list(age_groups.keys()), p=list(age_groups.values()))
        gender = np.random.choice(list(gender_groups.keys()), p=list(gender_groups.values()))
        region = np.random.choice(list(region_groups.keys()), p=list(region_groups.values()))
        
        # Add user
        user_data.append({
            'userid': i,
            'treatment': treatment,
            'age': age,
            'gender': gender,
            'region': region
        })
    
    # Create DataFrame for users
    users_df = pd.DataFrame(user_data)
    
    # Define survey questions
    questions = [
        'product_satisfaction',
        'user_experience',
        'customer_service',
        'value_for_money',
        'likelihood_to_recommend'
    ]
    
    # Define question baselines and treatment effects
    question_baselines = {
        'product_satisfaction': 72,
        'user_experience': 68,
        'customer_service': 65,
        'value_for_money': 60,
        'likelihood_to_recommend': 70
    }
    
    # Treatment effects around 5 points as specified
    treatment_effects = {
        'product_satisfaction': 5.2,
        'user_experience': 4.7,
        'customer_service': 6.1,
        'value_for_money': 4.5,
        'likelihood_to_recommend': 5.8
    }
    
    # Define effects of demographic factors on scores
    age_effects = {
        '18-24': -2,
        '25-34': 0,
        '35-44': 1,
        '45-54': 2,
        '55-64': 1,
        '65+': -1
    }
    
    gender_effects = {
        'Male': 1,
        'Female': -1,
        'Other': 0
    }
    
    region_effects = {
        'North': 2,
        'South': -1,
        'East': 0,
        'West': -1
    }
    
    # Create response data
    response_data = []
    
    # Not all users respond to all questions - define response rates
    response_rate = 0.85  # 85% of users respond
    
    # Add noise to simulate real survey data
    noise_scale = 5.0
    
    print("Generating realistic A/B test data...")
    for user in user_data:
        # Determine if user responds based on response rate
        if np.random.random() < response_rate:
            # Add base user variability
            user_effect = np.random.normal(0, 8)  # Random user effect
            
            # Generate response for each question
            for question in questions:
                baseline = question_baselines[question]
                
                # Base score with demographic effects
                score = baseline + user_effect
                score += age_effects[user['age']]
                score += gender_effects[user['gender']]
                score += region_effects[user['region']]
                
                # Add treatment effect
                if user['treatment'] == 1:
                    score += treatment_effects[question]
                
                # Add noise
                score += np.random.normal(0, noise_scale)
                
                # Ensure score is between 0 and 100
                score = max(0, min(100, score))
                
                # Round to nearest integer for realistic scores
                score_rounded = round(score)
                
                # Map to Likert scale for display
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
                
                # Add to response data
                response_data.append({
                    'userid': user['userid'],
                    'question': question,
                    'raw_response': likert_response,
                    'likert_score': likert_score,
                    'score': score_rounded,
                    'treatment': user['treatment'],
                    'age': user['age'],
                    'gender': user['gender'],
                    'region': user['region']
                })
    
    # Create the full DataFrame
    ab_data = pd.DataFrame(response_data)
    
    # Print data stats
    print("\nA/B Test Dataset Statistics:")
    print(f"Total responses: {len(ab_data)}")
    print(f"Unique users: {ab_data['userid'].nunique()}")
    print(f"Treatment group size: {ab_data[ab_data['treatment']==1]['userid'].nunique()}")
    print(f"Control group size: {ab_data[ab_data['treatment']==0]['userid'].nunique()}")
    print(f"Questions: {', '.join(questions)}")
    
    print("\nSample data (first few rows):")
    print(ab_data[['userid', 'question', 'score', 'treatment', 'age', 'gender']].head())
    
    # Create response validator and data validator
    response_validator = ResponseValidator()
    data_validator = DataValidator()
    
    # Define Likert scale encoding map for validation
    encoding_map = {q: {
        'Very Dissatisfied': 1,
        'Dissatisfied': 2,
        'Neutral': 3,
        'Satisfied': 4,
        'Very Satisfied': 5
    } for q in questions}
    
    # Validate response encoding
    try:
        mapping = response_validator.validate_encoding(ab_data, 'question', 'raw_response', 'likert_score', encoding_map)
        print("\nResponse encoding validation successful.")
    except ValueError as e:
        print(f"\nEncoding validation error: {e}")
    
    # Check for data abnormalities
    issues = data_validator.check_abnormalities(
        ab_data, 'ab', ['question', 'score', 'treatment'], 'userid', 'treatment'
    )
    
    if issues:
        print("\nData issues found:")
        for issue_type, details in issues.items():
            print(f"  - {issue_type}: {details}")
    else:
        print("\nNo data issues found. Data is ready for analysis.")
    
    # Population data for weighting to correct for demographic imbalances
    # In a real scenario, this would come from census or other population data
    pop_data_age_gender = []
    
    # Create realistic population proportions
    for age in age_groups.keys():
        for gender in gender_groups.keys():
            # Add some random variation to make it different from sample
            pop_proportion = age_groups[age] * gender_groups[gender] * (1 + np.random.uniform(-0.1, 0.1))
            pop_data_age_gender.append({
                'age': age,
                'gender': gender,
                'proportion': pop_proportion
            })
    
    # Normalize proportions to sum to 1
    pop_df = pd.DataFrame(pop_data_age_gender)
    pop_df['proportion'] = pop_df['proportion'] / pop_df['proportion'].sum()
    
    # Initialize weight calculator
    weight_calculator = PostStratificationWeightCalculator()
    
    # Compute post-stratification weights
    print("\nCalculating post-stratification weights to correct for demographic imbalances...")
    ab_data['weights'] = weight_calculator.compute_weights(ab_data, pop_df, ['age', 'gender'])
    
    # Print weight summary
    weight_summary = ab_data.groupby(['age', 'gender'])['weights'].mean().reset_index()
    print("\nPost-stratification weights by demographic group:")
    print(weight_summary)
    
    # Initialize estimator
    estimator = CausalEstimator()
    
    # Run causal analysis
    print("\nRunning A/B test analysis...")
    results = estimator.run_analysis(
        ab_data, 'ab', 'question', 'score', 'treatment', 
        covariates=['age', 'gender', 'region'], weights_col='weights'
    )
    
    # Print results summary
    print("\nResults Summary:")
    
    # Create summary table data
    summary_data = []
    
    for question, result in results.items():
        # Extract treatment effect
        effect = result.params['treatment']
        p_value = result.pvalues['treatment']
        ci_low, ci_high = result.conf_int(alpha=0.1).loc['treatment']
        
        # Format question name for display
        question_display = question.replace('_', ' ').title()
        
        # Calculate percentage improvement
        baseline = question_baselines[question]
        percent_improvement = (effect / baseline) * 100
        
        # Add to summary data
        summary_data.append({
            'Question': question_display,
            'Baseline': baseline,
            'Treatment Effect': round(effect, 2),
            'Percent Improvement': f"{percent_improvement:.2f}%",
            'p-value': p_value,
            '90% CI': f"({ci_low:.2f}, {ci_high:.2f})",
            'Significant': 'Yes' if p_value < 0.1 else 'No'
        })
        
        # Also print individual model summaries
        print(f"\nQuestion: {question_display}")
        print(f"Treatment effect: {effect:.2f} points (SE: {result.bse['treatment']:.2f})")
        print(f"Percent improvement: {percent_improvement:.2f}%")
        print(f"p-value: {p_value:.4f}" + (" (Significant)" if p_value < 0.1 else ""))
        print(f"90% Confidence Interval: ({ci_low:.2f}, {ci_high:.2f})")
    
    # Create and print summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print("\nOverall Impact Summary:")
    print(summary_df[['Question', 'Treatment Effect', 'Percent Improvement', 'p-value', 'Significant']].to_string(index=False))
    
    # Initialize visualizer
    visualizer = EffectVisualizer()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Overall effects
    fig = visualizer.visualize_effects(results, 'ab')
    plt.tight_layout()
    plt.savefig('ab_test_results.png')
    print("Overall results plot saved to 'ab_test_results.png'")
    
    # Breakdown by gender
    fig_gender = visualizer.visualize_effects(results, 'ab', breakdown_col='gender', data=ab_data)
    plt.tight_layout()
    plt.savefig('ab_test_results_by_gender.png')
    print("Gender breakdown plot saved to 'ab_test_results_by_gender.png'")
    
    # Breakdown by age group
    fig_age = visualizer.visualize_effects(results, 'ab', breakdown_col='age', data=ab_data)
    plt.tight_layout()
    plt.savefig('ab_test_results_by_age.png')
    print("Age breakdown plot saved to 'ab_test_results_by_age.png'")
    
    # Breakdown by region
    fig_region = visualizer.visualize_effects(results, 'ab', breakdown_col='region', data=ab_data)
    plt.tight_layout()
    plt.savefig('ab_test_results_by_region.png')
    print("Region breakdown plot saved to 'ab_test_results_by_region.png'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
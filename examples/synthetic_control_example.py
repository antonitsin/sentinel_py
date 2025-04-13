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
    # Create a more realistic sample for Synthetic Control analysis
    # We'll have 20,000 users (10k treated, 10k control), with 28 days of data (14 pre, 14 post)
    np.random.seed(42)  # For reproducibility
    
    # Generate user IDs and treatments
    n_users = 20000
    n_treated = 10000
    user_ids = list(range(1, n_users + 1))
    treatments = [1] * n_treated + [0] * (n_users - n_treated)  # First half are treated
    
    # Generate demographics (covariates)
    age_groups = {
        '18-24': 0.15,
        '25-34': 0.25,
        '35-44': 0.20,
        '45-54': 0.18,
        '55-64': 0.12,
        '65+': 0.10
    }
    ages = np.random.choice(list(age_groups.keys()), size=n_users, p=list(age_groups.values()))
    
    gender_groups = {
        'M': 0.48,
        'F': 0.51,
        'Other': 0.01
    }
    genders = np.random.choice(list(gender_groups.keys()), size=n_users, p=list(gender_groups.values()))
    
    # Set education levels
    education_groups = {
        'High school': 0.25,
        'Some college': 0.20,
        'Bachelor': 0.35,
        'Graduate': 0.20
    }
    educations = np.random.choice(list(education_groups.keys()), size=n_users, p=list(education_groups.values()))
    
    # Define time periods: 14 days pre-treatment, 14 days post-treatment
    # Using actual dates to be more realistic
    start_date = datetime(2023, 1, 1)
    pre_dates = [(start_date + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(14)]
    post_dates = [(start_date + timedelta(days=d+14)).strftime('%Y-%m-%d') for d in range(14)]
    all_dates = pre_dates + post_dates
    
    # Map dates to time periods for the model
    date_to_period = {date: i for i, date in enumerate(all_dates)}
    pre_periods = list(range(len(pre_dates)))
    post_periods = list(range(len(pre_dates), len(all_dates)))
    
    # Questions - typical satisfaction survey questions
    questions = [
        'overall_satisfaction',
        'ease_of_use',
        'customer_support',
        'value_for_money',
        'likelihood_to_recommend'
    ]
    
    # Create the dataframe
    rows = []
    
    # Define the data generating process
    # Baselines for different questions (on a 100-point scale)
    question_baselines = {
        'overall_satisfaction': 75,
        'ease_of_use': 70,
        'customer_support': 65,
        'value_for_money': 60,
        'likelihood_to_recommend': 72
    }
    
    # Treatment effect (around 5 points as requested)
    treatment_effects = {
        'overall_satisfaction': 5.2,
        'ease_of_use': 4.8,
        'customer_support': 5.5,
        'value_for_money': 4.6,
        'likelihood_to_recommend': 6.0
    }
    
    # Add different trend patterns for different user segments
    # This simulates how different demographic groups might respond differently over time
    age_trends = {
        '18-24': 0.1,
        '25-34': 0.05,
        '35-44': 0.02,
        '45-54': -0.02,
        '55-64': -0.05,
        '65+': -0.1
    }
    
    gender_trends = {
        'M': 0.03,
        'F': -0.03,
        'Other': 0.0
    }
    
    education_trends = {
        'High school': -0.05,
        'Some college': 0.0,
        'Bachelor': 0.05,
        'Graduate': 0.10
    }
    
    # Base variation between different users
    user_random_effects = np.random.normal(0, 5, n_users)
    
    # Day-of-week effects (weekend vs weekday)
    # These create realistic cyclical patterns in the data
    day_of_week_effects = {
        0: 2,  # Monday
        1: 1,  # Tuesday
        2: 0,  # Wednesday
        3: -1,  # Thursday
        4: -2,  # Friday
        5: 3,  # Saturday
        6: 2   # Sunday
    }
    
    # Add random noise (measurement error)
    noise_scale = 3.0
    
    # This factor represents how many users actually respond on a given day (response rate)
    # In real surveys, not every user responds every day
    daily_response_probability = 0.3
    
    print("Generating realistic synthetic data...")
    # Generate data for all users across all time periods
    for i, user_id in enumerate(user_ids):
        treatment = treatments[i]
        age = ages[i]
        gender = genders[i]
        education = educations[i]
        user_effect = user_random_effects[i]
        
        # Only some users respond on each date
        response_dates = [date for date in all_dates if np.random.random() < daily_response_probability]
        
        for date in response_dates:
            # Get time period index and day of week
            t = date_to_period[date]
            day_of_week = datetime.strptime(date, '%Y-%m-%d').weekday()
            
            # For each question
            for q in questions:
                # Base score (0-100 scale, more realistic for customer surveys)
                baseline = question_baselines[q]
                
                # Add user-specific base effect
                score = baseline + user_effect
                
                # Add demographic trends
                score += age_trends[age] * t
                score += gender_trends[gender] * t
                score += education_trends[education] * t
                
                # Add day-of-week effects
                score += day_of_week_effects[day_of_week]
                
                # Add treatment effect for treated units in post-treatment periods
                if treatment == 1 and t >= len(pre_dates):
                    # Gradual treatment effect that increases over time
                    days_since_treatment = t - len(pre_dates)
                    ramp_up_factor = min(1.0, days_since_treatment / 5)  # Full effect after 5 days
                    score += treatment_effects[q] * ramp_up_factor
                
                # Add noise
                score += np.random.normal(0, noise_scale)
                
                # Ensure score is between 0 and 100
                score = max(0, min(100, score))
                
                # Round to nearest integer for realistic survey scores
                score_rounded = round(score)
                
                # Map score to 5-point Likert scale for display purposes
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
                
                rows.append({
                    'userid': user_id,
                    'question': q,
                    'date': date,
                    'time': t,
                    'raw_response': likert_response,
                    'likert_score': likert_score,
                    'score': score_rounded,  # Keep the 0-100 score for analysis
                    'treatment': treatment,
                    'age': age,
                    'gender': gender,
                    'education': education
                })
    
    # Create the DataFrame
    synth_data = pd.DataFrame(rows)
    
    # Print data stats
    print("\nSynthetic Data Statistics:")
    print(f"Total observations: {len(synth_data)}")
    print(f"Unique users: {synth_data['userid'].nunique()}")
    print(f"Treatment group size: {synth_data[synth_data['treatment']==1]['userid'].nunique()}")
    print(f"Control group size: {synth_data[synth_data['treatment']==0]['userid'].nunique()}")
    print(f"Time periods: {synth_data['time'].nunique()}")
    print(f"Date range: {synth_data['date'].min()} to {synth_data['date'].max()}")
    
    print("\nSample data (first few rows):")
    print(synth_data[['userid', 'question', 'date', 'time', 'score', 'treatment']].head())
    
    # Initialize validators and estimators
    data_validator = DataValidator()
    estimator = CausalEstimator()
    visualizer = EffectVisualizer()
    
    # Check for data abnormalities
    issues = data_validator.check_abnormalities(
        synth_data, 'synth_control', ['question', 'score', 'treatment', 'time'],
        'userid', 'treatment', 'time'
    )
    
    if issues:
        print("\nData issues found:")
        for issue_type, details in issues.items():
            print(f"  - {issue_type}: {details}")
    else:
        print("\nNo data issues found. Data is ready for analysis.")
    
    # Run Synthetic Control analysis
    print("\nRunning synthetic control analysis...")
    results = estimator.run_analysis(
        synth_data, 'synth_control', 'question', 'score', 'treatment', 'time',
        covariates=['age', 'gender', 'education'], 
        synth_pre_periods=pre_periods,
        userid_col='userid'
    )
    
    # Print results summary
    print("\nResults Summary:")
    summary_data = []
    for question, result in results.items():
        print(f"\nQuestion: {question}")
        effect = result['avg_effect']
        print(f"Average treatment effect: {effect:.2f} points")
        
        # Calculate percent improvement relative to question baseline
        baseline = question_baselines[question]
        percent_improvement = (effect / baseline) * 100
        print(f"Percent improvement: {percent_improvement:.2f}%")
        
        # Record for summary table
        summary_data.append({
            'Question': question.replace('_', ' ').title(),
            'Baseline': baseline,
            'Effect': effect,
            'Improvement': f"{percent_improvement:.2f}%"
        })
    
    # Print a nice summary table
    summary_df = pd.DataFrame(summary_data)
    print("\nOverall Impact Summary:")
    print(summary_df.to_string(index=False))
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # Plot for each question
    for question, result in results.items():
        unit_effects = result['unit_effects']
        if not unit_effects:
            print(f"No unit effects data for {question}, skipping plot")
            continue
            
        # Get data for the first treated unit as an example
        first_unit = list(unit_effects.keys())[0]
        unit_data = unit_effects[first_unit]
        
        # Prepare data for plotting
        observed = unit_data['observed']
        synthetic = unit_data['synthetic']
        
        # Convert to DataFrame for easier plotting
        plot_data = pd.DataFrame({
            'Period': list(observed.keys()),
            'Observed': list(observed.values()),
            'Synthetic': list(synthetic.values())
        })
        plot_data = plot_data.sort_values('Period')
        
        # Add dates for x-axis labels
        plot_data['Date'] = [all_dates[p] for p in plot_data['Period']]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines
        ax.plot(plot_data['Date'], plot_data['Observed'], 'b-', marker='o', label='Treated')
        ax.plot(plot_data['Date'], plot_data['Synthetic'], 'r--', marker='x', label='Synthetic Control')
        
        # Add vertical line at treatment start
        treatment_start_date = all_dates[len(pre_dates)]
        ax.axvline(x=treatment_start_date, color='k', linestyle='--', alpha=0.7, label='Treatment Start')
        
        # Calculate shaded area for treatment effect
        plot_data['Effect'] = plot_data['Observed'] - plot_data['Synthetic']
        post_data = plot_data[plot_data['Period'] >= len(pre_dates)]
        
        # Add effect size text annotation
        avg_effect = post_data['Effect'].mean()
        ax.text(0.7, 0.05, f'Avg. Effect: {avg_effect:.2f} points', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Styling
        title = question.replace('_', ' ').title()
        ax.set_title(f'{title}: Treated vs Synthetic Control', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust ylim to focus on the relevant part of the data
        min_y = min(plot_data['Observed'].min(), plot_data['Synthetic'].min()) - 5
        max_y = max(plot_data['Observed'].max(), plot_data['Synthetic'].max()) + 5
        ax.set_ylim(min_y, max_y)
        
        plt.tight_layout()
        plt.savefig(f'synthetic_control_{question}.png')
        print(f"Plot saved to synthetic_control_{question}.png")
    
    # Also create standard effect visualization for all questions
    fig_effects = visualizer.visualize_effects(results, 'synth_control')
    plt.tight_layout()
    plt.savefig('synthetic_control_effects.png')
    print("Effects summary plot saved to 'synthetic_control_effects.png'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
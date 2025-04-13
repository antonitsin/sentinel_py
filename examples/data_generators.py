import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ab_test_data(
    n_treatment=10000, 
    n_control=10000, 
    questions=None,
    baselines=None,
    treatment_effects=None,
    response_rate=0.85,
    noise_scale=5.0,
    seed=42
):
    """
    Generate synthetic A/B test data with known effects.
    
    Parameters:
    -----------
    n_treatment : int
        Number of users in treatment group
    n_control : int
        Number of users in control group
    questions : list
        List of question names. If None, default questions will be used.
    baselines : dict
        Dict mapping questions to baseline scores. If None, defaults will be used.
    treatment_effects : dict
        Dict mapping questions to treatment effects. If None, defaults will be used.
    response_rate : float
        Proportion of users who respond to the survey (0-1)
    noise_scale : float
        Standard deviation of random noise added to scores
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic A/B test data
    dict
        Dict with true parameter values used to generate the data
    """
    np.random.seed(seed)
    
    # Default questions if not specified
    if questions is None:
        questions = [
            'product_satisfaction',
            'user_experience',
            'customer_service',
            'value_for_money',
            'likelihood_to_recommend'
        ]
    
    # Default baselines if not specified
    if baselines is None:
        baselines = {
            'product_satisfaction': 72,
            'user_experience': 68,
            'customer_service': 65,
            'value_for_money': 60,
            'likelihood_to_recommend': 70
        }
        # Fill in any missing questions with default value of 70
        for q in questions:
            if q not in baselines:
                baselines[q] = 70
    
    # Default treatment effects if not specified
    if treatment_effects is None:
        treatment_effects = {
            'product_satisfaction': 5.2,
            'user_experience': 4.7,
            'customer_service': 6.1,
            'value_for_money': 4.5,
            'likelihood_to_recommend': 5.8
        }
        # Fill in any missing questions with default effect of 5.0
        for q in questions:
            if q not in treatment_effects:
                treatment_effects[q] = 5.0
    
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
    
    # Demographic effects
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
    
    # Total users
    n_users = n_treatment + n_control
    
    # Generate user data
    user_data = []
    for i in range(1, n_users + 1):
        # Assign treatment - first users are in treatment group
        treatment = 1 if i <= n_treatment else 0
        
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
    
    # Generate response data
    response_data = []
    
    for user in user_data:
        # Determine if user responds based on response rate
        if np.random.random() < response_rate:
            # Add base user variability
            user_effect = np.random.normal(0, 8)  # Random user effect
            
            # Generate response for each question
            for question in questions:
                baseline = baselines[question]
                
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
    
    # Return data and true parameters
    true_params = {
        'baselines': baselines,
        'treatment_effects': treatment_effects,
        'demographics': {
            'age_effects': age_effects,
            'gender_effects': gender_effects,
            'region_effects': region_effects
        }
    }
    
    return ab_data, true_params


def generate_did_data(
    n_treatment=10000,
    n_control=10000,
    pre_periods=14,
    post_periods=14,
    questions=None,
    baselines=None,
    treatment_effects=None,
    time_trends=None,
    response_rate_per_period=0.7,
    noise_scale=5.0,
    seed=42
):
    """
    Generate synthetic Difference-in-Differences data with known effects.
    
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
    questions : list
        List of question names. If None, default questions will be used.
    baselines : dict
        Dict mapping questions to baseline scores. If None, defaults will be used.
    treatment_effects : dict
        Dict mapping questions to treatment effects. If None, defaults will be used.
    time_trends : dict
        Dict mapping questions to natural time trends (change over time without treatment).
        If None, defaults will be used.
    response_rate_per_period : float
        Probability of a user responding in each time period (0-1)
    noise_scale : float
        Standard deviation of random noise added to scores
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic DiD data
    dict
        Dict with true parameter values used to generate the data
    """
    np.random.seed(seed)
    
    # Default questions if not specified
    if questions is None:
        questions = [
            'app_usability',
            'feature_satisfaction',
            'customer_support',
            'performance_rating',
            'overall_satisfaction'
        ]
    
    # Default baselines if not specified
    if baselines is None:
        baselines = {
            'app_usability': 70,
            'feature_satisfaction': 65,
            'customer_support': 62,
            'performance_rating': 68,
            'overall_satisfaction': 72
        }
        # Fill in any missing questions with default value of 70
        for q in questions:
            if q not in baselines:
                baselines[q] = 70
    
    # Default treatment effects if not specified
    if treatment_effects is None:
        treatment_effects = {
            'app_usability': 5.2,
            'feature_satisfaction': 4.8,
            'customer_support': 5.5,
            'performance_rating': 6.2,
            'overall_satisfaction': 7.0
        }
        # Fill in any missing questions with default effect of 5.0
        for q in questions:
            if q not in treatment_effects:
                treatment_effects[q] = 5.0
    
    # Default time trends if not specified
    if time_trends is None:
        time_trends = {
            'app_usability': 1.0,  # Natural 1-point increase over time
            'feature_satisfaction': 0.5,
            'customer_support': -0.5,  # Natural decline over time
            'performance_rating': 0.8,
            'overall_satisfaction': 0.3
        }
        # Fill in any missing questions with default trend of 0.0 (no natural trend)
        for q in questions:
            if q not in time_trends:
                time_trends[q] = 0.0
    
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
    
    # Activity levels affect response frequency
    response_probabilities = {
        'Low': response_rate_per_period * 0.7,    # Lower response rate for low activity users
        'Medium': response_rate_per_period,       # Base response rate
        'High': min(1.0, response_rate_per_period * 1.3)  # Higher response rate for high activity users
    }
    
    # Define time periods
    # Time is defined as 0 (pre) or 1 (post) for the actual DiD model
    pre_indicator = 0
    post_indicator = 1
    
    # For more realistic data, generate dates
    start_date = datetime(2023, 1, 1)
    pre_dates = [(start_date + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(pre_periods)]
    post_dates = [(start_date + timedelta(days=d+pre_periods)).strftime('%Y-%m-%d') for d in range(post_periods)]
    all_dates = pre_dates + post_dates
    
    # Total users
    n_users = n_treatment + n_control
    
    # Generate user data
    users = []
    for i in range(1, n_users + 1):
        # Assign treatment (first users are treated)
        treatment = 1 if i <= n_treatment else 0
        
        # Generate demographics
        age = np.random.choice(list(age_groups.keys()), p=list(age_groups.values()))
        gender = np.random.choice(list(gender_groups.keys()), p=list(gender_groups.values()))
        activity = np.random.choice(list(activity_levels.keys()), p=list(activity_levels.values()))
        
        # Add user-specific baseline variation
        user_baseline_effect = np.random.normal(0, 8)
        
        users.append({
            'userid': i,
            'treatment': treatment,
            'age': age,
            'gender': gender,
            'activity': activity,
            'baseline_effect': user_baseline_effect
        })
    
    # Generate survey responses
    responses = []
    
    for user in users:
        # Determine probability of response based on activity level
        response_prob = response_probabilities[user['activity']]
        
        # For each date, determine if user responds
        for date in all_dates:
            # Determine if pre or post period
            is_post = date in post_dates
            time_indicator = post_indicator if is_post else pre_indicator
            
            # Decide if user responds on this date
            if np.random.random() < response_prob:
                # Add a random day effect
                day_effect = np.random.normal(0, 2)
                
                for question in questions:
                    baseline = baselines[question]
                    
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
    
    # Return data and true parameters
    true_params = {
        'baselines': baselines,
        'treatment_effects': treatment_effects,
        'time_trends': time_trends,
        'demographics': {
            'age_effects': age_effects,
            'gender_effects': gender_effects
        }
    }
    
    return did_data, true_params


def generate_synthetic_control_data(
    n_treatment_units=10,
    n_control_units=40,
    n_users_per_unit=1000,
    pre_periods=14,
    post_periods=14,
    questions=None,
    baselines=None,
    treatment_effects=None,
    unit_trends=None,
    noise_scale=3.0,
    daily_response_probability=0.3,
    seed=42
):
    """
    Generate synthetic data for Synthetic Control analysis with known effects.
    
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
    questions : list
        List of question names. If None, default questions will be used.
    baselines : dict
        Dict mapping questions to baseline scores. If None, defaults will be used.
    treatment_effects : dict
        Dict mapping questions to treatment effects. If None, defaults will be used.
    unit_trends : dict or None
        If provided, a dict mapping unit IDs to trend coefficients.
        If None, random trends will be generated.
    noise_scale : float
        Standard deviation of random noise added to scores
    daily_response_probability : float
        Probability of a user responding on any given day (0-1)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic data for Synthetic Control
    dict
        Dict with true parameter values used to generate the data
    """
    np.random.seed(seed)
    
    # Default questions if not specified
    if questions is None:
        questions = [
            'overall_satisfaction',
            'ease_of_use',
            'customer_support',
            'value_for_money',
            'likelihood_to_recommend'
        ]
    
    # Default baselines if not specified
    if baselines is None:
        baselines = {
            'overall_satisfaction': 75,
            'ease_of_use': 70,
            'customer_support': 65,
            'value_for_money': 60,
            'likelihood_to_recommend': 72
        }
        # Fill in any missing questions with default value of 70
        for q in questions:
            if q not in baselines:
                baselines[q] = 70
    
    # Default treatment effects if not specified
    if treatment_effects is None:
        treatment_effects = {
            'overall_satisfaction': 5.2,
            'ease_of_use': 4.8,
            'customer_support': 5.5,
            'value_for_money': 4.6,
            'likelihood_to_recommend': 6.0
        }
        # Fill in any missing questions with default effect of 5.0
        for q in questions:
            if q not in treatment_effects:
                treatment_effects[q] = 5.0
    
    # Total units and users
    n_units = n_treatment_units + n_control_units
    
    # Generate unit IDs
    unit_ids = list(range(1, n_units + 1))
    
    # Assign units to treatment or control
    treatments = [1] * n_treatment_units + [0] * n_control_units
    
    # Generate unit-specific trends if not provided
    if unit_trends is None:
        unit_trends = {}
        for unit_id in unit_ids:
            # Control units have varied trends
            if unit_id > n_treatment_units:
                unit_trends[unit_id] = np.random.uniform(0.05, 0.15)
            # Treatment units have similar trends to some control units
            else:
                unit_trends[unit_id] = np.random.uniform(0.08, 0.12)
    
    # Generate unit-specific baselines
    unit_baselines = {}
    for unit_id in unit_ids:
        # Add some variation to baseline scores for each unit
        unit_baselines[unit_id] = {}
        for q in questions:
            unit_baselines[unit_id][q] = baselines[q] + np.random.normal(0, 5)
    
    # Define demographics for users
    age_groups = {
        '18-24': 0.15,
        '25-34': 0.25,
        '35-44': 0.20,
        '45-54': 0.18,
        '55-64': 0.12,
        '65+': 0.10
    }
    
    gender_groups = {
        'M': 0.48,
        'F': 0.51,
        'Other': 0.01
    }
    
    education_groups = {
        'High school': 0.25,
        'Some college': 0.20,
        'Bachelor': 0.35,
        'Graduate': 0.20
    }
    
    # Demographic trends (how different demographics respond over time)
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
    
    # Define time periods
    start_date = datetime(2023, 1, 1)
    pre_dates = [(start_date + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(pre_periods)]
    post_dates = [(start_date + timedelta(days=d+pre_periods)).strftime('%Y-%m-%d') for d in range(post_periods)]
    all_dates = pre_dates + post_dates
    
    # Map dates to time periods for the model
    date_to_period = {date: i for i, date in enumerate(all_dates)}
    time_periods = list(range(len(all_dates)))
    pre_period_indices = list(range(len(pre_dates)))
    post_period_indices = list(range(len(pre_dates), len(all_dates)))
    
    # Day-of-week effects
    day_of_week_effects = {
        0: 2,    # Monday
        1: 1,    # Tuesday
        2: 0,    # Wednesday
        3: -1,   # Thursday
        4: -2,   # Friday
        5: 3,    # Saturday
        6: 2     # Sunday
    }
    
    # Create user data and responses
    rows = []
    user_id_counter = 1
    
    # For each unit
    for i, unit_id in enumerate(unit_ids):
        treatment = treatments[i]
        
        # Generate users for this unit (with some variation in number)
        n_users_this_unit = int(n_users_per_unit * np.random.uniform(0.8, 1.2))
        
        # For each user in this unit
        for _ in range(n_users_this_unit):
            # Generate user demographics
            age = np.random.choice(list(age_groups.keys()), p=list(age_groups.values()))
            gender = np.random.choice(list(gender_groups.keys()), p=list(gender_groups.values()))
            education = np.random.choice(list(education_groups.keys()), p=list(education_groups.values()))
            
            # Add base user variability
            user_effect = np.random.normal(0, 5)
            
            # Only some users respond on each date
            response_dates = [date for date in all_dates if np.random.random() < daily_response_probability]
            
            for date in response_dates:
                # Get time period index and day of week
                t = date_to_period[date]
                day_of_week = datetime.strptime(date, '%Y-%m-%d').weekday()
                
                # For each question
                for q in questions:
                    # Base score with unit-specific baseline
                    baseline = unit_baselines[unit_id][q]
                    
                    # Add user-specific effect
                    score = baseline + user_effect
                    
                    # Add unit trend over time
                    score += unit_trends[unit_id] * t
                    
                    # Add demographic trends
                    score += age_trends[age] * t
                    score += gender_trends[gender] * t
                    score += education_trends[education] * t
                    
                    # Add day-of-week effects
                    score += day_of_week_effects[day_of_week]
                    
                    # Add treatment effect for treated units in post-treatment periods
                    if treatment == 1 and t in post_period_indices:
                        # Gradual treatment effect that increases over time
                        days_since_treatment = t - len(pre_dates)
                        ramp_up_factor = min(1.0, days_since_treatment / 5)  # Full effect after 5 days
                        score += treatment_effects[q] * ramp_up_factor
                    
                    # Add noise
                    score += np.random.normal(0, noise_scale)
                    
                    # Ensure score is between 0 and 100
                    score = max(0, min(100, score))
                    
                    # Round to nearest integer for realistic scores
                    score_rounded = round(score)
                    
                    # Map to Likert scale
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
                    rows.append({
                        'userid': user_id_counter,
                        'unitid': unit_id,
                        'question': q,
                        'date': date,
                        'time': t,
                        'raw_response': likert_response,
                        'likert_score': likert_score,
                        'score': score_rounded,
                        'treatment': treatment,
                        'age': age,
                        'gender': gender,
                        'education': education
                    })
            
            # Increment user ID counter
            user_id_counter += 1
    
    # Create the DataFrame
    synth_data = pd.DataFrame(rows)
    
    # Return data and true parameters
    true_params = {
        'baselines': baselines,
        'unit_baselines': unit_baselines,
        'treatment_effects': treatment_effects,
        'unit_trends': unit_trends,
        'demographics': {
            'age_trends': age_trends,
            'gender_trends': gender_trends,
            'education_trends': education_trends
        },
        'day_of_week_effects': day_of_week_effects,
        'pre_periods': pre_period_indices,
        'post_periods': post_period_indices
    }
    
    return synth_data, true_params


def test_estimator_accuracy(true_effects, estimated_effects):
    """
    Calculate accuracy metrics for estimated effects compared to true effects.
    
    Parameters:
    -----------
    true_effects : dict
        Dictionary mapping questions to true treatment effects
    estimated_effects : dict
        Dictionary mapping questions to estimated treatment effects
    
    Returns:
    --------
    dict
        Dictionary of accuracy metrics (MAE, RMSE, bias)
    """
    errors = []
    abs_errors = []
    sq_errors = []
    
    for question in true_effects:
        if question in estimated_effects:
            true = true_effects[question]
            est = estimated_effects[question]
            error = est - true
            
            errors.append(error)
            abs_errors.append(abs(error))
            sq_errors.append(error**2)
    
    # Calculate metrics
    if len(errors) > 0:
        metrics = {
            'mean_error': np.mean(errors),  # Bias
            'mae': np.mean(abs_errors),     # Mean Absolute Error
            'rmse': np.sqrt(np.mean(sq_errors)),  # Root Mean Squared Error
            'max_abs_error': np.max(abs_errors),  # Maximum Absolute Error
            'questions_evaluated': len(errors)
        }
        return metrics
    else:
        return {'error': 'No matching questions found for evaluation'} 
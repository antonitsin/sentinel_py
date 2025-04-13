import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_survey_analyzer import (
    ResponseValidator,
    DataValidator,
    CausalEstimator, 
    PostStratificationWeightCalculator,
    EffectVisualizer
)

# This is a quickstart script showing how to use the package
if __name__ == "__main__":
    print("Causal Survey Analyzer Quickstart")
    
    # Create example data
    data = pd.DataFrame({
        'userid': [1, 1, 2, 2],
        'question': ['q1', 'q2', 'q1', 'q2'],
        'raw_response': ['Agree', 'Disagree', 'Strongly Agree', 'Agree'],
        'score': [4, 2, 5, 4],
        'treatment': [1, 1, 0, 0],
        'age': [25, 25, 30, 30],
        'gender': ['M', 'M', 'F', 'F']
    })
    
    # Check for data issues
    response_validator = ResponseValidator()
    data_validator = DataValidator()
    
    # Validate encoding
    encoding_map = {'q1': {'Strongly Agree': 5, 'Agree': 4}, 
                    'q2': {'Agree': 4, 'Disagree': 2}}
    
    try:
        mapping = response_validator.validate_encoding(
            data, 'question', 'raw_response', 'score', encoding_map
        )
        print("Response encoding validation successful.")
    except ValueError as e:
        print(f"Encoding validation error: {e}")
    
    # Check for data abnormalities
    issues = data_validator.check_abnormalities(
        data, 'ab', ['question', 'score', 'treatment'], 'userid', 'treatment'
    )
    
    if issues:
        print("\nData issues found:")
        for issue_type, details in issues.items():
            print(f"  - {issue_type}: {details}")
    else:
        print("\nNo data issues found. Data is ready for analysis.")
    
    # Population data for weighting
    pop_data = pd.DataFrame({
        'age': [25, 30],
        'gender': ['M', 'F'],
        'proportion': [0.5, 0.5]
    })
    
    # Apply population weights
    weight_calculator = PostStratificationWeightCalculator()
    data['weights'] = weight_calculator.compute_weights(data, pop_data, ['age', 'gender'])
    
    # Run analysis
    estimator = CausalEstimator()
    results = estimator.run_analysis(
        data, 'ab', 'question', 'score', 'treatment',
        covariates=['age', 'gender'],
        weights_col='weights'
    )
    
    # Print results summary
    for question, result in results.items():
        # Extract treatment effect
        effect = result.params['treatment']
        p_value = result.pvalues['treatment']
        
        print(f"\nQuestion: {question}")
        print(f"Treatment effect: {effect:.2f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Significant: {'Yes' if p_value < 0.1 else 'No'}")
    
    # Visualize results
    visualizer = EffectVisualizer()
    fig = visualizer.visualize_effects(results, 'ab')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig('causal_analysis_results.png')
    print("\nAnalysis complete! Results saved to causal_analysis_results.png") 
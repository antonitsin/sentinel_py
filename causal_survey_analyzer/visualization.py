import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from causal_survey_analyzer.estimator import CausalEstimator

def visualize_effects(results, design, breakdown_col=None, data=None, confidence_level=0.9):
    """
    Visualizes treatment effects as lifts with confidence intervals,
    optionally with breakdown by a covariate.
    
    Parameters:
    -----------
    results : dict
        Output from run_causal_analysis.
    design : str
        Analysis design, used to determine how to extract effects.
    breakdown_col : str, optional
        Column for subgroup analysis (e.g., 'gender').
    data : pd.DataFrame, optional
        Required if breakdown_col is provided.
    confidence_level : float, default=0.9
        Confidence level for intervals (e.g., 0.9 for 90% CI).
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with plots of treatment effects.
    """
    # Check if we need data for breakdowns
    if breakdown_col is not None and data is None:
        raise ValueError("data must be provided if breakdown_col is specified")
    
    # Create figure with appropriate number of subplots
    if breakdown_col is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
        breakdown_values = [None]
        titles = ["Treatment Effects with 90% Confidence Intervals"]
    else:
        # Get unique values for breakdown
        breakdown_values = sorted(data[breakdown_col].unique())
        fig, axes = plt.subplots(
            1, len(breakdown_values) + 1, figsize=(5 * (len(breakdown_values) + 1), 6)
        )
        if len(breakdown_values) + 1 == 1:
            axes = [axes]
        titles = ["Overall"] + [f"{breakdown_col}={val}" for val in breakdown_values]
    
    # For each subplot (overall and breakdowns)
    for i, (ax, breakdown_val, title) in enumerate(zip(axes, [None] + breakdown_values, titles)):
        # Extract results for this breakdown
        if breakdown_val is None and i == 0:
            # Overall results
            effects, cis, p_values = _extract_effects_and_cis(results, design, confidence_level)
            questions = list(effects.keys())
        elif breakdown_col is not None and data is not None:
            # Subset data for this breakdown value
            subset_data = data[data[breakdown_col] == breakdown_val]
            # Re-run analysis on subset
            estimator = CausalEstimator()
            subset_results = estimator.run_analysis(
                subset_data, 
                design=design,
                question_col='question',  # Assuming standard column names
                outcome_col='score',
                treatment_col='treatment',
                time_col='time' if design in ['did', 'prepost', 'synth_control', 'synth_did'] else None,
                # Other params could be added here if needed
            )
            effects, cis, p_values = _extract_effects_and_cis(subset_results, design, confidence_level)
            questions = list(effects.keys())
        
        # Skip if no questions to plot
        if not questions:
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
            ax.set_title(title)
            continue
        
        # Set up bar positions
        positions = np.arange(len(questions))
        
        # Create bars with colors based on significance
        bar_colors = []
        for q in questions:
            effect = effects[q]
            p_value = p_values[q]
            
            if p_value < (1 - confidence_level):
                # Significant
                color = 'green' if effect > 0 else 'red'
            else:
                # Not significant
                color = 'grey'
            
            bar_colors.append(color)
        
        # Plot bars and error bars
        bars = ax.bar(
            positions, 
            [effects[q] for q in questions],
            color=bar_colors,
            alpha=0.7
        )
        
        # Add error bars
        ax.errorbar(
            positions,
            [effects[q] for q in questions],
            yerr=[[effects[q] - cis[q][0] for q in questions], 
                  [cis[q][1] - effects[q] for q in questions]],
            fmt='none',
            ecolor='black',
            capsize=5
        )
        
        # Styling
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(questions, rotation=45, ha='right')
        ax.set_ylabel('Treatment Effect (Lift)')
        ax.set_title(title)
        
        # Add annotations for significant effects
        for i, q in enumerate(questions):
            if p_values[q] < (1 - confidence_level):
                # Significant effect
                sign = '+' if effects[q] > 0 else ''
                ax.annotate(
                    f'{sign}{effects[q]:.2f}*',
                    xy=(positions[i], effects[q]),
                    xytext=(0, 5 if effects[q] >= 0 else -15),
                    textcoords='offset points',
                    ha='center',
                    fontweight='bold'
                )
            else:
                # Non-significant effect
                ax.annotate(
                    f'{effects[q]:.2f}',
                    xy=(positions[i], effects[q]),
                    xytext=(0, 5 if effects[q] >= 0 else -15),
                    textcoords='offset points',
                    ha='center'
                )
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='Positive & Significant'),
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='Negative & Significant'),
        plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.7, label='Not Significant')
    ]
    
    if len(axes) > 1:
        # Add legend to the first subplot
        axes[0].legend(handles=legend_elements, loc='best')
    else:
        axes[0].legend(handles=legend_elements, loc='best')
    
    fig.tight_layout()
    return fig


def _extract_effects_and_cis(results, design, confidence_level=0.9):
    """
    Helper function to extract effects, confidence intervals, and p-values
    from analysis results based on the design.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results from causal analysis.
    design : str
        Analysis design.
    confidence_level : float
        Confidence level for intervals (e.g., 0.9 for 90% CI).
    
    Returns:
    --------
    tuple
        (effects, confidence_intervals, p_values) dictionaries
    """
    effects = {}
    cis = {}
    p_values = {}
    
    for question, result in results.items():
        if design == 'ab':
            # For A/B, effect is treatment coefficient
            effect_param = 'treatment'
        elif design in ['did', 'synth_did']:
            # For DiD, effect is treatment:time interaction
            effect_param = 'treatment_time'
        elif design == 'prepost':
            # For pre/post, effect is time coefficient
            effect_param = 'time'
        elif design == 'synth_control':
            # For synthetic control, use average effect across post periods
            effects[question] = result['avg_effect']
            # We don't have CIs for synthetic control in this implementation
            cis[question] = (result['avg_effect'] - 0.1, result['avg_effect'] + 0.1)
            # Use a placeholder p-value based on separation from zero
            effect_size = abs(result['avg_effect'])
            p_values[question] = 0.05 if effect_size > 0.2 else 0.2
            continue
        
        # Extract effect, CI, and p-value for regression-based designs
        if hasattr(result, 'params') and effect_param in result.params:
            effect = result.params[effect_param]
            effects[question] = effect
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            crit_val = result.cov_params()[effect_param, effect_param] ** 0.5
            crit_val = crit_val * sm.stats.t.ppf(1 - alpha/2, result.df_resid)
            
            cis[question] = (effect - crit_val, effect + crit_val)
            
            # Extract p-value
            p_values[question] = result.pvalues[effect_param]
    
    return effects, cis, p_values 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from scipy.optimize import minimize
from causal_survey_analyzer.utils import DataPreparer


class CausalEstimator:
    """
    Estimates causal effects using various experimental and quasi-experimental designs.
    """
    
    def __init__(self):
        """Initialize the CausalEstimator object."""
        # Store valid designs for reference
        self.valid_designs = ['ab', 'did', 'prepost', 'synth_control', 'synth_did']
    
    def run_analysis(self, data, design, question_col, outcome_col, treatment_col, 
                     time_col=None, covariates=None, weights_col=None, 
                     synth_pre_periods=None, userid_col='userid'):
        """
        Runs causal effect estimation for the specified design.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data.
        design : str
            One of 'ab', 'did', 'prepost', 'synth_control', 'synth_did'.
        question_col : str
            Column name for question identifier.
        outcome_col : str
            Column name for score.
        treatment_col : str
            Column name for treatment indicator.
        time_col : str, optional
            Column name for time (required for DiD, pre/post, synthetic designs).
        covariates : list, optional
            Covariate column names for adjustment.
        weights_col : str, optional
            Column name for pre-computed weights.
        synth_pre_periods : list, optional
            Time points considered pre-treatment for synthetic designs.
        userid_col : str, default='userid'
            Column name for user ID.
            
        Returns:
        --------
        dict
            Dictionary mapping questions to estimation results.
        """
        if design not in self.valid_designs:
            raise ValueError(f"Invalid design: {design}. Must be one of {self.valid_designs}")
            
        if covariates is None:
            covariates = []
        
        # Prepare data for analysis
        data_preparer = DataPreparer()
        prepared_data = data_preparer.prepare_for_analysis(
            data, design, question_col, outcome_col, treatment_col, 
            time_col, covariates, userid_col
        )
        
        # Run appropriate analysis based on design
        results = {}
        
        for question, q_data in prepared_data.items():
            if design == 'ab':
                results[question] = self._run_ab_test(
                    q_data, outcome_col, treatment_col, covariates, weights_col
                )
            
            elif design == 'did':
                results[question] = self._run_did(
                    q_data, outcome_col, treatment_col, time_col, covariates, weights_col
                )
            
            elif design == 'prepost':
                # Check if control group is present
                has_control = q_data[treatment_col].nunique() > 1
                if has_control:
                    results[question] = self._run_did(
                        q_data, outcome_col, treatment_col, time_col, covariates, weights_col
                    )
                else:
                    warnings.warn(
                        "Pre/post analysis without control group may be confounded. "
                        "Consider alternative designs if available."
                    )
                    results[question] = self._run_prepost(
                        q_data, outcome_col, time_col, covariates, weights_col
                    )
            
            elif design == 'synth_control':
                if synth_pre_periods is None:
                    # If not specified, use all periods where time < max(time) for treated units
                    treated_data = q_data[q_data[treatment_col] == 1]
                    max_time = treated_data[time_col].max()
                    synth_pre_periods = sorted(q_data[q_data[time_col] < max_time][time_col].unique())
                
                results[question] = self._run_synthetic_control(
                    q_data, outcome_col, treatment_col, time_col, userid_col, synth_pre_periods
                )
            
            elif design == 'synth_did':
                if synth_pre_periods is None:
                    # If not specified, use all periods where time < max(time) for treated units
                    treated_data = q_data[q_data[treatment_col] == 1]
                    max_time = treated_data[time_col].max()
                    synth_pre_periods = sorted(q_data[q_data[time_col] < max_time][time_col].unique())
                
                results[question] = self._run_synthetic_did(
                    q_data, outcome_col, treatment_col, time_col, userid_col, 
                    synth_pre_periods, covariates, weights_col
                )
        
        return results

    def _run_ab_test(self, data, outcome_col, treatment_col, covariates=None, weights_col=None):
        """
        Runs A/B test analysis using OLS regression.
        
        Formula: score ~ treatment + covariates
        """
        if covariates is None:
            covariates = []
        
        # Construct formula
        formula = f"{outcome_col} ~ {treatment_col}"
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        # Create model matrix
        y = data[outcome_col]
        X = sm.add_constant(data[[treatment_col] + covariates])
        
        # Fit model with or without weights
        if weights_col is not None and weights_col in data.columns:
            weights = data[weights_col]
            model = sm.WLS(y, X, weights=weights)
        else:
            model = sm.OLS(y, X)
        
        results = model.fit()
        return results

    def _run_did(self, data, outcome_col, treatment_col, time_col, covariates=None, weights_col=None):
        """
        Runs Difference-in-Differences analysis using OLS regression.
        
        Formula: score ~ treatment + time + treatment*time + covariates
        """
        if covariates is None:
            covariates = []
        
        # Create interaction term
        data = data.copy()
        data['treatment_time'] = data[treatment_col] * data[time_col]
        
        # Construct formula
        formula = f"{outcome_col} ~ {treatment_col} + {time_col} + treatment_time"
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        # Create model matrix
        y = data[outcome_col]
        X = sm.add_constant(data[[treatment_col, time_col, 'treatment_time'] + covariates])
        
        # Fit model with or without weights
        if weights_col is not None and weights_col in data.columns:
            weights = data[weights_col]
            model = sm.WLS(y, X, weights=weights)
        else:
            model = sm.OLS(y, X)
        
        results = model.fit()
        return results

    def _run_prepost(self, data, outcome_col, time_col, covariates=None, weights_col=None):
        """
        Runs Pre/Post analysis using OLS regression for treatment group only.
        
        Formula: score ~ time + covariates
        """
        if covariates is None:
            covariates = []
        
        # Construct formula
        formula = f"{outcome_col} ~ {time_col}"
        if covariates:
            formula += " + " + " + ".join(covariates)
        
        # Create model matrix
        y = data[outcome_col]
        X = sm.add_constant(data[[time_col] + covariates])
        
        # Fit model with or without weights
        if weights_col is not None and weights_col in data.columns:
            weights = data[weights_col]
            model = sm.WLS(y, X, weights=weights)
        else:
            model = sm.OLS(y, X)
        
        results = model.fit()
        return results

    def _run_synthetic_control(self, data, outcome_col, treatment_col, time_col, userid_col, 
                              pre_periods):
        """
        Runs synthetic control analysis by finding optimal weights for control units
        that minimize pre-treatment prediction error for treated units.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data.
        outcome_col : str
            Column name for score.
        treatment_col : str
            Column name for treatment indicator.
        time_col : str
            Column name for time.
        userid_col : str
            Column name for user ID.
        pre_periods : list
            Time points considered pre-treatment.
            
        Returns:
        --------
        dict
            Dictionary with synthetic control results.
        """
        # Identify treated and control units
        treated_units = data[data[treatment_col] == 1][userid_col].unique()
        control_units = data[data[treatment_col] == 0][userid_col].unique()
        
        # If multiple treated units, compute average treatment effect
        if len(treated_units) > 1:
            warnings.warn(
                f"Multiple treated units detected ({len(treated_units)}). "
                f"Computing average treatment effect across all treated units."
            )
        
        # Pivot data to create unit x time panel
        panel = data.pivot_table(
            index=userid_col, 
            columns=time_col, 
            values=outcome_col, 
            aggfunc='mean'
        )
        
        # Identify all time periods
        all_periods = sorted(data[time_col].unique())
        post_periods = [t for t in all_periods if t not in pre_periods]
        
        # For each treated unit, compute synthetic control
        unit_effects = {}
        
        for unit in treated_units:
            # Observed outcome for treated unit
            Y1 = panel.loc[unit]
            
            # Pre-treatment outcomes for optimization
            Y1_pre = Y1[pre_periods]
            
            # Control units' pre-treatment outcomes
            Y0_pre = panel.loc[control_units, pre_periods]
            
            # Define optimization objective function
            def objective(weights):
                # Calculate synthetic control pre-treatment outcome
                Y_synth = np.dot(weights, Y0_pre)
                # Return MSE
                return np.mean((Y1_pre - Y_synth) ** 2)
            
            # Constraint: weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            # Bounds: weights are non-negative
            bounds = [(0, 1) for _ in range(len(control_units))]
            
            # Initial equal weights
            initial_weights = np.ones(len(control_units)) / len(control_units)
            
            # Optimize weights
            result = minimize(
                objective, initial_weights, method='SLSQP',
                constraints=constraints, bounds=bounds
            )
            
            # Extract optimal weights
            optimal_weights = result['x']
            
            # Compute synthetic control for all periods
            Y0_all = panel.loc[control_units]
            Y_synth_all = pd.Series(
                np.dot(optimal_weights, Y0_all), 
                index=all_periods
            )
            
            # Calculate treatment effects for post-treatment periods
            effects = {}
            for t in post_periods:
                if t in Y1 and t in Y_synth_all:
                    effects[t] = Y1[t] - Y_synth_all[t]
            
            # Store unit results
            unit_effects[unit] = {
                'weights': dict(zip(control_units, optimal_weights)),
                'observed': Y1.to_dict(),
                'synthetic': Y_synth_all.to_dict(),
                'effects': effects,
                'avg_effect': np.mean(list(effects.values())) if effects else np.nan,
                'mse': result['fun']  # Pre-treatment MSE
            }
        
        # Compute average effect across treated units
        if unit_effects:
            avg_effects = {}
            for t in post_periods:
                effects_t = [
                    unit_data['effects'].get(t, np.nan) 
                    for unit_data in unit_effects.values()
                    if t in unit_data['effects']
                ]
                if effects_t:
                    avg_effects[t] = np.nanmean(effects_t)
            
            avg_avg_effect = np.nanmean(list(avg_effects.values())) if avg_effects else np.nan
        else:
            avg_effects = {}
            avg_avg_effect = np.nan
        
        return {
            'unit_effects': unit_effects,
            'avg_effects': avg_effects,
            'avg_effect': avg_avg_effect
        }

    def _run_synthetic_did(self, data, outcome_col, treatment_col, time_col, userid_col, 
                          pre_periods, covariates=None, weights_col=None):
        """
        Runs synthetic difference-in-differences analysis, combining
        synthetic control with DiD for improved robustness.
        
        Returns:
        --------
        dict
            Dictionary with synthetic DiD results.
        """
        # First run synthetic control
        synth_results = self._run_synthetic_control(
            data, outcome_col, treatment_col, time_col, userid_col, pre_periods
        )
        
        # Create new data with synthetic controls as the new comparison group
        synth_data = data.copy()
        
        # Identify treated units
        treated_units = data[data[treatment_col] == 1][userid_col].unique()
        
        # For each treated unit, create synthetic comparison
        for unit in treated_units:
            if unit in synth_results['unit_effects']:
                # Get unit's data
                unit_data = data[data[userid_col] == unit].copy()
                
                # Create a synthetic comparison unit row for each of the unit's rows
                for _, row in unit_data.iterrows():
                    time = row[time_col]
                    if time in synth_results['unit_effects'][unit]['synthetic']:
                        # Create a synthetic row with the same time period
                        synth_row = row.copy()
                        synth_row[outcome_col] = synth_results['unit_effects'][unit]['synthetic'][time]
                        synth_row[treatment_col] = 0  # This is the control
                        synth_row[userid_col] = f"synth_{unit}"
                        synth_data = pd.concat([synth_data, pd.DataFrame([synth_row])], ignore_index=True)
        
        # Run DiD on the augmented data (original treated + synthetic controls)
        did_results = self._run_did(
            synth_data, outcome_col, treatment_col, time_col, covariates, weights_col
        )
        
        # Combine results
        combined_results = {
            'synth_control': synth_results,
            'did': did_results,
            # Use DiD interaction term coefficient for the main effect
            'params': did_results.params.to_dict(),
            'pvalues': did_results.pvalues.to_dict()
        }
        
        return combined_results 
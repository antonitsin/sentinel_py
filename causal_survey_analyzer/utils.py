import pandas as pd
import numpy as np
import warnings


class PostStratificationWeightCalculator:
    """
    Calculates post-stratification weights to correct for sample representation.
    """
    
    @staticmethod
    def compute_weights(sample_data, population_data, variables):
        """
        Computes post-stratification weights based on population data.
        
        Parameters:
        -----------
        sample_data : pd.DataFrame
            Survey data.
        population_data : pd.DataFrame
            Population data with proportions for stratification variables.
            Must contain a 'proportion' column.
        variables : list
            Stratification variables (e.g., ['age', 'gender']).
            
        Returns:
        --------
        pd.Series
            Weights for each row in sample_data.
        
        Notes:
        ------
        Weights = (population proportion) / (sample proportion) per stratum.
        """
        # Validate inputs
        for var in variables:
            if var not in sample_data.columns:
                raise ValueError(f"Variable '{var}' not found in sample_data")
            if var not in population_data.columns:
                raise ValueError(f"Variable '{var}' not found in population_data")
        
        if 'proportion' not in population_data.columns:
            raise ValueError("Population data must contain a 'proportion' column")
        
        # Check if proportions sum to 1
        pop_sum = population_data['proportion'].sum()
        if not np.isclose(pop_sum, 1.0, atol=1e-10):
            warnings.warn(f"Population proportions sum to {pop_sum}, not 1.0. Normalizing.")
            population_data = population_data.copy()
            population_data['proportion'] = population_data['proportion'] / pop_sum
        
        # Get unique user IDs to count each user only once
        if 'userid' in sample_data.columns:
            user_data = sample_data.drop_duplicates('userid')
        else:
            # If no userid, treat each row as a separate observation
            user_data = sample_data
        
        # Compute sample proportions
        sample_counts = user_data.groupby(variables).size().reset_index(name='count')
        total_sample = sample_counts['count'].sum()
        sample_counts['sample_proportion'] = sample_counts['count'] / total_sample
        
        # Merge with population proportions
        merged = pd.merge(sample_counts, population_data, on=variables, how='left')
        
        # Handle missing strata in population data
        if merged['proportion'].isna().any():
            missing_strata = merged[merged['proportion'].isna()]
            warnings.warn(
                f"Some strata in the sample are not in the population data:\n{missing_strata[variables]}"
                f"\nAssigning zero weight to these strata."
            )
            merged['proportion'] = merged['proportion'].fillna(0)
        
        # Compute weights
        merged['weight'] = merged['proportion'] / merged['sample_proportion']
        
        # Replace infinite weights (from zero sample proportion) with a large value
        if np.isinf(merged['weight']).any():
            max_weight = merged[~np.isinf(merged['weight'])]['weight'].max() * 10
            merged.loc[np.isinf(merged['weight']), 'weight'] = max_weight
            warnings.warn(
                f"Some strata have zero sample proportion but non-zero population proportion. "
                f"Setting weight to {max_weight}."
            )
        
        # Create a mapping from strata to weights
        weight_map = pd.Series(
            merged['weight'].values,
            index=pd.MultiIndex.from_frame(merged[variables])
        )
        
        # Apply weights to each row in the original data
        weights = sample_data.set_index(variables).index.map(lambda idx: weight_map.get(idx, 1.0))
        
        return weights


class DataPreparer:
    """
    Prepares data for causal analysis by restructuring it based on the design.
    """
    
    @staticmethod
    def prepare_for_analysis(data, design, question_col, outcome_col, treatment_col, 
                            time_col=None, covariates=None, userid_col='userid'):
        """
        Prepares data for causal analysis by restructuring it based on the design.
        
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
            List of covariate column names.
        userid_col : str, default='userid'
            Column name for user ID.
            
        Returns:
        --------
        dict
            Dictionary with prepared data for each question.
        """
        if covariates is None:
            covariates = []
        
        # Validate design
        valid_designs = ['ab', 'did', 'prepost', 'synth_control', 'synth_did']
        if design not in valid_designs:
            raise ValueError(f"Invalid design: {design}. Must be one of {valid_designs}")
        
        # Check if design requires time
        design_requires_time = design in ['did', 'prepost', 'synth_control', 'synth_did']
        if design_requires_time and (time_col is None or time_col not in data.columns):
            raise ValueError(f"Design '{design}' requires a time column")
        
        # Prepare data question by question
        questions = data[question_col].unique()
        prepared_data = {}
        
        for q in questions:
            q_data = data[data[question_col] == q].copy()
            
            # List of columns to keep
            cols_to_keep = [userid_col, outcome_col, treatment_col] + covariates
            if design_requires_time:
                cols_to_keep.append(time_col)
            
            # Keep only necessary columns
            q_data = q_data[cols_to_keep]
            
            # For AB design, ensure we only have post-treatment data
            if design == 'ab' and time_col in q_data.columns:
                q_data = q_data[q_data[time_col] == 1]
                q_data = q_data.drop(columns=[time_col])
            
            prepared_data[q] = q_data
        
        return prepared_data 
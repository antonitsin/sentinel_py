import pandas as pd
import numpy as np
import warnings


class ResponseValidator:
    """
    Validates the encoding of raw survey responses to scores.
    """
    
    @staticmethod
    def validate_encoding(data, question_col, raw_response_col, score_col, encoding_map=None):
        """
        Verifies that raw responses are correctly mapped to scores for each question.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing raw responses and scores.
        question_col : str
            Column name for question identifier.
        raw_response_col : str
            Column name for raw responses.
        score_col : str
            Column name for scores.
        encoding_map : dict, optional
            Dictionary per question mapping raw responses to scores 
            (e.g., {'q1': {'Strongly Agree': 5, 'Agree': 4, ...}}).
            If None, infers mapping from data and returns it.
        
        Returns:
        --------
        dict
            Validated mappings per question.
        
        Raises:
        -------
        ValueError
            If inconsistencies are found in the mapping.
        """
        # Check required columns exist
        for col in [question_col, raw_response_col, score_col]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Get all unique questions
        questions = data[question_col].unique()
        
        # Initialize an empty mapping if none provided
        if encoding_map is None:
            encoding_map = {}
            for q in questions:
                encoding_map[q] = {}
        
        # Validate or infer mapping for each question
        observed_mappings = {}
        
        for q in questions:
            q_data = data[data[question_col] == q]
            observed_map = {}
            
            # Extract observed mappings
            for _, row in q_data.iterrows():
                raw_resp = row[raw_response_col]
                score = row[score_col]
                
                if raw_resp in observed_map and observed_map[raw_resp] != score:
                    raise ValueError(
                        f"Inconsistent mapping for question {q}: "
                        f"'{raw_resp}' maps to both {observed_map[raw_resp]} and {score}"
                    )
                
                observed_map[raw_resp] = score
            
            observed_mappings[q] = observed_map
            
            # If encoding_map has entry for this question, validate against it
            if q in encoding_map:
                for raw_resp, expected_score in encoding_map[q].items():
                    if raw_resp in observed_map and observed_map[raw_resp] != expected_score:
                        raise ValueError(
                            f"Mapping error for question {q}: "
                            f"'{raw_resp}' maps to {observed_map[raw_resp]} in data "
                            f"but {expected_score} in provided mapping"
                        )
        
        # If no encoding_map was provided, return the observed mappings
        if encoding_map == {}:
            return observed_mappings
        
        # Update encoding_map with any missing observed mappings
        for q, mappings in observed_mappings.items():
            if q not in encoding_map:
                encoding_map[q] = {}
            for raw_resp, score in mappings.items():
                if raw_resp not in encoding_map[q]:
                    encoding_map[q][raw_resp] = score
        
        return encoding_map


class DataValidator:
    """
    Validates survey data for causal analysis, detecting potential issues.
    """
    
    @staticmethod
    def check_abnormalities(data, design, required_cols, userid_col, treatment_col, time_col=None):
        """
        Detects potential issues in the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data.
        design : str
            One of 'ab', 'did', 'prepost', 'synth_control', 'synth_did'.
        required_cols : list
            Columns expected in data (e.g., ['question', 'score', 'treatment']).
        userid_col : str
            Column name for user ID.
        treatment_col : str
            Column name for treatment indicator.
        time_col : str, optional
            Column name for time (required for DiD, pre/post, synthetic designs).
        
        Returns:
        --------
        dict
            Dictionary summarizing issues found.
        """
        issues = {}
        
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in data.columns]
        if userid_col not in data.columns:
            missing_cols.append(userid_col)
        if treatment_col not in data.columns:
            missing_cols.append(treatment_col)
        
        design_requires_time = design in ['did', 'prepost', 'synth_control', 'synth_did']
        if design_requires_time and (time_col is None or time_col not in data.columns):
            missing_cols.append(time_col if time_col else 'time_col')
        
        if missing_cols:
            issues['missing_columns'] = missing_cols
            return issues  # Return early as other checks need these columns
        
        # Check for missing values
        missing_values = {}
        for col in required_cols + [userid_col, treatment_col]:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = missing_count
        
        if design_requires_time and time_col:
            missing_time = data[time_col].isna().sum()
            if missing_time > 0:
                missing_values[time_col] = missing_time
        
        if missing_values:
            issues['missing_values'] = missing_values
        
        # Check for duplicates
        if design_requires_time and time_col:
            duplicates = data.duplicated(subset=[userid_col, 'question', time_col], keep='first').sum()
        else:
            duplicates = data.duplicated(subset=[userid_col, 'question'], keep='first').sum()
        
        if duplicates > 0:
            issues['duplicates'] = duplicates
        
        # Check if treatment is binary
        unique_treatments = data[treatment_col].unique()
        if not all(treatment in [0, 1] for treatment in unique_treatments):
            issues['non_binary_treatment'] = unique_treatments.tolist()
        
        # Check if time is binary (for did/prepost)
        if design in ['did', 'prepost'] and time_col:
            unique_times = data[time_col].unique()
            if not all(time in [0, 1] for time in unique_times):
                issues['non_binary_time'] = unique_times.tolist()
        
        # Check consistent treatment within user across time
        if design_requires_time and time_col:
            treatment_consistency = data.groupby(userid_col)[treatment_col].nunique() > 1
            inconsistent_users = treatment_consistency[treatment_consistency].index.tolist()
            if inconsistent_users:
                issues['inconsistent_treatment'] = inconsistent_users
        
        # Specific checks for synthetic designs
        if design in ['synth_control', 'synth_did']:
            # Need multiple control units
            control_units = data[data[treatment_col] == 0][userid_col].nunique()
            if control_units < 2:
                issues['insufficient_control_units'] = control_units
            
            # Need pre-treatment periods
            if time_col:
                treatment_units = data[data[treatment_col] == 1][userid_col].unique()
                for unit in treatment_units:
                    unit_data = data[data[userid_col] == unit]
                    if unit_data[time_col].min() != 0:
                        if 'missing_pre_treatment' not in issues:
                            issues['missing_pre_treatment'] = []
                        issues['missing_pre_treatment'].append(unit)
        
        return issues 
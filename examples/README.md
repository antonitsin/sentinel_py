# Causal Survey Analyzer Examples

This directory contains example scripts and tutorials demonstrating how to use the `causal_survey_analyzer` package.

## Installation

Before running the examples, make sure the package is installed:

1. Install the package:
   ```bash
   # From the project root directory
   pip install -e .
   
   # Or if you're using it from PyPI
   pip install causal_survey_analyzer
   ```

2. Install dependencies for the examples:
   ```bash
   pip install pandas numpy matplotlib jupyter nbconvert
   ```

## Contents

- `data_generators.py`: Functions to generate synthetic data for different experimental designs
- `test_estimators.py`: Script to test different estimator types with synthetic data
- `quickstart_tutorial.py`: Comprehensive tutorial demonstrating all major features

## Using the Quickstart Tutorial

The quickstart tutorial demonstrates how to use the main features of the causal_survey_analyzer package:

1. A/B Test analysis
2. Difference-in-Differences (DiD) analysis
3. Synthetic Control analysis

### Running the Tutorial

To run the tutorial, simply execute the script:

```bash
python quickstart_tutorial.py
```

If you encounter any errors related to missing modules, make sure the package is installed correctly (see Installation section above).

### Running Without Plots (Headless Mode)

If you're running the script in an environment without a display, you can modify it to save plots to files instead:

```python
# At the top of the script, after imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Then for each plot, add:
plt.savefig('plot_name.png')
# Instead of plt.show()
```

### Converting to Jupyter Notebook (Optional)

If you prefer to work with Jupyter notebooks, you can convert the Python script to a notebook:

1. Install the necessary tools:
   ```bash
   pip install jupyter nbconvert
   ```

2. Convert the script to a notebook:
   ```bash
   jupyter-nbconvert --to notebook --execute quickstart_tutorial.py
   ```

This will create a file named `quickstart_tutorial.ipynb` with all the code cells executed.

## Data Generators

The `data_generators.py` module provides functions to generate synthetic data for:

- A/B tests (`generate_ab_test_data`)
- Difference-in-Differences analysis (`generate_did_data`)
- Synthetic control analysis (`generate_synthetic_control_data`)

These generators are useful for testing and educational purposes.

## Test Estimators

The `test_estimators.py` script evaluates the performance of different causal estimators on synthetic data. It generates data with known treatment effects and compares the estimated effects to the true effects.

## Troubleshooting

If you encounter issues running the examples:

1. Make sure all dependencies are installed
2. Check that you're running the scripts from the correct directory
3. Verify that the package is installed correctly using `pip list | grep causal_survey_analyzer`
4. If you get matplotlib errors, try running in headless mode as described above 
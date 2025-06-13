import unittest
import os
from auto_causal.components.code_generator import generate_code

# Helper to create a dummy dataset file for tests
def create_dummy_csv(path='dummy_test_data.csv'):
    import pandas as pd
    df = pd.DataFrame({
        'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'outcome': [10, 12, 11, 13, 9, 14, 10, 15, 11, 16],
        'covariate1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'covariate2': [5.5, 6.5, 5.8, 6.2, 5.1, 6.8, 5.3, 6.1, 5.9, 6.3]
    })
    df.to_csv(path, index=False)
    return path

class TestCodeGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create dummy data file before running tests
        cls.dummy_data_path = create_dummy_csv()

    @classmethod
    def tearDownClass(cls):
        # Clean up dummy data file after tests
        if os.path.exists(cls.dummy_data_path):
            os.remove(cls.dummy_data_path)

    def test_generate_code_structure(self):
        '''Test if generate_code returns the expected dictionary structure.'''
        method = "regression_adjustment"
        dataset_path = self.dummy_data_path
        variables = {
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "covariates": ["covariate1", "covariate2"]
        }
        method_info = {"method": method} # Basic info needed
        validation_result = {"valid": True} # Assume valid

        result = generate_code(method, dataset_path, variables)

        self.assertIsInstance(result, dict)
        self.assertIn("code", result)
        self.assertIsInstance(result["code"], str)
        self.assertIn("imports", result)
        self.assertIsInstance(result["imports"], list)
        self.assertIn("packages", result)
        self.assertIsInstance(result["packages"], list)
        self.assertIn("effect_estimate", result) # Check default value exists
        
        # Check basic content for regression adjustment
        self.assertIn("smf.ols", result["code"])
        self.assertIn(f"`{variables['outcome_variable']}` ~ `{variables['treatment_variable']}`", result['code'])
        self.assertIn("covariate1", result["code"])
        self.assertIn("covariate2", result["code"])
        self.assertIn(dataset_path, result['code'])

    def test_generate_code_imports_packages(self):
        '''Test if correct imports and packages are listed.'''
        # Example for DiD which uses more packages
        method = "difference_in_differences"
        dataset_path = self.dummy_data_path 
        variables = {
            "treatment_variable": "treatment", 
            "outcome_variable": "outcome",
            "covariates": [],
            "time_variable": "time", # Need to add time/group to dummy data for real test
            "group_variable": "unit"
        }
        method_info = { # Pass necessary info for DiD generation
            "method": method,
            "time_var": "time", 
            "group_var": "unit"
        }
        validation_result = {"valid": True}

        result = generate_code(method, dataset_path, variables)

        self.assertIn("import seaborn as sns", result["imports"])
        self.assertIn("statsmodels", result["packages"])
        self.assertIn("seaborn", result["packages"])
        self.assertIn("matplotlib", result["packages"])


if __name__ == '__main__':
    unittest.main() 
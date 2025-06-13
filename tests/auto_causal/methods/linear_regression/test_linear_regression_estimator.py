import unittest
import pandas as pd
import numpy as np
from auto_causal.methods.linear_regression.estimator import estimate_effect

class TestLinearRegressionEstimator(unittest.TestCase):

    def _generate_data(
        self, 
        n=100, 
        treatment_type='binary_numeric', # binary_numeric, binary_categorical, multi_categorical, continuous
        reference_level=None, 
        seed=42
    ):
        np.random.seed(seed)
        data = pd.DataFrame()
        data['X1'] = np.random.normal(0, 1, n)
        data['X2'] = np.random.rand(n) * 10

        if treatment_type == 'binary_numeric':
            data['T'] = np.random.binomial(1, 0.5, n)
            # Y = 10 + 5*T + 3*X1 - 2*X2 + err
            data['Y'] = 10 + 5 * data['T'] + 3 * data['X1'] - 2 * data['X2'] + np.random.normal(0, 2, n)
        elif treatment_type == 'binary_categorical':
            data['T_cat'] = np.random.choice(['Control', 'Treated'], size=n, p=[0.5, 0.5])
            treatment_map = {'Control': 0, 'Treated': 1}
            data['Y'] = 10 + 5 * data['T_cat'].map(treatment_map) + 3 * data['X1'] - 2 * data['X2'] + np.random.normal(0, 2, n)
        elif treatment_type == 'multi_categorical':
            # Levels: A, B, C. Let C be reference if specified, otherwise Patsy picks one.
            levels = ['A', 'B', 'C']
            data['T_multi'] = np.random.choice(levels, size=n, p=[0.3, 0.3, 0.4])
            # Y = 10 + (5 if T=A else 0) + (-3 if T=B else 0) + 3*X1 - 2*X2 + err (effects relative to C)
            effect_A = 5
            effect_B = -3
            data['Y'] = 10 + \
                        data['T_multi'].apply(lambda x: effect_A if x == 'A' else (effect_B if x == 'B' else 0)) + \
                        3 * data['X1'] - 2 * data['X2'] + np.random.normal(0, 2, n)
        elif treatment_type == 'continuous':
            data['T_cont'] = np.random.normal(5, 2, n)
            data['Y'] = 10 + 2 * data['T_cont'] + 3 * data['X1'] - 2 * data['X2'] + np.random.normal(0, 2, n)
        
        return data

    def test_binary_numeric_treatment(self):
        df = self._generate_data(treatment_type='binary_numeric')
        results = estimate_effect(df, treatment='T', outcome='Y', covariates=['X1', 'X2'])
        self.assertIn('effect_estimate', results)
        self.assertIsNotNone(results['effect_estimate'])
        self.assertTrue('T' in results['formula'])
        self.assertFalse('C(T' in results['formula'])
        self.assertIsNone(results.get('estimated_effects_by_level'))
        self.assertAlmostEqual(results['effect_estimate'], 5, delta=1.0) # Check if close to true effect

    def test_binary_categorical_treatment(self):
        df = self._generate_data(treatment_type='binary_categorical')
        # preprocess_data will likely convert T_cat to 0/1 based on first value as reference if not specified
        # The estimator then sees it as numeric 0/1 unless it stays category and C() is used.
        # Current logic in estimator for C(T) is based on dtype and nunique.
        results = estimate_effect(df, treatment='T_cat', outcome='Y', covariates=['X1', 'X2'])
        self.assertIn('effect_estimate', results)
        self.assertIsNotNone(results['effect_estimate'])
        # Expect C(T_cat) if T_cat is object/category dtype and has 2 unique values.
        self.assertIn(f"C({df['T_cat'].name})", results['formula'])
        self.assertIsNone(results.get('estimated_effects_by_level'))
        self.assertAlmostEqual(results['effect_estimate'], 5, delta=1.5) # Wider delta due to encoding/ref choice

    def test_multi_categorical_treatment_with_reference(self):
        reference = 'C'
        df = self._generate_data(treatment_type='multi_categorical', reference_level=reference)
        results = estimate_effect(
            df, 
            treatment='T_multi', 
            outcome='Y', 
            covariates=['X1', 'X2'], 
            treatment_reference_level=reference,
            column_mappings={ # Simulate that preprocess_data did not alter T_multi's type
                'T_multi': {'original_dtype': 'object', 'transformed_as': 'original'}
            }
        )
        self.assertIn('estimated_effects_by_level', results)
        self.assertIsNotNone(results['estimated_effects_by_level'])
        self.assertEqual(results['reference_level_used'], reference)
        self.assertTrue(f"C(T_multi, Treatment(reference='{reference}'))" in results['formula'])
        self.assertIn('A', results['estimated_effects_by_level'])
        self.assertIn('B', results['estimated_effects_by_level'])
        self.assertNotIn('C', results['estimated_effects_by_level']) # Reference level should not have its own effect listed
        self.assertAlmostEqual(results['estimated_effects_by_level']['A']['estimate'], 5, delta=1.5)
        self.assertAlmostEqual(results['estimated_effects_by_level']['B']['estimate'], -3, delta=1.5)
        self.assertIsNone(results['effect_estimate']) # Main effect is None for multi-level

    def test_multi_categorical_treatment_no_reference(self):
        df = self._generate_data(treatment_type='multi_categorical')
        results = estimate_effect(
            df, 
            treatment='T_multi', 
            outcome='Y', 
            covariates=['X1', 'X2'],
            column_mappings={ # Simulate that preprocess_data did not alter T_multi's type
                'T_multi': {'original_dtype': 'object', 'transformed_as': 'original'}
            }
        )
        # Without explicit reference, Patsy picks one (usually first alphabetically: A)
        # The output structure for 'estimated_effects_by_level' would have effects relative to this implicit ref.
        # The current linear_regression_estimator.py when no ref is given AND it's categorical AND >2 levels 
        # might not populate estimated_effects_by_level clearly. It falls to single effect logic.
        # This test highlights that area. For now, we check if formula uses C() and some effect is found.
        self.assertTrue(f"C(T_multi)" in results['formula'] or "T_multi[T." in results['formula'])
        self.assertIsNotNone(results['effect_estimate']) # It will pick one of the level effects
        # A more detailed check would be needed for which specific levels are present vs implicit reference.

    def test_continuous_treatment(self):
        df = self._generate_data(treatment_type='continuous')
        results = estimate_effect(df, treatment='T_cont', outcome='Y', covariates=['X1', 'X2'])
        self.assertIn('effect_estimate', results)
        self.assertIsNotNone(results['effect_estimate'])
        self.assertTrue('T_cont' in results['formula'])
        self.assertFalse('C(T_cont' in results['formula'])
        self.assertIsNone(results.get('estimated_effects_by_level'))
        self.assertAlmostEqual(results['effect_estimate'], 2, delta=1.0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 
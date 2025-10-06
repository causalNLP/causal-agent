"""
Tests for the synthetic data generator module.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from causal_agent.synthetic.generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_observations = 100
        self.n_continuous_covars = 3
        self.n_binary_covars = 2
        self.seed = 42
    
    def test_data_generator_initialization_basic(self):
        """Test basic initialization of DataGenerator."""
        generator = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=self.seed
        )
        
        self.assertEqual(generator.n_observations, self.n_observations)
        self.assertEqual(generator.n_continuous_covars, self.n_continuous_covars)
        self.assertEqual(generator.n_binary_covars, 2)  # default value
        self.assertEqual(generator.n_covars, self.n_continuous_covars + 2)
        self.assertEqual(generator.n_treatments, 1)  # default value
        self.assertEqual(generator.true_effect, 0)  # default value
        self.assertEqual(generator.seed, self.seed)
        self.assertIsNone(generator.data)
        self.assertIsNone(generator.method)
    
    def test_data_generator_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        n_binary_covars = 3
        n_treatments = 2
        true_effect = 1.5
        heterogeneity = 1
        
        generator = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            n_binary_covars=n_binary_covars,
            n_treatments=n_treatments,
            true_effect=true_effect,
            seed=self.seed,
            heterogeneity=heterogeneity
        )
        
        self.assertEqual(generator.n_binary_covars, n_binary_covars)
        self.assertEqual(generator.n_covars, self.n_continuous_covars + n_binary_covars)
        self.assertEqual(generator.n_treatments, n_treatments)
        self.assertEqual(generator.true_effect, true_effect)
    
    def test_data_generator_initialization_with_custom_mean_covar(self):
        """Test initialization with custom mean and covariance."""
        custom_mean = np.array([1.0, 2.0, 3.0])
        custom_covar = np.eye(3)
        
        generator = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            mean=custom_mean,
            covar=custom_covar,
            seed=self.seed
        )
        
        np.testing.assert_array_equal(generator.mean, custom_mean)
        np.testing.assert_array_equal(generator.covar, custom_covar)
    
    def test_data_generator_default_mean_generation(self):
        """Test that default mean is generated when not provided."""
        generator = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=self.seed
        )
        
        self.assertIsNotNone(generator.mean)
        self.assertEqual(len(generator.mean), self.n_continuous_covars)
        self.assertTrue(all(3 <= x <= 20 for x in generator.mean))
    
    def test_data_generator_seed_reproducibility(self):
        """Test that the same seed produces reproducible results."""
        generator1 = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=self.seed
        )
        
        generator2 = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=self.seed
        )
        
        np.testing.assert_array_equal(generator1.mean, generator2.mean)
    
    def test_data_generator_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        generator1 = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=42
        )
        
        generator2 = DataGenerator(
            n_observations=self.n_observations,
            n_continuous_covars=self.n_continuous_covars,
            seed=123
        )
        
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(generator1.mean, generator2.mean))


if __name__ == '__main__':
    unittest.main()
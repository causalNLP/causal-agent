"""Synthetic data generators for consistent testing across causal_agent tests."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DatasetType(Enum):
    """Types of synthetic datasets that can be generated."""
    RCT = "randomized_controlled_trial"
    OBSERVATIONAL = "observational"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    FRONT_DOOR = "front_door"


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 500
    n_features: int = 5
    treatment_effect: float = 0.5
    noise_level: float = 0.1
    confounding_strength: float = 0.3
    instrument_strength: float = 0.8
    random_seed: int = 42
    
    # RDD specific
    cutoff_value: float = 0.0
    bandwidth: float = 2.0
    
    # DiD specific
    n_periods: int = 20
    n_units: int = 50
    treatment_start_period: int = 10
    
    # Additional parameters
    binary_treatment: bool = True
    binary_outcome: bool = False
    missing_data_rate: float = 0.0


class SyntheticDataGenerator:
    """Generator for various types of synthetic causal inference datasets."""
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """Initialize the generator with configuration."""
        self.config = config or SyntheticDataConfig()
        np.random.seed(self.config.random_seed)
    
    def generate_rct_data(self) -> pd.DataFrame:
        """Generate randomized controlled trial data."""
        n = self.config.n_samples
        n_features = self.config.n_features
        
        # Generate baseline features
        X = np.random.normal(0, 1, (n, n_features))
        
        # Random treatment assignment (key feature of RCT)
        treatment = np.random.binomial(1, 0.5, n)
        
        # Outcome depends on features and treatment
        outcome = (
            X.sum(axis=1) * 0.1 +  # Feature effects
            self.config.treatment_effect * treatment +  # Treatment effect
            np.random.normal(0, self.config.noise_level, n)  # Noise
        )
        
        # Create DataFrame
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        data['treatment'] = treatment
        data['outcome'] = outcome
        
        # Add metadata
        data.attrs['dataset_type'] = DatasetType.RCT.value
        data.attrs['true_treatment_effect'] = self.config.treatment_effect
        data.attrs['confounders'] = []  # No confounders in RCT
        
        return self._add_missing_data(data)
    
    def generate_observational_data(self) -> pd.DataFrame:
        """Generate observational data with confounding."""
        n = self.config.n_samples
        n_features = self.config.n_features
        
        # Generate confounders
        confounders = np.random.normal(0, 1, (n, n_features))
        
        # Treatment assignment depends on confounders (selection bias)
        treatment_logits = (
            self.config.confounding_strength * confounders.sum(axis=1) +
            np.random.normal(0, 0.5, n)
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on confounders and treatment
        outcome = (
            confounders.sum(axis=1) * 0.2 +  # Confounder effects
            self.config.treatment_effect * treatment +  # Treatment effect
            np.random.normal(0, self.config.noise_level, n)  # Noise
        )
        
        # Create DataFrame
        data = pd.DataFrame(confounders, columns=[f'confounder_{i}' for i in range(n_features)])
        data['treatment'] = treatment
        data['outcome'] = outcome
        
        # Add metadata
        data.attrs['dataset_type'] = DatasetType.OBSERVATIONAL.value
        data.attrs['true_treatment_effect'] = self.config.treatment_effect
        data.attrs['confounders'] = [f'confounder_{i}' for i in range(n_features)]
        
        return self._add_missing_data(data)
    
    def generate_iv_data(self) -> pd.DataFrame:
        """Generate instrumental variable data."""
        n = self.config.n_samples
        
        # Generate instrument (randomly assigned)
        instrument = np.random.binomial(1, 0.5, n)
        
        # Generate unobserved confounder
        unobserved_confounder = np.random.normal(0, 1, n)
        
        # Generate observed covariates
        observed_covariates = np.random.normal(0, 1, (n, 2))
        
        # Treatment depends on instrument and unobserved confounder
        treatment_logits = (
            self.config.instrument_strength * instrument +
            self.config.confounding_strength * unobserved_confounder +
            0.1 * observed_covariates.sum(axis=1)
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome depends on treatment and unobserved confounder (not instrument directly)
        outcome = (
            self.config.treatment_effect * treatment +
            0.5 * unobserved_confounder +  # Confounding
            0.1 * observed_covariates.sum(axis=1) +
            np.random.normal(0, self.config.noise_level, n)
        )
        
        # Create DataFrame (unobserved confounder not included)
        data = pd.DataFrame({
            'instrument': instrument,
            'covariate_0': observed_covariates[:, 0],
            'covariate_1': observed_covariates[:, 1],
            'treatment': treatment,
            'outcome': outcome
        })
        
        # Add metadata
        data.attrs['dataset_type'] = DatasetType.INSTRUMENTAL_VARIABLE.value
        data.attrs['true_treatment_effect'] = self.config.treatment_effect
        data.attrs['instrument'] = 'instrument'
        data.attrs['confounders'] = ['covariate_0', 'covariate_1']
        
        return self._add_missing_data(data)
    
    def generate_rdd_data(self) -> pd.DataFrame:
        """Generate regression discontinuity data."""
        n = self.config.n_samples
        
        # Generate running variable around cutoff
        running_var = np.random.uniform(
            -self.config.bandwidth, 
            self.config.bandwidth, 
            n
        )
        
        # Treatment assignment based on cutoff
        treatment = (running_var >= self.config.cutoff_value).astype(int)
        
        # Generate additional covariates
        covariates = np.random.normal(0, 1, (n, 2))
        
        # Outcome with discontinuity at cutoff
        # Smooth function of running variable + jump at cutoff
        outcome = (
            0.3 * running_var +  # Smooth trend
            0.05 * running_var**2 +  # Non-linear trend
            self.config.treatment_effect * treatment +  # Discontinuity
            0.1 * covariates.sum(axis=1) +  # Covariate effects
            np.random.normal(0, self.config.noise_level, n)
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'running_var': running_var,
            'covariate_0': covariates[:, 0],
            'covariate_1': covariates[:, 1],
            'treatment': treatment,
            'outcome': outcome
        })
        
        # Add metadata
        data.attrs['dataset_type'] = DatasetType.REGRESSION_DISCONTINUITY.value
        data.attrs['true_treatment_effect'] = self.config.treatment_effect
        data.attrs['running_variable'] = 'running_var'
        data.attrs['cutoff'] = self.config.cutoff_value
        
        return self._add_missing_data(data)
    
    def generate_did_data(self) -> pd.DataFrame:
        """Generate difference-in-differences panel data."""
        n_units = self.config.n_units
        n_periods = self.config.n_periods
        treatment_start = self.config.treatment_start_period
        
        data = []
        
        for unit in range(n_units):
            # Determine if unit is in treatment group
            treated_unit = unit < n_units // 2
            
            # Unit-specific fixed effect
            unit_effect = np.random.normal(0, 0.5)
            
            for period in range(n_periods):
                # Time-specific fixed effect
                time_effect = 0.02 * period + np.random.normal(0, 0.1)
                
                # Treatment indicator
                post_treatment = period >= treatment_start
                treatment = 1 if (treated_unit and post_treatment) else 0
                
                # Generate time-varying covariates
                covariate = np.random.normal(0, 1)
                
                # Outcome with parallel trends assumption
                outcome = (
                    unit_effect +  # Unit fixed effect
                    time_effect +  # Time fixed effect
                    0.1 * covariate +  # Covariate effect
                    self.config.treatment_effect * treatment +  # Treatment effect
                    np.random.normal(0, self.config.noise_level)
                )
                
                data.append({
                    'unit': unit,
                    'period': period,
                    'treated_unit': treated_unit,
                    'post_treatment': post_treatment,
                    'treatment': treatment,
                    'covariate': covariate,
                    'outcome': outcome
                })
        
        df = pd.DataFrame(data)
        
        # Add metadata
        df.attrs['dataset_type'] = DatasetType.DIFFERENCE_IN_DIFFERENCES.value
        df.attrs['true_treatment_effect'] = self.config.treatment_effect
        df.attrs['treatment_start_period'] = treatment_start
        df.attrs['n_treated_units'] = n_units // 2
        
        return df
    
    def generate_front_door_data(self) -> pd.DataFrame:
        """Generate data suitable for front-door criterion."""
        n = self.config.n_samples
        
        # Generate unobserved confounder
        unobserved_confounder = np.random.normal(0, 1, n)
        
        # Generate observed covariates
        observed_covariates = np.random.normal(0, 1, (n, 2))
        
        # Treatment depends on unobserved confounder
        treatment_logits = (
            self.config.confounding_strength * unobserved_confounder +
            0.1 * observed_covariates.sum(axis=1)
        )
        treatment_prob = 1 / (1 + np.exp(-treatment_logits))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Mediator depends on treatment (not on unobserved confounder)
        mediator_logits = (
            0.8 * treatment +  # Strong effect of treatment on mediator
            0.1 * observed_covariates.sum(axis=1)
        )
        mediator_prob = 1 / (1 + np.exp(-mediator_logits))
        mediator = np.random.binomial(1, mediator_prob)
        
        # Outcome depends on mediator and unobserved confounder (not directly on treatment)
        outcome = (
            self.config.treatment_effect * mediator +  # Effect through mediator
            0.5 * unobserved_confounder +  # Confounding
            0.1 * observed_covariates.sum(axis=1) +
            np.random.normal(0, self.config.noise_level, n)
        )
        
        # Create DataFrame (unobserved confounder not included)
        data = pd.DataFrame({
            'covariate_0': observed_covariates[:, 0],
            'covariate_1': observed_covariates[:, 1],
            'treatment': treatment,
            'mediator': mediator,
            'outcome': outcome
        })
        
        # Add metadata
        data.attrs['dataset_type'] = DatasetType.FRONT_DOOR.value
        data.attrs['true_treatment_effect'] = self.config.treatment_effect
        data.attrs['mediator'] = 'mediator'
        data.attrs['confounders'] = ['covariate_0', 'covariate_1']
        
        return self._add_missing_data(data)
    
    def _add_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add missing data according to configuration."""
        if self.config.missing_data_rate > 0:
            n_missing = int(len(data) * self.config.missing_data_rate)
            missing_indices = np.random.choice(len(data), n_missing, replace=False)
            
            # Randomly select columns to have missing values (exclude treatment and outcome)
            feature_cols = [col for col in data.columns if col not in ['treatment', 'outcome']]
            if feature_cols:
                missing_col = np.random.choice(feature_cols)
                data.loc[missing_indices, missing_col] = np.nan
        
        return data
    
    def generate_dataset(self, dataset_type: DatasetType) -> pd.DataFrame:
        """Generate dataset of specified type."""
        generators = {
            DatasetType.RCT: self.generate_rct_data,
            DatasetType.OBSERVATIONAL: self.generate_observational_data,
            DatasetType.INSTRUMENTAL_VARIABLE: self.generate_iv_data,
            DatasetType.REGRESSION_DISCONTINUITY: self.generate_rdd_data,
            DatasetType.DIFFERENCE_IN_DIFFERENCES: self.generate_did_data,
            DatasetType.FRONT_DOOR: self.generate_front_door_data,
        }
        
        if dataset_type not in generators:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return generators[dataset_type]()
    
    def generate_multiple_datasets(self, 
                                 dataset_types: List[DatasetType],
                                 configs: Optional[List[SyntheticDataConfig]] = None) -> Dict[str, pd.DataFrame]:
        """Generate multiple datasets with different configurations."""
        datasets = {}
        
        if configs is None:
            configs = [self.config] * len(dataset_types)
        
        for dataset_type, config in zip(dataset_types, configs):
            # Temporarily update config
            original_config = self.config
            self.config = config
            
            # Generate dataset
            datasets[dataset_type.value] = self.generate_dataset(dataset_type)
            
            # Restore original config
            self.config = original_config
        
        return datasets


def create_benchmark_datasets() -> Dict[str, pd.DataFrame]:
    """Create a standard set of benchmark datasets for testing."""
    generator = SyntheticDataGenerator()
    
    # Standard configurations for different scenarios
    configs = {
        'small_rct': SyntheticDataConfig(n_samples=100, treatment_effect=0.5),
        'large_rct': SyntheticDataConfig(n_samples=1000, treatment_effect=0.3),
        'weak_effect_obs': SyntheticDataConfig(n_samples=500, treatment_effect=0.1, confounding_strength=0.5),
        'strong_effect_obs': SyntheticDataConfig(n_samples=500, treatment_effect=0.8, confounding_strength=0.3),
        'weak_instrument': SyntheticDataConfig(n_samples=300, instrument_strength=0.3),
        'strong_instrument': SyntheticDataConfig(n_samples=300, instrument_strength=1.0),
        'narrow_rdd': SyntheticDataConfig(n_samples=200, bandwidth=1.0),
        'wide_rdd': SyntheticDataConfig(n_samples=500, bandwidth=3.0),
        'short_panel': SyntheticDataConfig(n_periods=10, n_units=20),
        'long_panel': SyntheticDataConfig(n_periods=50, n_units=100),
    }
    
    datasets = {}
    
    # Generate RCT datasets
    for name, config in [('small_rct', configs['small_rct']), ('large_rct', configs['large_rct'])]:
        generator.config = config
        datasets[name] = generator.generate_rct_data()
    
    # Generate observational datasets
    for name, config in [('weak_effect_obs', configs['weak_effect_obs']), 
                        ('strong_effect_obs', configs['strong_effect_obs'])]:
        generator.config = config
        datasets[name] = generator.generate_observational_data()
    
    # Generate IV datasets
    for name, config in [('weak_instrument', configs['weak_instrument']), 
                        ('strong_instrument', configs['strong_instrument'])]:
        generator.config = config
        datasets[name] = generator.generate_iv_data()
    
    # Generate RDD datasets
    for name, config in [('narrow_rdd', configs['narrow_rdd']), 
                        ('wide_rdd', configs['wide_rdd'])]:
        generator.config = config
        datasets[name] = generator.generate_rdd_data()
    
    # Generate DiD datasets
    for name, config in [('short_panel', configs['short_panel']), 
                        ('long_panel', configs['long_panel'])]:
        generator.config = config
        datasets[name] = generator.generate_did_data()
    
    return datasets
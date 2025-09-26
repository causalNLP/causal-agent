Synthetic Data Generation System
===============================

This document provides comprehensive guidance on the synthetic data generation system used for testing, validation, and benchmarking of causal inference methods in CAIS. The system is a critical component that enables rigorous testing of the autonomous agent's decision-making capabilities and method selection logic.

.. contents::
   :local:
   :depth: 3

Overview
--------

The synthetic data generation system is a foundational component of CAIS that enables:

* **Decision Tree Validation**: Test the agent's method selection logic with known ground truth scenarios
* **Method Performance Testing**: Validate causal inference methods with controlled data generation parameters
* **Assumption Violation Testing**: Generate data that violates specific method assumptions to test robustness
* **Agent Workflow Testing**: Create comprehensive test scenarios for the complete autonomous analysis pipeline
* **Educational Examples**: Provide clear examples for tutorials and documentation with known causal relationships
* **Benchmarking**: Create standardized datasets for comparing method performance across different scenarios

The system generates realistic datasets that mirror real-world causal inference challenges while maintaining known causal relationships, enabling validation of both individual methods and the agent's decision-making process.

System Architecture and Decision Tree Integration
-------------------------------------------------

The synthetic data generation system is tightly integrated with CAIS's decision tree logic, enabling comprehensive testing of the autonomous agent's method selection capabilities.

.. mermaid::

   graph TB
       subgraph "Decision Tree Testing Framework"
           SCENARIOS[Scenario Definitions]
           GENERATORS[Method-Specific Generators]
           VALIDATION[Ground Truth Validation]
       end
       
       subgraph "Agent Decision Points"
           EXPERIMENTAL[Experimental Design Detection]
           TEMPORAL[Temporal Structure Analysis]
           CONFOUNDING[Confounding Assessment]
           INSTRUMENTS[Instrument Validation]
       end
       
       subgraph "Method Generators"
           RCT[RCT Generator]
           DID[DiD Generator]
           IV[IV Generator]
           RDD[RDD Generator]
           PS[Propensity Score Generator]
           MULTI[Multi-Treatment RCT]
           FRONT[Front-Door Generator]
       end
       
       subgraph "Testing Scenarios"
           CANONICAL[Canonical Scenarios]
           VIOLATIONS[Assumption Violations]
           EDGE[Edge Cases]
           MIXED[Mixed Method Scenarios]
       end
       
       SCENARIOS --> GENERATORS
       GENERATORS --> RCT
       GENERATORS --> DID
       GENERATORS --> IV
       GENERATORS --> RDD
       GENERATORS --> PS
       GENERATORS --> MULTI
       GENERATORS --> FRONT
       
       GENERATORS --> EXPERIMENTAL
       GENERATORS --> TEMPORAL
       GENERATORS --> CONFOUNDING
       GENERATORS --> INSTRUMENTS
       
       VALIDATION --> CANONICAL
       VALIDATION --> VIOLATIONS
       VALIDATION --> EDGE
       VALIDATION --> MIXED

Decision Tree Validation Through Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synthetic data system validates the agent's decision tree logic by generating datasets with specific characteristics that should trigger particular method selections:

**Experimental Design Detection**:
   - RCT data with random treatment assignment tests the agent's ability to detect experimental designs
   - Multi-treatment RCT data validates handling of complex experimental structures
   - Quasi-experimental data tests the distinction between experimental and observational studies

**Temporal Structure Recognition**:
   - Panel data with treatment timing variation tests DiD method selection
   - Cross-sectional data ensures DiD is not incorrectly selected
   - Time-series data with interventions validates temporal analysis capabilities

**Confounding Assessment**:
   - Observational data with measured confounders tests propensity score method selection
   - Data with unmeasured confounding validates the agent's caution in method selection
   - Instrumental variable scenarios test the agent's ability to leverage instruments

**Method Exclusion Logic**:
   - Weak instrument scenarios test first-stage F-statistic thresholds
   - Assumption violation scenarios validate the agent's diagnostic capabilities
   - Edge cases test fallback method selection when primary methods fail

Data Generation Framework
------------------------

Core Components
~~~~~~~~~~~~~~~

The synthetic data generation framework consists of several interconnected components that work together to create comprehensive test scenarios for the CAIS autonomous agent.

.. mermaid::

   graph TB
       subgraph "Generation Pipeline"
           CONFIG[Configuration System]
           BASE[Base Data Generator]
           METHODS[Method-Specific Generators]
           CONTEXT[Context Generation]
       end
       
       subgraph "Method Generators"
           RCT[RCT Generator]
           MULTI[Multi-Treatment RCT]
           DID_CAN[Canonical DiD]
           DID_TWFE[TWFE DiD]
           IV[IV Generator]
           IV_ENC[Encouragement Design]
           RDD[RDD Generator]
           PSM[PSM Generator]
           PSW[PSW Generator]
           FRONT[Front-Door Generator]
       end
       
       subgraph "Output Processing"
           STORAGE[Data Storage]
           METADATA[Metadata Management]
           CONTEXT_GEN[Context Generation]
           VALIDATION[Ground Truth Validation]
       end
       
       subgraph "Testing Integration"
           DECISION[Decision Tree Testing]
           AGENT[Agent Workflow Testing]
           BENCHMARK[Performance Benchmarking]
       end
       
       CONFIG --> BASE
       BASE --> METHODS
       METHODS --> RCT
       METHODS --> MULTI
       METHODS --> DID_CAN
       METHODS --> DID_TWFE
       METHODS --> IV
       METHODS --> IV_ENC
       METHODS --> RDD
       METHODS --> PSM
       METHODS --> PSW
       METHODS --> FRONT
       
       METHODS --> STORAGE
       STORAGE --> METADATA
       STORAGE --> CONTEXT_GEN
       STORAGE --> VALIDATION
       
       VALIDATION --> DECISION
       VALIDATION --> AGENT
       VALIDATION --> BENCHMARK

Configuration System
~~~~~~~~~~~~~~~~~~~~

The configuration system (``data_generation/settings.sh``) provides centralized parameter management for all data generation processes:

.. code-block:: bash

   # Dataset sizes for different methods
   export RCT_SIZE=10
   export MULTI_RCT_SIZE=5
   export CANONICAL_DID_SIZE=5
   export TWFE_DID_SIZE=5
   export OBSERVATIONAL_SIZE=5
   export IV_SIZE=5
   export ENCOURAGEMENT_SIZE=5
   export RDD_SIZE=5
   
   # Observation counts
   export MIN_OBS=300
   export MAX_OBS=500
   export DEFAULT_OBS=1000
   
   # Covariate specifications
   export N_CONTINUOUS=5
   export N_BINARY=4
   
   # Method-specific parameters
   export MAX_TREATMENTS=5      # Multi-treatment RCT
   export MAX_PERIODS=10        # TWFE DiD
   export CUTOFF=25            # RDD cutoff range

This configuration system enables:

* **Consistent Parameter Management**: Centralized control over data generation parameters
* **Scalable Testing**: Easy adjustment of dataset sizes for different testing scenarios
* **Method-Specific Tuning**: Tailored parameters for each causal inference method
* **Reproducible Results**: Fixed parameters ensure consistent test outcomes

Base Data Generator Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DataGenerator`` base class provides common functionality for all method-specific generators:

.. code-block:: python

   class DataGenerator:
       """Base class for generating synthetic data with common functionality"""
       
       def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, 
                    mean=None, covar=None, n_treatments=1, true_effect=0, 
                    seed=111, heterogeneity=0):
           # Initialize parameters and random state
           np.random.seed(seed)
           self.n_observations = n_observations
           self.n_continuous_covars = n_continuous_covars
           self.n_covars = n_continuous_covars + n_binary_covars
           self.true_effect = true_effect
           self.method = None  # Set by subclasses
           
           # Generate covariate parameters
           if mean is None:
               self.mean = np.random.randint(3, 20, size=self.n_continuous_covars)
           if covar is None:
               self.covar = np.identity(self.n_continuous_covars)
       
       def generate_covariates(self):
           """Generate correlated continuous and binary covariates"""
           # Continuous covariates from multivariate normal
           X_c = np.random.multivariate_normal(
               mean=self.mean, 
               cov=self.covar,
               size=self.n_observations
           )
           
           # Binary covariates from binomial
           p = np.random.uniform(0.3, 0.7)
           X_b = np.random.binomial(
               1, p, 
               size=(self.n_observations, self.n_binary_covars)
           ).astype(int)
           
           # Combine and discretize
           covariates = np.hstack((X_c, X_b))
           return covariates.astype(int)
       
       def generate_data(self):
           """Generate complete synthetic dataset (implemented by subclasses)"""
           raise NotImplementedError("Invoke the method in the subclass")
       
       def test_data(self, print_=False):
           """Test generated data using appropriate method"""
           raise NotImplementedError("This method should be overridden by subclasses")
       
       def save_data(self, folder, filename):
           """Save generated data as CSV file"""
           if self.data is None:
               raise ValueError("Data not generated yet. Please generate data first.")
           
           path = Path(folder)
           path.mkdir(parents=True, exist_ok=True)
           if not filename.endswith('.csv'):
               filename += '.csv'
           self.data.to_csv(path / filename, index=False)

Key Features:

* **Reproducible Generation**: Seed-based random number generation ensures consistent results
* **Flexible Covariate Structure**: Configurable continuous and binary covariates with realistic correlations
* **Method-Agnostic Base**: Common functionality shared across all causal inference methods
* **Validation Integration**: Built-in testing capabilities for generated data
* **Standardized Output**: Consistent data format and storage mechanisms

Base Data Generator
~~~~~~~~~~~~~~~~~~~

The foundation of the synthetic data system:

.. code-block:: python

   # causal_agent/synthetic/generator.py
   
   from abc import ABC, abstractmethod
   import numpy as np
   import pandas as pd
   from typing import Dict, List, Any, Optional, Tuple
   from dataclasses import dataclass
   
   @dataclass
   class DataGenerationConfig:
       """Configuration for synthetic data generation"""
       n_observations: int = 1000
       n_continuous_covars: int = 3
       n_binary_covars: int = 2
       true_effect: float = 1.0
       noise_level: float = 1.0
       seed: int = 42
       heterogeneity: bool = False
       
   class BaseDataGenerator(ABC):
       """
       Base class for synthetic data generation with common functionality.
       
       This class provides the foundation for all method-specific data generators,
       including covariate generation, noise modeling, and metadata management.
       """
       
       def __init__(self, config: DataGenerationConfig):
           self.config = config
           self.data = None
           self.metadata = {}
           self.true_parameters = {}
           
           # Set random seed for reproducibility
           np.random.seed(config.seed)
           
           # Initialize covariate parameters
           self.covariate_means = np.random.uniform(-2, 2, config.n_continuous_covars)
           self.covariate_cov = self._generate_covariance_matrix()
       
       def _generate_covariance_matrix(self) -> np.ndarray:
           """Generate realistic covariance matrix for covariates"""
           n_vars = self.config.n_continuous_covars
           
           # Generate correlation matrix
           correlations = np.random.uniform(-0.5, 0.5, size=(n_vars, n_vars))
           correlations = (correlations + correlations.T) / 2  # Make symmetric
           np.fill_diagonal(correlations, 1.0)
           
           # Ensure positive definite
           eigenvals, eigenvecs = np.linalg.eigh(correlations)
           eigenvals = np.maximum(eigenvals, 0.1)  # Ensure positive eigenvalues
           correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
           
           # Convert to covariance matrix
           std_devs = np.random.uniform(0.5, 2.0, n_vars)
           covariance = np.outer(std_devs, std_devs) * correlations
           
           return covariance
       
       def generate_covariates(self) -> np.ndarray:
           """Generate correlated continuous covariates"""
           return np.random.multivariate_normal(
               mean=self.covariate_means,
               cov=self.covariate_cov,
               size=self.config.n_observations
           )
       
       def generate_binary_covariates(self) -> np.ndarray:
           """Generate binary covariates"""
           return np.random.binomial(
               1, 0.5, 
               size=(self.config.n_observations, self.config.n_binary_covars)
           )
       
       def add_noise(self, signal: np.ndarray) -> np.ndarray:
           """Add noise to signal with specified noise level"""
           noise = np.random.normal(0, self.config.noise_level, len(signal))
           return signal + noise
       
       @abstractmethod
       def generate_treatment(self, covariates: np.ndarray) -> np.ndarray:
           """Generate treatment assignment (method-specific)"""
           pass
       
       @abstractmethod
       def generate_outcome(
           self, 
           treatment: np.ndarray, 
           covariates: np.ndarray
       ) -> np.ndarray:
           """Generate outcome variable (method-specific)"""
           pass
       
       @abstractmethod
       def get_method_name(self) -> str:
           """Return the causal method this generator is designed for"""
           pass
       
       def generate_data(self) -> pd.DataFrame:
           """Generate complete synthetic dataset"""
           # Generate covariates
           continuous_covars = self.generate_covariates()
           binary_covars = self.generate_binary_covariates()
           
           # Generate treatment
           treatment = self.generate_treatment(continuous_covars)
           
           # Generate outcome
           outcome = self.generate_outcome(treatment, continuous_covars)
           
           # Create DataFrame
           data = pd.DataFrame()
           
           # Add continuous covariates
           for i in range(self.config.n_continuous_covars):
               data[f'X{i+1}'] = continuous_covars[:, i]
           
           # Add binary covariates
           for i in range(self.config.n_binary_covars):
               data[f'B{i+1}'] = binary_covars[:, i]
           
           # Add treatment and outcome
           data['treatment'] = treatment
           data['outcome'] = outcome
           
           # Store metadata
           self.metadata = {
               'method': self.get_method_name(),
               'n_observations': self.config.n_observations,
               'n_continuous_covars': self.config.n_continuous_covars,
               'n_binary_covars': self.config.n_binary_covars,
               'true_effect': self.config.true_effect,
               'noise_level': self.config.noise_level,
               'seed': self.config.seed,
               'heterogeneity': self.config.heterogeneity
           }
           
           self.data = data
           return data
       
       def get_true_parameters(self) -> Dict[str, Any]:
           """Return true parameters for validation"""
           return {
               'true_effect': self.config.true_effect,
               'treatment_variable': 'treatment',
               'outcome_variable': 'outcome',
               'covariates': [f'X{i+1}' for i in range(self.config.n_continuous_covars)] +
                           [f'B{i+1}' for i in range(self.config.n_binary_covars)],
               'method': self.get_method_name(),
               **self.true_parameters
           }
       
       def save_data(self, filepath: str, include_metadata: bool = True):
           """Save generated data and metadata"""
           if self.data is None:
               raise ValueError("No data generated. Call generate_data() first.")
           
           # Save data
           self.data.to_csv(filepath, index=False)
           
           # Save metadata
           if include_metadata:
               metadata_path = filepath.replace('.csv', '_metadata.json')
               import json
               with open(metadata_path, 'w') as f:
                   json.dump({
                       'metadata': self.metadata,
                       'true_parameters': self.get_true_parameters()
                   }, f, indent=2)

Method-Specific Generators and Decision Tree Testing
---------------------------------------------------

Each generator is designed to create data that tests specific aspects of the CAIS decision tree logic and method selection capabilities.

Randomized Controlled Trial (RCT) Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RCT generator creates data with random treatment assignment, testing the agent's ability to detect experimental designs and select appropriate analysis methods.

**Decision Tree Testing**:
   - Tests detection of random treatment assignment
   - Validates selection of simple difference-in-means analysis
   - Confirms rejection of more complex methods when randomization is present

.. code-block:: python

   class RCTGenerator(DataGenerator):
       """Generate synthetic data for Randomized Controlled Trials"""
       
       def generate_data(self):
           X = self.generate_covariates()
           cols = [f"X{i+1}" for i in range(self.n_covars)]
           df = pd.DataFrame(X, columns=cols)
           
           # Pure random assignment - key for decision tree testing
           df['D'] = np.random.binomial(1, 0.5, size=self.n_observations)
           
           # Outcome generation with treatment effect
           vec = np.random.uniform(0, 1, size=self.n_covars)
           intercept = np.random.normal(50, 3)
           noise = np.random.normal(0, 1, size=self.n_observations)
           df['Y'] = (intercept + X.dot(vec) + 
                     self.true_effect * df['D'] + noise)
           
           self.data = df
           return df
       
       def test_data(self, print_=False):
           """Validate using simple OLS regression"""
           model = smf.ols('Y ~ D', data=self.data).fit()
           est = model.params['D']
           conf_int = model.conf_int().loc['D']
           
           result = f"TRUE ATE: {self.true_effect:.3f}, ESTIMATED ATE: {est:.3f}, " \
                   f"95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
           return result

**Agent Testing Scenarios**:
   - **Random Assignment Detection**: Agent should identify random treatment assignment
   - **Method Selection**: Should select difference-in-means or simple regression
   - **Covariate Handling**: Should recognize that covariate adjustment is optional but can improve precision

Multi-Treatment RCT Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests the agent's handling of complex experimental designs with multiple treatment arms.

**Decision Tree Testing**:
   - Tests detection of multi-arm experimental designs
   - Validates handling of multiple treatment comparisons
   - Confirms appropriate statistical adjustments for multiple comparisons

.. code-block:: python

   class MultiTreatRCTGenerator(DataGenerator):
       """Generate synthetic data for multi-treatment RCTs"""
       
       def __init__(self, n_observations, n_continuous_covars, n_treatments, 
                    true_effect_vec=None, **kwargs):
           super().__init__(n_observations, n_continuous_covars, **kwargs)
           self.n_treatments = n_treatments
           self.true_effect_vec = true_effect_vec or np.zeros(n_treatments)
           
       def generate_data(self):
           X = self.generate_covariates()
           cols = [f"X{i+1}" for i in range(self.n_covars)]
           df = pd.DataFrame(X, columns=cols)
           
           # Multi-arm randomization
           df['D'] = np.random.randint(0, self.n_treatments+1, 
                                      size=self.n_observations)
           
           # Treatment effects vary by arm
           treatment_effects = np.array(self.true_effect_vec)
           df['treat_effect'] = treatment_effects[df['D']]
           
           # Outcome generation
           vec = np.random.uniform(0, 1, size=self.n_covars)
           intercept = np.random.normal(50, 3)
           noise = np.random.normal(0, 1, size=self.n_observations)
           df['Y'] = intercept + X.dot(vec) + df['treat_effect'] + noise
           
           df.drop(columns='treat_effect', inplace=True)
           self.data = df
           return df

**Agent Testing Scenarios**:
   - **Multi-Arm Recognition**: Agent should detect multiple treatment groups
   - **Comparison Strategy**: Should handle pairwise comparisons appropriately
   - **Statistical Power**: Should account for reduced power in multi-arm designs

Difference-in-Differences (DiD) Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two DiD generators test different aspects of temporal analysis and panel data handling.

**Canonical DiD Generator**

Tests the agent's ability to detect and analyze simple before-after treatment scenarios.

**Decision Tree Testing**:
   - Tests detection of panel structure with treatment timing
   - Validates parallel trends assumption checking
   - Confirms selection of DiD over other methods when appropriate

.. code-block:: python

   class DiDGenerator(DataGenerator):
       """Generate synthetic data for Difference-in-Differences analysis"""
       
       def canonical_did_model(self):
           """Classical 2x2 DiD with pre/post and treatment/control"""
           # Treatment assignment
           frac_treated = np.random.uniform(0.35, 0.65)
           n_treated = int(frac_treated * self.n_observations)
           treatment_status = np.zeros(self.n_observations, dtype=int)
           treatment_status[:n_treated] = 1
           np.random.shuffle(treatment_status)
           
           # Generate pre and post periods
           X = self.generate_covariates()
           cols = [f"X{i+1}" for i in range(self.n_covars)]
           covar_df = pd.DataFrame(X, columns=cols)
           
           # Time-invariant treatment effect and time effect
           treat_effect = np.random.normal(0, 1)
           time_effect = np.random.normal(0, 1)
           
           # Pre-period data
           pre_outcome = (intercept + covar_term + pre_noise + 
                         treat_effect * treatment_status)
           pre_data = pd.DataFrame({
               'unit_id': unit_ids, 'post': 0, 'D': treatment_status,
               'Y': pre_outcome
           })
           
           # Post-period data with treatment effect
           post_outcome = (intercept + time_effect + covar_term + 
                          self.true_effect * treatment_status +
                          treat_effect * treatment_status + post_noise)
           post_data = pd.DataFrame({
               'unit_id': unit_ids, 'post': 1, 'D': treatment_status,
               'Y': post_outcome
           })
           
           # Combine periods
           df = pd.concat([pre_data, post_data], ignore_index=True)
           return df.merge(covar_df, left_on="unit_id", right_index=True)

**Two-Way Fixed Effects (TWFE) DiD Generator**

Tests the agent's handling of staggered treatment adoption and complex panel structures.

**Decision Tree Testing**:
   - Tests detection of staggered treatment timing
   - Validates handling of multiple time periods
   - Confirms appropriate use of fixed effects

.. code-block:: python

   def twfe_model(self):
       """Generate panel data for Two-Way Fixed Effects DiD"""
       # Create panel structure
       unit_ids = np.arange(1, self.n_observations + 1)
       time_periods = np.arange(0, self.n_periods)
       
       df = pd.DataFrame([(i, t) for i in unit_ids for t in time_periods],
                        columns=["unit", "time"])
       
       # Staggered treatment adoption
       frac_treated = np.random.uniform(0.35, 0.65)
       n_treated = int(frac_treated * self.n_observations)
       treated_units = np.random.choice(unit_ids, size=n_treated, replace=False)
       treatment_start = {unit: np.random.randint(1, self.n_periods) 
                         for unit in treated_units}
       
       # Treatment indicator
       df["treat_post"] = df.apply(
           lambda row: int(row["unit"] in treatment_start and
                          row["time"] >= treatment_start[row["unit"]]), 
           axis=1
       )
       
       # Fixed effects and outcome generation
       unit_effects = dict(zip(unit_ids, np.random.normal(0, 1.0, self.n_observations)))
       time_effects = dict(zip(time_periods, np.random.normal(0, 1, len(time_periods))))
       
       df["Y"] = (intercept + covar_term + 
                 df["unit"].map(unit_effects) + 
                 df["time"].map(time_effects) + 
                 self.true_effect * df["treat_post"] + noise)
       
       return df

**Agent Testing Scenarios**:
   - **Panel Detection**: Agent should identify panel data structure
   - **Treatment Timing**: Should detect staggered vs. simultaneous treatment
   - **Fixed Effects**: Should include appropriate fixed effects in analysis
   - **Parallel Trends**: Should test parallel trends assumption when possible

Instrumental Variables (IV) Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two IV generators test different aspects of instrumental variable analysis and endogeneity handling.

**Standard IV Generator**

Tests the agent's ability to detect and utilize instrumental variables for endogenous treatments.

**Decision Tree Testing**:
   - Tests detection of potential endogeneity
   - Validates instrument strength assessment (first-stage F-statistic)
   - Confirms appropriate use of 2SLS estimation

.. code-block:: python

   class IVGenerator(DataGenerator):
       """Generate synthetic data for Instrumental Variables analysis"""
       
       def generate_data(self):
           X = self.generate_covariates()
           
           # Instrument (exogenous)
           Z = np.random.normal(mean, 2, size=self.n_observations).astype(int)
           
           # Unobserved confounder (creates endogeneity)
           U = np.random.normal(0, 1, size=self.n_observations)
           
           # Endogenous treatment
           vec1 = np.random.normal(0, 0.5, size=self.n_covars)
           intercept1 = np.random.normal(30, 2)
           D = (self.alpha * Z + X @ vec1 + 
               np.random.normal(size=self.n_observations) + 
               intercept1)
           
           if not self.encouragement:
               D = D + self.beta_d * U  # Add endogeneity
           
           # Outcome with confounding
           intercept2 = np.random.normal(50, 3)
           vec2 = np.random.normal(0, 0.5, size=self.n_covars)
           Y = (self.true_effect * D + X @ vec2 + 
               np.random.normal(size=self.n_observations) + intercept2)
           
           if not self.encouragement:
               Y = Y + self.beta_y * U  # Add confounding
           
           df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(self.n_covars)])
           df['Z'] = Z
           df['D'] = D.astype(int)
           df['Y'] = Y
           
           self.data = df
           return df

**Encouragement Design Generator**

Tests the agent's handling of encouragement designs and compliance issues.

**Decision Tree Testing**:
   - Tests detection of encouragement design structure
   - Validates handling of partial compliance
   - Confirms appropriate LATE (Local Average Treatment Effect) interpretation

**Agent Testing Scenarios**:
   - **Instrument Detection**: Agent should identify potential instruments (Z variable)
   - **Strength Assessment**: Should calculate and evaluate first-stage F-statistic
   - **Endogeneity Testing**: Should test for endogeneity when possible
   - **Method Selection**: Should choose IV over OLS when endogeneity is detected

Regression Discontinuity (RDD) Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests the agent's ability to detect and analyze regression discontinuity designs.

**Decision Tree Testing**:
   - Tests detection of running variable and cutoff
   - Validates bandwidth selection for local analysis
   - Confirms appropriate polynomial specification

.. code-block:: python

   class RDDGenerator(DataGenerator):
       """Generate synthetic data for Regression Discontinuity Design"""
       
       def generate_data(self):
           X = self.generate_covariates()
           cols = [f"X{i+1}" for i in range(self.n_covars)]
           df = pd.DataFrame(X, columns=cols)
           
           # Running variable around cutoff
           df['running_X'] = (np.random.normal(0, 2, size=self.n_observations) + 
                             self.cutoff)
           
           # Sharp discontinuity in treatment
           df['D'] = (df['running_X'] >= self.cutoff).astype(int)
           
           # Outcome with smooth function and discontinuity
           df['running_centered'] = df['running_X'] - self.cutoff
           
           # Different slopes above and below cutoff
           m_below = 1.5
           m_above = 0.8
           
           df["Y"] = (intercept + self.true_effect * df['D'] + 
                     m_below * df['running_centered'] * (1 - df['D']) +  
                     m_above * df['running_centered'] * df['D'] +  
                     X @ coeffs + 
                     np.random.normal(0, 0.5, size=self.n_observations))
           
           self.data = df[[col for col in df.columns if col != 'running_centered']]
           return self.data

**Agent Testing Scenarios**:
   - **Discontinuity Detection**: Agent should identify running variable and cutoff
   - **Bandwidth Selection**: Should choose appropriate bandwidth for analysis
   - **Specification Testing**: Should test for appropriate polynomial order
   - **Validity Checks**: Should perform density and covariate balance tests

Propensity Score Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two generators test different propensity score methods and observational data analysis.

**Propensity Score Matching (PSM) Generator**

Tests the agent's ability to handle selection bias through matching methods.

**Decision Tree Testing**:
   - Tests detection of observational data with selection bias
   - Validates propensity score estimation and matching procedures
   - Confirms appropriate balance checking

.. code-block:: python

   class PSMGenerator(ObservationalDataGenerator):
       """Generate synthetic data for Propensity Score Matching"""
       
       def test_data(self, print_=False):
           """Test using propensity score matching"""
           lr = LogisticRegression(solver='lbfgs')
           X = self.data[[f"X{i+1}" for i in range(self.n_covars)]]
           lr.fit(X, self.data['D'])
           ps_hat = lr.predict_proba(X)[:, 1]
           
           # Perform 1:1 nearest neighbor matching
           treated = self.data[self.data['D'] == 1]
           control = self.data[self.data['D'] == 0]
           
           match_idxs = [np.abs(ps_hat[control.index] - ps_hat[i]).argmin() 
                        for i in treated.index]
           matches = control.iloc[match_idxs]
           
           # Calculate ATT
           att = treated['Y'].mean() - matches['Y'].mean()
           
           result = f"Estimated ATT (matching): {att:.3f} | True: {self.true_effect}"
           return result

**Propensity Score Weighting (PSW) Generator**

Tests the agent's ability to use inverse probability weighting for causal inference.

**Decision Tree Testing**:
   - Tests detection of observational data requiring reweighting
   - Validates inverse probability weighting procedures
   - Confirms appropriate weight calculation and trimming

**Agent Testing Scenarios**:
   - **Selection Bias Detection**: Agent should identify potential confounding
   - **Propensity Score Estimation**: Should estimate propensity scores appropriately
   - **Method Choice**: Should choose between matching and weighting based on data characteristics
   - **Balance Assessment**: Should check covariate balance after adjustment

Front-Door Criterion Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests the agent's ability to handle mediation analysis and front-door identification.

**Decision Tree Testing**:
   - Tests detection of mediation structure (D → M → Y)
   - Validates front-door criterion application
   - Confirms appropriate sequential regression approach

.. code-block:: python

   class FrontDoorGenerator(DataGenerator):
       """Generate synthetic data satisfying the front-door criterion"""
       
       def generate_data(self):
           X = self.generate_covariates()
           cols = [f"X{i+1}" for i in range(self.n_covars)]
           df = pd.DataFrame(X, columns=cols)
           
           # Latent confounder U affects both D and Y
           U = np.random.normal(0, 1, self.n_observations)
           
           # Treatment depends on U and X (confounded)
           vec_d = np.random.uniform(0.5, 1.5, size=self.n_covars)
           df['D'] = (X @ vec_d + 0.8 * U + 
                     np.random.normal(0, 1, self.n_observations)) > 0
           df['D'] = df['D'].astype(int)
           
           # Mediator depends on D and X (front-door path)
           vec_m = np.random.uniform(0.5, 1.5, size=self.n_covars)
           df['M'] = X @ vec_m + df['D'] * 1.5 + np.random.normal(0, 1, self.n_observations)
           
           # Outcome depends on M, U, and X (not directly on D)
           vec_y = np.random.uniform(0.5, 1.5, size=self.n_covars)
           df['Y'] = (50 + 2.0 * df['M'] + 1.0 * U + X @ vec_y + 
                     np.random.normal(0, 1, self.n_observations))
           
           self.data = df
           return df

**Agent Testing Scenarios**:
   - **Mediation Detection**: Agent should identify mediator variables
   - **Front-Door Validity**: Should assess front-door criterion assumptions
   - **Sequential Analysis**: Should perform appropriate two-stage analysis

.. code-block:: python

   class RCTDataGenerator(BaseDataGenerator):
       """Generate data from randomized controlled trials"""
       
       def get_method_name(self) -> str:
           return "randomized_controlled_trial"
       
       def generate_treatment(self, covariates: np.ndarray) -> np.ndarray:
           """Generate randomly assigned treatment"""
           # Pure randomization - independent of covariates
           return np.random.binomial(1, 0.5, self.config.n_observations)
       
       def generate_outcome(
           self, 
           treatment: np.ndarray, 
           covariates: np.ndarray
       ) -> np.ndarray:
           """Generate outcome with treatment effect"""
           # Base outcome from covariates
           base_outcome = (
               2.0 +  # Intercept
               0.5 * covariates[:, 0] +  # Effect of X1
               0.3 * covariates[:, 1] +  # Effect of X2
               -0.2 * covariates[:, 2]   # Effect of X3
           )
           
           # Add treatment effect
           if self.config.heterogeneity:
               # Heterogeneous treatment effects
               treatment_effect = (
                   self.config.true_effect * 
                   (1 + 0.5 * covariates[:, 0])  # Effect varies with X1
               )
           else:
               # Homogeneous treatment effect
               treatment_effect = self.config.true_effect
           
           outcome = base_outcome + treatment_effect * treatment
           
           # Add noise
           return self.add_noise(outcome)

Difference-in-Differences Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate panel data suitable for DiD analysis:

.. code-block:: python

   class DifferenceInDifferencesGenerator(BaseDataGenerator):
       """Generate panel data for Difference-in-Differences analysis"""
       
       def __init__(self, config: DataGenerationConfig, n_periods: int = 4, n_units: int = 50):
           super().__init__(config)
           self.n_periods = n_periods
           self.n_units = n_units
           self.config.n_observations = n_units * n_periods
       
       def get_method_name(self) -> str:
           return "difference_in_differences"
       
       def generate_data(self) -> pd.DataFrame:
           """Generate panel data with treatment timing variation"""
           data_list = []
           
           # Generate unit-specific effects
           unit_effects = np.random.normal(0, 1, self.n_units)
           
           # Generate time effects
           time_effects = np.random.normal(0, 0.5, self.n_periods)
           
           # Determine treatment timing (some units treated in period 3)
           treatment_units = np.random.choice(
               self.n_units, 
               size=self.n_units // 2, 
               replace=False
           )
           treatment_start_period = 2  # Treatment starts in period 3 (0-indexed)
           
           for unit in range(self.n_units):
               for period in range(self.n_periods):
                   # Generate covariates (time-varying)
                   covariates = np.random.multivariate_normal(
                       self.covariate_means, 
                       self.covariate_cov
                   )
                   
                   # Treatment assignment
                   is_treated_unit = unit in treatment_units
                   is_post_treatment = period >= treatment_start_period
                   treatment = 1 if (is_treated_unit and is_post_treatment) else 0
                   
                   # Outcome generation
                   outcome = (
                       unit_effects[unit] +  # Unit fixed effect
                       time_effects[period] +  # Time fixed effect
                       0.5 * covariates[0] +  # Covariate effects
                       0.3 * covariates[1] +
                       self.config.true_effect * treatment +  # Treatment effect
                       np.random.normal(0, self.config.noise_level)  # Noise
                   )
                   
                   # Create row
                   row = {
                       'unit_id': unit,
                       'time_period': period,
                       'treatment': treatment,
                       'outcome': outcome,
                       'treated_unit': int(is_treated_unit),
                       'post_treatment': int(is_post_treatment)
                   }
                   
                   # Add covariates
                   for i, covar in enumerate(covariates):
                       row[f'X{i+1}'] = covar
                   
                   data_list.append(row)
           
           self.data = pd.DataFrame(data_list)
           
           # Update metadata
           self.metadata.update({
               'n_units': self.n_units,
               'n_periods': self.n_periods,
               'treatment_start_period': treatment_start_period,
               'n_treated_units': len(treatment_units)
           })
           
           return self.data

Instrumental Variables Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate data with instrumental variables:

.. code-block:: python

   class InstrumentalVariableGenerator(BaseDataGenerator):
       """Generate data with instrumental variables for endogeneity"""
       
       def __init__(self, config: DataGenerationConfig, instrument_strength: float = 0.5):
           super().__init__(config)
           self.instrument_strength = instrument_strength
       
       def get_method_name(self) -> str:
           return "instrumental_variable"
       
       def generate_data(self) -> pd.DataFrame:
           """Generate data with endogenous treatment and valid instrument"""
           # Generate covariates
           covariates = self.generate_covariates()
           
           # Generate unobserved confounder
           unobserved_confounder = np.random.normal(0, 1, self.config.n_observations)
           
           # Generate instrument (exogenous)
           instrument = np.random.normal(0, 1, self.config.n_observations)
           
           # Generate endogenous treatment
           # Treatment depends on instrument, covariates, and unobserved confounder
           treatment_propensity = (
               self.instrument_strength * instrument +  # Instrument effect
               0.3 * covariates[:, 0] +  # Covariate effects
               0.2 * covariates[:, 1] +
               0.4 * unobserved_confounder  # Endogeneity source
           )
           
           treatment_prob = 1 / (1 + np.exp(-treatment_propensity))
           treatment = np.random.binomial(1, treatment_prob)
           
           # Generate outcome
           # Outcome depends on treatment, covariates, and unobserved confounder
           outcome = (
               2.0 +  # Intercept
               self.config.true_effect * treatment +  # Treatment effect
               0.5 * covariates[:, 0] +  # Covariate effects
               0.3 * covariates[:, 1] +
               -0.2 * covariates[:, 2] +
               0.6 * unobserved_confounder +  # Confounding
               np.random.normal(0, self.config.noise_level)  # Noise
           )
           
           # Create DataFrame
           data = pd.DataFrame({
               'treatment': treatment,
               'outcome': outcome,
               'instrument': instrument,
               'unobserved_confounder': unobserved_confounder  # For validation only
           })
           
           # Add covariates
           for i in range(self.config.n_continuous_covars):
               data[f'X{i+1}'] = covariates[:, i]
           
           # Store additional parameters
           self.true_parameters.update({
               'instrument_strength': self.instrument_strength,
               'instrument_variable': 'instrument',
               'first_stage_f_stat': self._calculate_first_stage_f_stat(instrument, treatment)
           })
           
           self.data = data
           return data
       
       def _calculate_first_stage_f_stat(self, instrument: np.ndarray, treatment: np.ndarray) -> float:
           """Calculate first-stage F-statistic for instrument strength"""
           from sklearn.linear_model import LinearRegression
           from scipy import stats
           
           # First stage regression: treatment ~ instrument
           X = instrument.reshape(-1, 1)
           reg = LinearRegression().fit(X, treatment)
           
           # Calculate F-statistic
           predictions = reg.predict(X)
           residuals = treatment - predictions
           
           mse = np.mean(residuals**2)
           coefficient = reg.coef_[0]
           se = np.sqrt(mse / np.sum((instrument - np.mean(instrument))**2))
           
           f_stat = (coefficient / se)**2
           return f_stat

Regression Discontinuity Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate data with discontinuous treatment assignment:

.. code-block:: python

   class RegressionDiscontinuityGenerator(BaseDataGenerator):
       """Generate data for Regression Discontinuity Design"""
       
       def __init__(self, config: DataGenerationConfig, cutoff: float = 0.0, bandwidth: float = 2.0):
           super().__init__(config)
           self.cutoff = cutoff
           self.bandwidth = bandwidth
       
       def get_method_name(self) -> str:
           return "regression_discontinuity"
       
       def generate_data(self) -> pd.DataFrame:
           """Generate data with discontinuous treatment assignment"""
           # Generate running variable (forcing variable)
           running_variable = np.random.uniform(
               self.cutoff - self.bandwidth, 
               self.cutoff + self.bandwidth, 
               self.config.n_observations
           )
           
           # Generate covariates
           covariates = self.generate_covariates()
           
           # Treatment assignment based on cutoff
           treatment = (running_variable >= self.cutoff).astype(int)
           
           # Generate outcome with discontinuity at cutoff
           # Smooth function of running variable
           smooth_outcome = (
               2.0 +  # Intercept
               0.5 * running_variable +  # Smooth trend
               -0.1 * running_variable**2 +  # Quadratic trend
               0.3 * covariates[:, 0] +  # Covariate effects
               0.2 * covariates[:, 1]
           )
           
           # Add treatment effect (discontinuity)
           outcome = smooth_outcome + self.config.true_effect * treatment
           
           # Add noise
           outcome = self.add_noise(outcome)
           
           # Create DataFrame
           data = pd.DataFrame({
               'treatment': treatment,
               'outcome': outcome,
               'running_variable': running_variable
           })
           
           # Add covariates
           for i in range(self.config.n_continuous_covars):
               data[f'X{i+1}'] = covariates[:, i]
           
           # Store additional parameters
           self.true_parameters.update({
               'cutoff': self.cutoff,
               'bandwidth': self.bandwidth,
               'running_variable': 'running_variable'
           })
           
           self.data = data
           return data

Propensity Score Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate observational data suitable for propensity score methods:

.. code-block:: python

   class PropensityScoreGenerator(BaseDataGenerator):
       """Generate observational data for propensity score methods"""
       
       def __init__(self, config: DataGenerationConfig, selection_strength: float = 1.0):
           super().__init__(config)
           self.selection_strength = selection_strength
       
       def get_method_name(self) -> str:
           return "propensity_score_matching"
       
       def generate_treatment(self, covariates: np.ndarray) -> np.ndarray:
           """Generate treatment with selection on observables"""
           # Treatment propensity depends on covariates
           propensity_logit = (
               -0.5 +  # Intercept (affects overall treatment rate)
               self.selection_strength * 0.8 * covariates[:, 0] +  # Strong selection
               self.selection_strength * 0.6 * covariates[:, 1] +  # Moderate selection
               self.selection_strength * 0.4 * covariates[:, 2]    # Weak selection
           )
           
           propensity_prob = 1 / (1 + np.exp(-propensity_logit))
           treatment = np.random.binomial(1, propensity_prob)
           
           # Store true propensity scores for validation
           self.true_parameters['true_propensity_scores'] = propensity_prob
           
           return treatment
       
       def generate_outcome(
           self, 
           treatment: np.ndarray, 
           covariates: np.ndarray
       ) -> np.ndarray:
           """Generate outcome with confounding"""
           # Base outcome depends on same covariates that affect treatment
           base_outcome = (
               3.0 +  # Intercept
               0.7 * covariates[:, 0] +  # Confounding variable
               0.5 * covariates[:, 1] +  # Confounding variable
               0.3 * covariates[:, 2] +  # Confounding variable
               -0.2 * covariates[:, 0] * covariates[:, 1]  # Interaction
           )
           
           # Add treatment effect
           if self.config.heterogeneity:
               # Heterogeneous effects based on covariates
               treatment_effect = (
                   self.config.true_effect * 
                   (1 + 0.3 * covariates[:, 0])
               )
           else:
               treatment_effect = self.config.true_effect
           
           outcome = base_outcome + treatment_effect * treatment
           
           return self.add_noise(outcome)

Data Generation Workflow and Scripts
------------------------------------

The synthetic data generation system includes a comprehensive workflow for creating, contextualizing, and validating synthetic datasets. This section documents the complete process from configuration to final dataset preparation.

Generation Pipeline Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data generation process follows a structured pipeline:

.. mermaid::

   graph LR
       subgraph "Configuration"
           CONFIG[settings.sh]
           PARAMS[Parameter Setup]
       end
       
       subgraph "Data Generation"
           SCRIPTS[Generation Scripts]
           GENERATORS[Method Generators]
           DATA[Raw Datasets]
       end
       
       subgraph "Context Creation"
           LLM[LLM Context Generation]
           LABELS[Variable Labels]
           STORIES[Background Stories]
           QUERIES[Causal Queries]
       end
       
       subgraph "Finalization"
           RENAME[Column Renaming]
           METADATA[Metadata Creation]
           VALIDATION[Ground Truth Files]
       end
       
       CONFIG --> PARAMS
       PARAMS --> SCRIPTS
       SCRIPTS --> GENERATORS
       GENERATORS --> DATA
       DATA --> LLM
       LLM --> LABELS
       LLM --> STORIES
       LLM --> QUERIES
       LABELS --> RENAME
       STORIES --> METADATA
       QUERIES --> VALIDATION

Step 1: Configuration and Parameter Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generation process begins with configuration in ``data_generation/settings.sh``:

.. code-block:: bash

   # Base directory for all synthetic data
   export BASE_FOLDER="data_generation/samples/synthetic"
   
   # Dataset sizes for each method
   export RCT_SIZE=10
   export MULTI_RCT_SIZE=5
   export CANONICAL_DID_SIZE=5
   export TWFE_DID_SIZE=5
   export OBSERVATIONAL_SIZE=5
   export IV_SIZE=5
   export ENCOURAGEMENT_SIZE=5
   export RDD_SIZE=5
   
   # Observation count ranges
   export MIN_OBS=300
   export MAX_OBS=500
   export DEFAULT_OBS=1000
   
   # Special parameters for TWFE (smaller for computational efficiency)
   export DEFAULT_OBS_TWFE=100
   export MIN_OBS_TWFE=50
   export MAX_OBS_TWFE=100
   
   # Covariate specifications
   export N_CONTINUOUS=5        # Maximum continuous covariates
   export N_BINARY=4           # Maximum binary covariates
   
   # Method-specific parameters
   export MAX_TREATMENTS=5      # Multi-treatment RCT arms
   export MAX_PERIODS=10        # TWFE time periods
   export CUTOFF=25            # RDD cutoff range

**Configuration Features**:
   - **Scalable Testing**: Easily adjust dataset sizes for different testing needs
   - **Method-Specific Tuning**: Tailored parameters for each causal method
   - **Resource Management**: Smaller datasets for computationally intensive methods
   - **Reproducible Setup**: Consistent parameters across all generation runs

Step 2: Raw Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual method scripts generate raw synthetic datasets:

**Single Method Generation**:

.. code-block:: bash

   # Generate RCT data
   bash data_generation/create_data/create_rct_data.sh
   
   # Generate DiD data
   bash data_generation/create_data/create_did_canonical_data.sh
   
   # Generate IV data
   bash data_generation/create_data/create_iv_data.sh

**Batch Generation**:

.. code-block:: bash

   # Generate all methods at once
   bash data_generation/create_synthetic_data_all.sh

Each generation script follows this pattern:

.. code-block:: bash

   #!/bin/sh
   source data_generation/settings.sh
   
   METHOD="rct"
   METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
   DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"
   
   python main/generate_synthetic.py \
       -md ${METADATA_FOLDER} \
       -d ${DATA_FOLDER} \
       -m ${METHOD} \
       -s ${DEFAULT_SIZE} \
       -mb ${N_BINARY} \
       -mc ${N_CONTINUOUS} \
       -o ${DEFAULT_OBS}

**Output Structure**:

.. code-block:: text

   data_generation/samples/synthetic/
   ├── rct/
   │   ├── data/
   │   │   ├── rct_data_0.csv
   │   │   ├── rct_data_1.csv
   │   │   └── ...
   │   └── metadata/
   │       └── rct.json
   ├── did_canonical/
   │   ├── data/
   │   └── metadata/
   └── ...

Step 3: Context Generation with LLM Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses LLM integration to generate realistic contexts for synthetic datasets, making them suitable for testing the complete CAIS workflow.

**Context Generation Process**:

.. code-block:: bash

   # Generate context for single method
   bash data_generation/create_context/create_context_rct.sh
   
   # Generate contexts for all methods
   bash data_generation/create_context_all.sh

**LLM Prompt Engineering**:

The context generation uses sophisticated prompts to create realistic scenarios:

.. code-block:: python

   def create_prompt(summary, method, domain, history):
       """Creates a prompt for generating realistic dataset contexts"""
       
       method_names = {
           "rct": "Randomized Control Trial",
           "did_canonical": "Canonical Difference in Differences",
           "iv": "Instrumental Variable",
           "rdd": "Regression Discontinuity Design",
           # ... other methods
       }
       
       domain_guides = {
           "education": "Education data often includes student performance, "
                       "school-level features, socioeconomic background...",
           "healthcare": "Healthcare data may include treatments, diagnoses, "
                        "hospital visits, recovery outcomes...",
           "labor": "Labor datasets typically include income, education, "
                   "job type, employment history...",
           "policy": "Policy evaluation data may track program participation, "
                    "regional differences, economic impact..."
       }
       
       prompt = f"""
       You are generating realistic contexts for synthetic datasets.
       
       Dataset: {method_names[method]} study in the {domain} domain.
       
       Dataset Summary: {summary}
       
       Previously Used Contexts (avoid duplication): {history}
       
       Tasks:
       1. Propose a realistic real-world scenario
       2. Assign realistic variable names in snake_case
       3. Provide one-line descriptions for each variable
       4. Write background paragraph about data collection
       5. Create a natural language causal question
       6. Write a 1-2 sentence summary
       
       Return as JSON with keys: variable_labels, description, question, summary, domain
       """
       
       return prompt

**Context Output Example**:

.. code-block:: json

   {
     "variable_labels": {
       "X1": "years_education",
       "X2": "household_income",
       "X3": "urban_residence",
       "D": "job_training_program",
       "Y": "monthly_earnings"
     },
     "description": "This dataset was collected from a randomized evaluation of a job training program conducted by the Department of Labor in 2019-2020. Participants were randomly assigned to receive either intensive job training or standard employment services.",
     "question": "What is the impact of the job training program on participants' monthly earnings?",
     "summary": "Randomized trial data measuring the effect of job training on employment outcomes.",
     "domain": "labor"
   }

Step 4: Data Finalization and Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final step combines raw data with generated contexts to create analysis-ready datasets:

.. code-block:: bash

   # Finalize all synthetic datasets
   bash data_generation/finalize_synthetic_dataset.sh

**Finalization Process**:

1. **Column Renaming**: Replace generic names (X1, X2, D, Y) with realistic variable names
2. **Metadata Integration**: Combine generation parameters with contextual information
3. **Ground Truth Files**: Create files with known causal effects for validation
4. **Analysis-Ready Format**: Prepare datasets for CAIS agent testing

**Final Output Structure**:

.. code-block:: text

   data_generation/samples/synthetic/
   ├── synthetic_data/           # Renamed datasets
   │   ├── rct_data_0.csv
   │   ├── did_canonical_data_0.csv
   │   └── ...
   ├── data_info/               # Ground truth files
   │   ├── rct_info.csv
   │   ├── did_canonical_info.csv
   │   └── ...
   └── [method]/
       ├── data/                # Original datasets
       ├── metadata/            # Generation metadata
       └── description/         # LLM-generated contexts

**Ground Truth File Format**:

.. code-block:: csv

   data_files,natural_language_query,data_description,method,answer,keywords
   rct_data_0.csv,"What is the impact of job training on earnings?","Randomized trial of job training program...","rct","1.23","Causality, Treatment effect"

Logging and Quality Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generation system includes comprehensive logging for quality control and debugging:

**Logging Configuration** (``data_generation/log_config.ini``):

.. code-block:: ini

   [loggers]
   keys=root,observational_data_logger,did_data_logger,iv_data_logger,rct_data_logger
   
   [handlers]
   keys=consoleHandler,obsHandler,didHandler,ivHandler,rctHandler
   
   [formatters]
   keys=simpleFormatter,complexFormatter
   
   [logger_rct_data_logger]
   level=DEBUG
   handlers=consoleHandler,rctHandler
   qualname=rct_data_logger
   propagate=0

**Quality Control Features**:
   - **Generation Validation**: Each generator tests its output against known ground truth
   - **Statistical Verification**: Automated checks of treatment effects and method assumptions
   - **Context Quality**: LLM-generated contexts are validated for realism and consistency
   - **Reproducibility**: All generation steps are logged with parameters and random seeds

Batch Processing and Agent Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system supports batch processing for comprehensive agent testing:

**Agent Testing Script** (``data_generation/run_agent.py``):

.. code-block:: python

   def run_caia(desc, question, df):
       """Run CAIS agent on synthetic dataset"""
       return run_causal_analysis(
           query=question, 
           dataset_path=df, 
           dataset_description=desc
       )
   
   def main():
       """Process multiple datasets and collect results"""
       meta_df = pd.read_csv(args.csv_meta)
       results = {}
       
       for idx, row in meta_df.iterrows():
           data_path = os.path.join(args.data_dir, str(row["data_files"]))
           
           try:
               res = run_caia(
                   desc=row["data_description"],
                   question=row["natural_language_query"],
                   df=data_path,
               )
               
               # Format results for validation
               formatted_result = {
                   "query": row["natural_language_query"],
                   "method": row["method"],
                   "true_answer": row["answer"],
                   "agent_result": res['results']['results'],
                   "explanation": res.get("explanation", ""),
                   "method_selected": res['results']['results'].get("method_used")
               }
               
               results[idx] = formatted_result
               
           except Exception as e:
               results[idx] = {"error": str(e)}
       
       # Save comprehensive results
       with open(args.output_json, "w") as f:
           json.dump(results, f, indent=2)

**Testing Capabilities**:
   - **Method Selection Validation**: Compare agent's method choice with expected method
   - **Effect Estimation Accuracy**: Compare estimated effects with known ground truth
   - **Decision Tree Logic**: Validate decision tree paths for different data types
   - **Error Handling**: Test agent behavior with edge cases and assumption violations

Scenario Generation and Testing
-------------------------------

The synthetic data system supports various testing scenarios to validate different aspects of the CAIS agent.

Assumption Violation Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate data that violates specific method assumptions to test agent robustness:

.. code-block:: python

   **Parallel Trends Violation (DiD)**:

Tests the agent's ability to detect and handle violations of the parallel trends assumption in difference-in-differences analysis.

.. code-block:: python

   def generate_parallel_trends_violation(base_generator, violation_strength=0.5):
       """Generate DiD data with differential pre-trends"""
       data = base_generator.generate_data()
       
       # Add differential time trends for treated units
       treated_units = data['treated_unit'] == 1
       time_trend_violation = (
           violation_strength * 
           data['time_period'] * 
           treated_units.astype(int)
       )
       
       data['outcome'] += time_trend_violation
       return data

**Agent Testing**: Should detect trend violations through pre-treatment trend tests and either warn users or suggest alternative methods.

**Weak Instrument (IV)**:

Tests the agent's handling of weak instruments that violate the relevance assumption.

.. code-block:: python

   def generate_weak_instrument(base_generator, weak_strength=0.1):
       """Generate IV data with weak first-stage relationship"""
       base_generator.instrument_strength = weak_strength
       data = base_generator.generate_data()
       
       # Calculate first-stage F-statistic for validation
       first_stage_f = calculate_first_stage_f_stat(
           data['instrument'], 
           data['treatment']
       )
       
       return data, first_stage_f

**Agent Testing**: Should calculate first-stage F-statistic and warn when F < 10, potentially suggesting alternative methods.

**Unmeasured Confounding (Propensity Score)**:

Tests the agent's behavior when key confounders are unmeasured, violating the unconfoundedness assumption.

.. code-block:: python

   def generate_unmeasured_confounding(base_generator, confounding_strength=0.8):
       """Generate data with unmeasured confounding"""
       data = base_generator.generate_data()
       
       # Add unmeasured confounder affecting both treatment and outcome
       n_obs = len(data)
       unmeasured_confounder = np.random.normal(0, 1, n_obs)
       
       # Retrospectively adjust treatment probabilities
       treatment_adjustment = confounding_strength * unmeasured_confounder
       adjusted_probs = 1 / (1 + np.exp(-treatment_adjustment))
       data['treatment'] = np.random.binomial(1, adjusted_probs)
       
       # Add confounding to outcome
       data['outcome'] += confounding_strength * unmeasured_confounder
       
       return data

**Agent Testing**: Should perform sensitivity analyses and warn about potential unmeasured confounding when balance tests fail.

**Manipulation of Running Variable (RDD)**:

Tests the agent's ability to detect manipulation around the cutoff in regression discontinuity designs.

.. code-block:: python

   def generate_rdd_manipulation(base_generator, manipulation_strength=0.3):
       """Generate RDD data with running variable manipulation"""
       data = base_generator.generate_data()
       
       # Add manipulation near cutoff
       near_cutoff = np.abs(data['running_variable'] - base_generator.cutoff) < 0.5
       manipulation_effect = (
           manipulation_strength * 
           np.random.normal(0, 1, len(data)) * 
           near_cutoff
       )
       
       data['running_variable'] += manipulation_effect
       
       # Recalculate treatment based on manipulated running variable
       data['treatment'] = (data['running_variable'] >= base_generator.cutoff).astype(int)
       
       return data

**Agent Testing**: Should perform McCrary density tests and detect discontinuities in the running variable distribution.

Edge Case and Robustness Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system generates challenging edge cases to test agent robustness:

**Small Sample Sizes**:

.. code-block:: python

   def generate_small_sample_data(method="rct", n_obs=30):
       """Generate small sample data to test statistical power"""
       config = DataGenerationConfig(n_observations=n_obs)
       generator = get_generator_class(method)(config)
       
       data = generator.generate_data()
       
       # Calculate expected statistical power
       effect_size = config.true_effect / config.noise_level
       power = calculate_statistical_power(n_obs, effect_size)
       
       return data, power

**Agent Testing**: Should warn about low statistical power and suggest larger samples or alternative methods.

**High-Dimensional Data**:

.. code-block:: python

   def generate_high_dimensional_data(method="observational", n_covariates=50):
       """Generate data with many covariates to test curse of dimensionality"""
       config = DataGenerationConfig(
           n_continuous_covars=n_covariates,
           n_observations=200  # Relatively small sample
       )
       
       generator = PropensityScoreGenerator(config)
       data = generator.generate_data()
       
       return data

**Agent Testing**: Should detect high-dimensional settings and suggest regularization or dimension reduction.

**Extreme Outliers**:

.. code-block:: python

   def generate_outlier_data(base_generator, outlier_fraction=0.05):
       """Generate data with extreme outliers"""
       data = base_generator.generate_data()
       
       n_outliers = int(outlier_fraction * len(data))
       outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
       
       # Add extreme values to outcome
       outlier_values = np.random.choice([-1, 1], n_outliers) * np.random.uniform(5, 10, n_outliers)
       data.loc[outlier_indices, 'outcome'] += outlier_values
       
       return data

**Agent Testing**: Should detect outliers and suggest robust estimation methods or outlier removal.

**Missing Data Patterns**:

.. code-block:: python

   def generate_missing_data(base_generator, missing_pattern="random", missing_rate=0.15):
       """Generate data with various missing data patterns"""
       data = base_generator.generate_data()
       
       if missing_pattern == "random":
           # Missing completely at random
           for col in data.columns:
               if col not in ['treatment', 'outcome']:
                   n_missing = int(missing_rate * len(data))
                   missing_indices = np.random.choice(len(data), n_missing, replace=False)
                   data.loc[missing_indices, col] = np.nan
       
       elif missing_pattern == "informative":
           # Missing not at random - higher missingness for treated units
           treated_indices = data[data['treatment'] == 1].index
           for col in data.columns:
               if col not in ['treatment', 'outcome']:
                   # Higher missing rate for treated units
                   treated_missing = np.random.choice(
                       treated_indices, 
                       int(missing_rate * 1.5 * len(treated_indices)), 
                       replace=False
                   )
                   data.loc[treated_missing, col] = np.nan
       
       return data

**Agent Testing**: Should detect missing data patterns and suggest appropriate handling methods (imputation, complete case analysis, etc.).

Usage Examples and Best Practices
---------------------------------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of generating and testing synthetic data:

.. code-block:: python

   # Step 1: Configure and generate base data
   from causal_agent.synthetic import RCTGenerator, DataGenerationConfig
   
   config = DataGenerationConfig(
       n_observations=1000,
       n_continuous_covars=3,
       n_binary_covars=2,
       true_effect=1.5,
       noise_level=1.0,
       seed=42
   )
   
   generator = RCTGenerator(config)
   data = generator.generate_data()
   
   # Step 2: Generate realistic context
   from causal_agent.synthetic.prompts import create_prompt, generate_data_summary
   
   summary = generate_data_summary(
       data, 
       n_cont_vars=3, 
       n_bin_vars=2, 
       method="rct"
   )
   
   prompt = create_prompt(summary, "rct", "education", "")
   # Use LLM to generate context (implementation depends on LLM provider)
   context = generate_context_with_llm(prompt)
   
   # Step 3: Rename columns with realistic names
   data_renamed = data.rename(columns=context['variable_labels'])
   
   # Step 4: Test with CAIS agent
   from causal_agent.agent import run_causal_analysis
   
   result = run_causal_analysis(
       query=context['question'],
       dataset_path=data_renamed,
       dataset_description=context['description']
   )
   
   # Step 5: Validate results
   true_effect = config.true_effect
   estimated_effect = result['results']['results']['causal_effect']
   
   print(f"True effect: {true_effect}")
   print(f"Estimated effect: {estimated_effect}")
   print(f"Method selected: {result['results']['results']['method_used']}")
   print(f"Expected method: RCT/Difference-in-means")

Batch Testing Example
~~~~~~~~~~~~~~~~~~~~~

For comprehensive testing across multiple methods and scenarios:

.. code-block:: python

   def run_comprehensive_test_suite():
       """Run comprehensive test suite across all methods and scenarios"""
       
       methods = ['rct', 'did_canonical', 'iv', 'rdd', 'observational']
       scenarios = ['canonical', 'assumption_violation', 'small_sample', 'outliers']
       
       results = {}
       
       for method in methods:
           for scenario in scenarios:
               print(f"Testing {method} with {scenario} scenario...")
               
               # Generate appropriate data
               if scenario == 'canonical':
                   data, true_params = generate_canonical_data(method)
               elif scenario == 'assumption_violation':
                   data, true_params = generate_violation_data(method)
               elif scenario == 'small_sample':
                   data, true_params = generate_small_sample_data(method)
               elif scenario == 'outliers':
                   data, true_params = generate_outlier_data(method)
               
               # Test with agent
               try:
                   result = test_with_agent(data, true_params)
                   results[f"{method}_{scenario}"] = {
                       'success': True,
                       'method_correct': result['method_used'] == true_params['expected_method'],
                       'effect_accuracy': abs(result['effect'] - true_params['true_effect']),
                       'explanation_quality': evaluate_explanation(result['explanation'])
                   }
               except Exception as e:
                   results[f"{method}_{scenario}"] = {
                       'success': False,
                       'error': str(e)
                   }
       
       return results

Best Practices for Synthetic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameter Selection**:
   - Use realistic effect sizes (typically 0.1 to 2.0 standard deviations)
   - Vary sample sizes to test statistical power considerations
   - Include appropriate noise levels to simulate real-world data
   - Use correlated covariates to reflect realistic data structures

**Validation Procedures**:
   - Always test generated data with known statistical methods
   - Verify that true parameters can be recovered under ideal conditions
   - Check that assumption violations produce expected biases
   - Validate that edge cases trigger appropriate agent responses

**Context Generation**:
   - Use domain-specific terminology and scenarios
   - Ensure variable names are realistic and interpretable
   - Create plausible data collection stories
   - Generate natural language questions that avoid statistical jargon

**Testing Integration**:
   - Test complete agent workflow, not just individual methods
   - Validate decision tree logic with appropriate data characteristics
   - Check error handling and edge case responses
   - Ensure explanations are accurate and helpful

**Documentation and Reproducibility**:
   - Document all generation parameters and random seeds
   - Save metadata alongside generated datasets
   - Include ground truth information for validation
   - Maintain version control for generation scripts and parameters

Integration with CAIS Testing Framework
--------------------------------------

The synthetic data generation system is fully integrated with the CAIS testing and validation framework, enabling comprehensive evaluation of the autonomous agent's capabilities.

Continuous Integration Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synthetic data system supports automated testing in CI/CD pipelines:

.. code-block:: yaml

   # .github/workflows/synthetic_data_tests.yml
   name: Synthetic Data Validation
   
   on: [push, pull_request]
   
   jobs:
     test-synthetic-data:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Setup Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.8'
         
         - name: Install dependencies
           run: |
             pip install -r requirements.txt
         
         - name: Generate synthetic datasets
           run: |
             bash data_generation/create_synthetic_data_all.sh
         
         - name: Test agent on synthetic data
           run: |
             python tests/test_synthetic_data_integration.py
         
         - name: Validate decision tree logic
           run: |
             python tests/test_decision_tree_validation.py

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

The system enables systematic performance benchmarking across different data characteristics:

.. code-block:: python

   class SyntheticDataBenchmark:
       """Benchmark CAIS performance on synthetic data"""
       
       def __init__(self):
           self.results = {}
           self.benchmark_configs = self._generate_benchmark_configs()
       
       def _generate_benchmark_configs(self):
           """Generate configurations for systematic benchmarking"""
           configs = []
           
           # Vary sample sizes
           for n_obs in [100, 500, 1000, 5000]:
               # Vary effect sizes
               for effect_size in [0.1, 0.5, 1.0, 2.0]:
                   # Vary noise levels
                   for noise in [0.5, 1.0, 2.0]:
                       configs.append({
                           'n_observations': n_obs,
                           'true_effect': effect_size,
                           'noise_level': noise
                       })
           
           return configs
       
       def run_benchmark_suite(self):
           """Run comprehensive benchmark across all configurations"""
           methods = ['rct', 'did_canonical', 'iv', 'rdd', 'observational']
           
           for method in methods:
               method_results = []
               
               for config in self.benchmark_configs:
                   # Generate data
                   generator = self._get_generator(method, config)
                   data = generator.generate_data()
                   
                   # Test with agent
                   start_time = time.time()
                   result = self._test_with_agent(data, generator.get_true_parameters())
                   execution_time = time.time() - start_time
                   
                   # Record results
                   method_results.append({
                       'config': config,
                       'execution_time': execution_time,
                       'method_correct': result['method_used'] == method,
                       'effect_accuracy': abs(result['effect'] - config['true_effect']),
                       'confidence_interval_coverage': self._check_ci_coverage(result, config),
                       'explanation_quality': self._evaluate_explanation(result['explanation'])
                   })
               
               self.results[method] = method_results
           
           return self.results

Quality Assurance and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system includes comprehensive quality assurance measures:

**Statistical Validation**:

.. code-block:: python

   def validate_synthetic_data_quality(data, true_parameters):
       """Comprehensive validation of synthetic data quality"""
       validation_results = {}
       
       # Check basic statistical properties
       validation_results['sample_size'] = len(data)
       validation_results['missing_data_rate'] = data.isnull().sum().sum() / data.size
       
       # Validate treatment assignment
       if 'treatment' in data.columns:
           treatment_rate = data['treatment'].mean()
           validation_results['treatment_rate'] = treatment_rate
           validation_results['treatment_balance'] = abs(treatment_rate - 0.5) < 0.1
       
       # Validate covariate balance (for observational data)
       if true_parameters.get('method') in ['propensity_score_matching', 'propensity_score_weighting']:
           balance_stats = calculate_covariate_balance(data)
           validation_results['covariate_balance'] = balance_stats
       
       # Validate known relationships
       if 'instrument' in data.columns:
           first_stage_f = calculate_first_stage_f_stat(data['instrument'], data['treatment'])
           validation_results['instrument_strength'] = first_stage_f
           validation_results['weak_instrument'] = first_stage_f < 10
       
       # Validate effect recovery
       estimated_effect = estimate_treatment_effect(data, true_parameters['method'])
       true_effect = true_parameters['true_effect']
       validation_results['effect_bias'] = abs(estimated_effect - true_effect)
       validation_results['effect_recovery_success'] = validation_results['effect_bias'] < 0.2
       
       return validation_results

**Decision Tree Logic Validation**:

.. code-block:: python

   def validate_decision_tree_logic(synthetic_datasets):
       """Validate that agent makes correct method selections"""
       validation_results = {}
       
       for dataset_name, (data, true_params) in synthetic_datasets.items():
           # Run agent analysis
           agent_result = run_causal_analysis(
               query=true_params['query'],
               dataset_path=data,
               dataset_description=true_params['description']
           )
           
           # Check method selection
           expected_method = true_params['expected_method']
           selected_method = agent_result['results']['results']['method_used']
           
           validation_results[dataset_name] = {
               'method_selection_correct': selected_method == expected_method,
               'expected_method': expected_method,
               'selected_method': selected_method,
               'decision_explanation': agent_result.get('explanation', ''),
               'effect_estimate': agent_result['results']['results']['causal_effect'],
               'true_effect': true_params['true_effect']
           }
       
       return validation_results

Future Enhancements and Extensions
----------------------------------

Planned Improvements
~~~~~~~~~~~~~~~~~~~

The synthetic data generation system continues to evolve with planned enhancements:

**Advanced Scenario Generation**:
   - **Mediation Analysis**: More sophisticated front-door and mediation scenarios
   - **Network Effects**: Data with spillover effects and network structures
   - **Time-Varying Treatments**: Complex temporal treatment patterns
   - **Survival Analysis**: Time-to-event outcomes with censoring

**Enhanced Realism**:
   - **Real Data Mimicking**: Generate synthetic data that closely mimics real dataset characteristics
   - **Domain-Specific Generators**: Specialized generators for healthcare, education, economics
   - **Complex Confounding**: More realistic confounding structures based on real-world patterns

**Improved Testing Capabilities**:
   - **Adversarial Testing**: Generate data specifically designed to challenge the agent
   - **Robustness Testing**: Systematic testing of agent behavior under various assumption violations
   - **Scalability Testing**: Large-scale datasets for performance evaluation

Contributing to the Synthetic Data System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Researchers and developers can contribute to the synthetic data system:

**Adding New Generators**:

.. code-block:: python

   class NewMethodGenerator(DataGenerator):
       """Template for adding new method generators"""
       
       def __init__(self, config, method_specific_params):
           super().__init__(config)
           self.method_specific_params = method_specific_params
           self.method = "new_method"
       
       def generate_data(self):
           """Implement method-specific data generation logic"""
           # 1. Generate covariates using base class
           X = self.generate_covariates()
           
           # 2. Generate treatment using method-specific logic
           treatment = self._generate_treatment(X)
           
           # 3. Generate outcome with known causal effect
           outcome = self._generate_outcome(treatment, X)
           
           # 4. Create DataFrame and return
           data = self._create_dataframe(X, treatment, outcome)
           self.data = data
           return data
       
       def test_data(self, print_=False):
           """Implement validation using appropriate statistical method"""
           # Test that true effect can be recovered
           pass

**Testing New Scenarios**:

.. code-block:: python

   def test_new_scenario():
       """Template for testing new scenarios"""
       # 1. Generate data with specific characteristics
       data = generate_scenario_data()
       
       # 2. Define expected agent behavior
       expected_method = "expected_method_name"
       expected_warnings = ["assumption_violation", "low_power"]
       
       # 3. Test with agent
       result = run_causal_analysis(query, data, description)
       
       # 4. Validate results
       assert result['method_used'] == expected_method
       assert all(warning in result['warnings'] for warning in expected_warnings)

**Documentation Standards**:
   - Document all generation parameters and their effects
   - Provide clear examples of when to use each generator
   - Include validation procedures for new methods
   - Explain integration with decision tree logic

Conclusion
----------

The synthetic data generation system is a cornerstone of the CAIS testing and validation framework. It enables:

* **Comprehensive Testing**: Systematic evaluation of agent decision-making across diverse scenarios
* **Method Validation**: Rigorous testing of causal inference methods with known ground truth
* **Decision Tree Validation**: Verification that the agent selects appropriate methods for different data characteristics
* **Robustness Assessment**: Testing agent behavior under assumption violations and edge cases
* **Performance Benchmarking**: Systematic evaluation of computational performance and statistical accuracy

The system's integration with LLM-based context generation creates realistic testing scenarios that closely mirror real-world causal inference challenges, ensuring that CAIS performs reliably across diverse applications and domains.

For researchers and practitioners using CAIS, the synthetic data system provides confidence in the agent's capabilities and helps identify appropriate use cases and limitations. For developers contributing to CAIS, it provides a comprehensive testing framework that ensures new features and methods integrate properly with the existing decision tree logic and maintain high standards of statistical accuracy and reliability.ut_dir / filename
               generator.save_data(str(filepath))
               
               datasets.append({
                   'filepath': str(filepath),
                   'config': config,
                   'true_parameters': generator.get_true_parameters()
               })
           
           return datasets
       
       def generate_comprehensive_suite(self):
           """Generate comprehensive test suite for all methods"""
           methods = [
               'rct', 'difference_in_differences', 'instrumental_variable',
               'regression_discontinuity', 'propensity_score_matching'
           ]
           
           all_datasets = {}
           
           for method in methods:
               print(f"Generating datasets for {method}...")
               datasets = self.generate_method_suite(method)
               all_datasets[method] = datasets
           
           # Save master index
           self._save_dataset_index(all_datasets)
           
           return all_datasets
       
       def _get_generator_class(self, method_name: str):
           """Get generator class for method"""
           generators = {
               'rct': RCTDataGenerator,
               'difference_in_differences': DifferenceInDifferencesGenerator,
               'instrumental_variable': InstrumentalVariableGenerator,
               'regression_discontinuity': RegressionDiscontinuityGenerator,
               'propensity_score_matching': PropensityScoreGenerator
           }
           return generators[method_name]
       
       def _vary_config(self, base_config: DataGenerationConfig, seed: int):
           """Create varied configuration for diversity"""
           config = DataGenerationConfig(
               n_observations=base_config.n_observations + np.random.randint(-200, 200),
               n_continuous_covars=max(2, base_config.n_continuous_covars + np.random.randint(-1, 2)),
               n_binary_covars=max(1, base_config.n_binary_covars + np.random.randint(-1, 2)),
               true_effect=base_config.true_effect + np.random.normal(0, 0.2),
               noise_level=max(0.1, base_config.noise_level + np.random.normal(0, 0.1)),
               seed=base_config.seed + seed,
               heterogeneity=np.random.choice([True, False])
           )
           return config
       
       def _save_dataset_index(self, all_datasets: Dict):
           """Save index of all generated datasets"""
           index_path = self.output_dir / "dataset_index.json"
           
           # Convert to serializable format
           serializable_index = {}
           for method, datasets in all_datasets.items():
               serializable_index[method] = []
               for dataset in datasets:
                   serializable_index[method].append({
                       'filepath': dataset['filepath'],
                       'config': dataset['config'].__dict__,
                       'true_parameters': dataset['true_parameters']
                   })
           
           import json
           with open(index_path, 'w') as f:
               json.dump(serializable_index, f, indent=2)

Data Validation
~~~~~~~~~~~~~~~

Validate generated synthetic data:

.. code-block:: python

   class SyntheticDataValidator:
       """Validate synthetic data quality and properties"""
       
       def __init__(self):
           self.validation_results = {}
       
       def validate_dataset(
           self, 
           data: pd.DataFrame, 
           true_parameters: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Comprehensive validation of synthetic dataset"""
           
           results = {
               'basic_properties': self._validate_basic_properties(data),
               'statistical_properties': self._validate_statistical_properties(data),
               'causal_structure': self._validate_causal_structure(data, true_parameters),
               'method_specific': self._validate_method_specific(data, true_parameters)
           }
           
           results['overall_quality'] = self._assess_overall_quality(results)
           
           return results
       
       def _validate_basic_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
           """Validate basic data properties"""
           return {
               'shape': data.shape,
               'missing_values': data.isnull().sum().to_dict(),
               'data_types': data.dtypes.to_dict(),
               'duplicates': data.duplicated().sum(),
               'treatment_balance': data['treatment'].value_counts().to_dict() if 'treatment' in data.columns else None
           }
       
       def _validate_statistical_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
           """Validate statistical properties"""
           numeric_cols = data.select_dtypes(include=[np.number]).columns
           
           return {
               'means': data[numeric_cols].mean().to_dict(),
               'std_devs': data[numeric_cols].std().to_dict(),
               'correlations': data[numeric_cols].corr().to_dict(),
               'outliers': self._detect_outliers(data[numeric_cols])
           }
       
       def _validate_causal_structure(
           self, 
           data: pd.DataFrame, 
           true_parameters: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Validate causal structure matches intended design"""
           
           # Estimate treatment effect using simple method
           if 'treatment' in data.columns and 'outcome' in data.columns:
               treated = data[data['treatment'] == 1]['outcome']
               control = data[data['treatment'] == 0]['outcome']
               
               estimated_effect = treated.mean() - control.mean()
               true_effect = true_parameters.get('true_effect', 0)
               
               return {
                   'estimated_effect': estimated_effect,
                   'true_effect': true_effect,
                   'effect_bias': abs(estimated_effect - true_effect),
                   'effect_recovery_ratio': estimated_effect / true_effect if true_effect != 0 else None
               }
           
           return {}
       
       def _validate_method_specific(
           self, 
           data: pd.DataFrame, 
           true_parameters: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Method-specific validation"""
           method = true_parameters.get('method', '')
           
           if method == 'instrumental_variable':
               return self._validate_iv_properties(data, true_parameters)
           elif method == 'regression_discontinuity':
               return self._validate_rdd_properties(data, true_parameters)
           elif method == 'difference_in_differences':
               return self._validate_did_properties(data, true_parameters)
           
           return {}
       
       def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
           """Detect outliers using IQR method"""
           outliers = {}
           
           for col in data.columns:
               Q1 = data[col].quantile(0.25)
               Q3 = data[col].quantile(0.75)
               IQR = Q3 - Q1
               
               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR
               
               outliers[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
           
           return outliers
       
       def _assess_overall_quality(self, results: Dict[str, Any]) -> str:
           """Assess overall data quality"""
           issues = []
           
           # Check for basic issues
           if results['basic_properties']['duplicates'] > 0:
               issues.append("duplicates")
           
           if any(v > 0 for v in results['basic_properties']['missing_values'].values()):
               issues.append("missing_values")
           
           # Check causal structure
           if 'effect_bias' in results['causal_structure']:
               if results['causal_structure']['effect_bias'] > 0.5:
                   issues.append("high_effect_bias")
           
           if len(issues) == 0:
               return "excellent"
           elif len(issues) <= 2:
               return "good"
           else:
               return "needs_improvement"

Testing Integration
-------------------

Using Synthetic Data in Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate synthetic data generation with the testing framework:

.. code-block:: python

   # tests/fixtures/synthetic_data.py
   
   import pytest
   from causal_agent.synthetic.generator import *
   
   @pytest.fixture
   def rct_data():
       """Generate RCT data for testing"""
       config = DataGenerationConfig(n_observations=500, true_effect=1.5)
       generator = RCTDataGenerator(config)
       return generator.generate_data(), generator.get_true_parameters()
   
   @pytest.fixture
   def did_data():
       """Generate DiD data for testing"""
       config = DataGenerationConfig(n_observations=1000, true_effect=2.0)
       generator = DifferenceInDifferencesGenerator(config, n_periods=4, n_units=50)
       return generator.generate_data(), generator.get_true_parameters()
   
   @pytest.fixture
   def iv_data():
       """Generate IV data for testing"""
       config = DataGenerationConfig(n_observations=800, true_effect=1.2)
       generator = InstrumentalVariableGenerator(config, instrument_strength=0.6)
       return generator.generate_data(), generator.get_true_parameters()
   
   # Example test using synthetic data
   def test_method_with_synthetic_data(rct_data):
       """Test causal method with synthetic RCT data"""
       data, true_params = rct_data
       
       # Run method
       from causal_agent.methods.experimental.diff_in_means.estimator import estimate_diff_in_means
       
       variables = Variables(
           treatment_variable='treatment',
           outcome_variable='outcome',
           covariates=[col for col in data.columns if col.startswith('X')],
           is_rct=True
       )
       
       results = estimate_diff_in_means(data, variables)
       
       # Validate against true parameters
       true_effect = true_params['true_effect']
       estimated_effect = results['effect_estimate']
       
       # Allow for sampling variation
       assert abs(estimated_effect - true_effect) < 0.5
       assert results['p_value'] < 0.05  # Should be significant

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

Integrate synthetic data testing into CI/CD:

.. code-block:: yaml

   # .github/workflows/synthetic_data_tests.yml
   
   name: Synthetic Data Tests
   
   on: [push, pull_request]
   
   jobs:
     synthetic-data-tests:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v2
       
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: 3.9
       
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
           pip install -e .
       
       - name: Generate synthetic test data
         run: |
           python -c "
           from causal_agent.synthetic.generator import BatchDataGenerator
           generator = BatchDataGenerator('test_synthetic_data')
           generator.generate_comprehensive_suite()
           "
       
       - name: Run synthetic data validation
         run: |
           pytest tests/synthetic/ -v --cov=causal_agent.synthetic
       
       - name: Run method tests with synthetic data
         run: |
           pytest tests/unit/methods/ -v -k "synthetic"

Best Practices
--------------

Data Generation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Realistic Parameters**: Use parameter values that reflect real-world scenarios
* **Known Ground Truth**: Always maintain known causal relationships for validation
* **Diverse Scenarios**: Generate data covering various conditions and edge cases
* **Reproducibility**: Use fixed seeds for reproducible test datasets
* **Documentation**: Clearly document the causal structure and assumptions

Validation Standards
~~~~~~~~~~~~~~~~~~~~

* **Effect Recovery**: Validate that methods recover true effects within reasonable bounds
* **Assumption Testing**: Generate data that both satisfies and violates method assumptions
* **Statistical Properties**: Ensure generated data has realistic statistical properties
* **Edge Case Coverage**: Test with small samples, outliers, and missing data
* **Performance Benchmarking**: Use large datasets to test scalability

Testing Integration
~~~~~~~~~~~~~~~~~~~

* **Automated Generation**: Integrate data generation into CI/CD pipelines
* **Comprehensive Coverage**: Test all methods with appropriate synthetic data
* **Performance Monitoring**: Track method performance across different data scenarios
* **Regression Testing**: Use synthetic data to detect performance regressions
* **Documentation Examples**: Use synthetic data for clear, reproducible examples

The synthetic data generation system provides a robust foundation for testing, validating, and benchmarking causal inference methods in CAIS, ensuring reliability and accuracy across diverse real-world scenarios.
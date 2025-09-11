"""Integration tests for the test fixtures and data management system."""

import pytest
import pandas as pd
import numpy as np
from tests.fixtures import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    DatasetType,
    MockLLMResponseGenerator,
    LLMResponseType,
    SharedDatasetManager,
    CausalAgentTestConfig,
    get_test_config,
    get_synthetic_data_generator,
    get_mock_llm_generator,
    get_dataset_manager
)


class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""
    
    def test_rct_data_generation(self):
        """Test RCT data generation."""
        generator = get_synthetic_data_generator()
        data = generator.generate_rct_data()
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 500  # Default config
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        
        # Check treatment randomization (should be roughly balanced)
        treatment_balance = data['treatment'].mean()
        assert 0.4 <= treatment_balance <= 0.6
        
        # Check metadata
        assert data.attrs['dataset_type'] == DatasetType.RCT.value
        assert data.attrs['true_treatment_effect'] == 0.5
    
    def test_observational_data_generation(self):
        """Test observational data generation."""
        generator = get_synthetic_data_generator()
        data = generator.generate_observational_data()
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        
        # Check for confounders
        confounder_cols = [col for col in data.columns if 'confounder' in col]
        assert len(confounder_cols) > 0
        
        # Check metadata
        assert data.attrs['dataset_type'] == DatasetType.OBSERVATIONAL.value
        assert len(data.attrs['confounders']) > 0
    
    def test_iv_data_generation(self):
        """Test instrumental variable data generation."""
        generator = get_synthetic_data_generator()
        data = generator.generate_iv_data()
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert 'instrument' in data.columns
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        
        # Check metadata
        assert data.attrs['dataset_type'] == DatasetType.INSTRUMENTAL_VARIABLE.value
        assert data.attrs['instrument'] == 'instrument'
    
    def test_rdd_data_generation(self):
        """Test regression discontinuity data generation."""
        generator = get_synthetic_data_generator()
        data = generator.generate_rdd_data()
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert 'running_var' in data.columns
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        
        # Check discontinuity
        cutoff = data.attrs['cutoff']
        treated = data[data['running_var'] >= cutoff]['treatment']
        untreated = data[data['running_var'] < cutoff]['treatment']
        
        assert treated.all()  # All above cutoff should be treated
        assert not untreated.any()  # All below cutoff should be untreated
    
    def test_did_data_generation(self):
        """Test difference-in-differences data generation."""
        generator = get_synthetic_data_generator()
        data = generator.generate_did_data()
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert 'unit' in data.columns
        assert 'period' in data.columns
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        
        # Check panel structure
        n_units = data['unit'].nunique()
        n_periods = data['period'].nunique()
        assert len(data) == n_units * n_periods
    
    def test_custom_config(self):
        """Test data generation with custom configuration."""
        config = SyntheticDataConfig(
            n_samples=100,
            treatment_effect=0.8,
            noise_level=0.05,
            random_seed=123
        )
        generator = SyntheticDataGenerator(config)
        data = generator.generate_rct_data()
        
        assert len(data) == 100
        assert data.attrs['true_treatment_effect'] == 0.8


class TestMockLLMResponses:
    """Test mock LLM response functionality."""
    
    def test_method_selection_response(self):
        """Test method selection mock responses."""
        generator = get_mock_llm_generator()
        response = generator.get_response(LLMResponseType.METHOD_SELECTION)
        
        assert isinstance(response.content, dict)
        assert 'recommended_method' in response.content
        assert 'confidence' in response.content
        assert 'reasoning' in response.content
        assert 'assumptions' in response.content
    
    def test_dataset_analysis_response(self):
        """Test dataset analysis mock responses."""
        generator = get_mock_llm_generator()
        response = generator.get_response(LLMResponseType.DATASET_ANALYSIS)
        
        assert isinstance(response.content, dict)
        assert 'dataset_summary' in response.content
        assert 'data_quality' in response.content
        assert 'variable_analysis' in response.content
    
    def test_custom_method_response(self):
        """Test custom method selection responses."""
        generator = get_mock_llm_generator()
        response_content = generator.get_method_selection_response(
            method="propensity_score",
            confidence=0.9
        )
        
        assert response_content['recommended_method'] == "propensity_score"
        assert response_content['confidence'] == 0.9
        assert 'assumptions' in response_content
    
    def test_result_interpretation_response(self):
        """Test result interpretation mock responses."""
        generator = get_mock_llm_generator()
        response = generator.get_response(LLMResponseType.RESULT_INTERPRETATION)
        
        assert isinstance(response.content, dict)
        assert 'effect_interpretation' in response.content
        assert 'confidence_assessment' in response.content
        assert 'interpretation_text' in response.content


class TestSharedDatasetManager:
    """Test shared dataset management functionality."""
    
    def test_dataset_registration(self, tmp_path):
        """Test dataset registration and loading."""
        # Create temporary dataset manager
        manager = SharedDatasetManager(tmp_path)
        
        # Create test dataset
        generator = get_synthetic_data_generator()
        data = generator.generate_rct_data()
        
        # Create metadata
        from tests.fixtures.shared_datasets import DatasetMetadata
        metadata = DatasetMetadata(
            name="test_rct",
            dataset_type="synthetic",
            n_samples=len(data),
            n_features=5,
            true_treatment_effect=0.5,
            description="Test RCT dataset",
            tags=["test", "rct"]
        )
        
        # Register dataset
        dataset_path = manager.register_dataset("test_rct", data, metadata)
        assert dataset_path.exists()
        
        # Load dataset
        loaded_data, loaded_metadata = manager.load_dataset("test_rct")
        assert len(loaded_data) == len(data)
        assert loaded_metadata.name == "test_rct"
        assert loaded_metadata.true_treatment_effect == 0.5
    
    def test_dataset_listing(self, tmp_path):
        """Test dataset listing functionality."""
        manager = SharedDatasetManager(tmp_path)
        
        # Should start empty
        datasets = manager.list_datasets()
        assert len(datasets) == 0
        
        # Add some datasets
        generator = get_synthetic_data_generator()
        
        for i, dataset_type in enumerate(["rct", "observational"]):
            data = generator.generate_rct_data() if dataset_type == "rct" else generator.generate_observational_data()
            
            from tests.fixtures.shared_datasets import DatasetMetadata
            metadata = DatasetMetadata(
                name=f"test_{dataset_type}",
                dataset_type="synthetic",
                n_samples=len(data),
                n_features=5,
                true_treatment_effect=0.5,
                description=f"Test {dataset_type} dataset",
                tags=["test", dataset_type]
            )
            
            manager.register_dataset(f"test_{dataset_type}", data, metadata)
        
        # List all datasets
        all_datasets = manager.list_datasets()
        assert len(all_datasets) == 2
        
        # List by tag
        rct_datasets = manager.list_datasets(tags=["rct"])
        assert len(rct_datasets) == 1
        assert "test_rct" in rct_datasets


class TestConfigurationManagement:
    """Test configuration management functionality."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = get_test_config()
        
        assert isinstance(config, CausalAgentTestConfig)
        assert config.random_seed == 42
        assert config.llm.mock_llm is True
        assert config.data.use_synthetic_data is True
    
    def test_config_environment_detection(self):
        """Test environment-specific configuration."""
        from tests.fixtures import get_config_manager
        manager = get_config_manager()
        
        # Test CI detection
        is_ci = manager.is_ci_environment()
        assert isinstance(is_ci, bool)
        
        # Test LLM usage decision
        should_use_real = manager.should_use_real_llm()
        assert isinstance(should_use_real, bool)
    
    def test_method_specific_config(self):
        """Test method-specific configuration."""
        from tests.fixtures import create_test_config_for_method
        
        config = create_test_config_for_method("propensity_score")
        assert "propensity_score" in config.method_configs
        
        ps_config = config.method_configs["propensity_score"]
        assert "max_execution_time" in ps_config
        assert "required_sample_size" in ps_config


class TestFixtureIntegration:
    """Test integration between different fixture components."""
    
    def test_end_to_end_fixture_usage(self):
        """Test using multiple fixtures together."""
        # Get components
        generator = get_synthetic_data_generator()
        llm_generator = get_mock_llm_generator()
        config = get_test_config()
        
        # Generate data
        data = generator.generate_observational_data()
        
        # Get mock LLM response for method selection
        method_response = llm_generator.get_method_selection_response(
            method="backdoor_adjustment"
        )
        
        # Verify integration
        assert len(data) > 0
        assert method_response['recommended_method'] == "backdoor_adjustment"
        assert config.data.use_synthetic_data is True
    
    def test_benchmark_dataset_creation(self):
        """Test benchmark dataset creation."""
        from tests.fixtures import get_benchmark_datasets
        
        datasets = get_benchmark_datasets()
        
        # Should have multiple datasets
        assert len(datasets) > 5
        
        # Check dataset types
        dataset_types = set()
        for name, data in datasets.items():
            if hasattr(data, 'attrs') and 'dataset_type' in data.attrs:
                dataset_types.add(data.attrs['dataset_type'])
        
        # Should have multiple types
        assert len(dataset_types) >= 3
    
    def test_performance_dataset_scaling(self):
        """Test performance datasets with different sizes."""
        from tests.fixtures import get_performance_datasets
        
        datasets = get_performance_datasets()
        
        # Should have datasets of different sizes
        sizes = []
        for name, data in datasets.items():
            sizes.append(len(data))
        
        # Should have increasing sizes
        sizes.sort()
        assert len(set(sizes)) > 1  # Multiple different sizes
        assert max(sizes) > min(sizes) * 5  # Significant size variation
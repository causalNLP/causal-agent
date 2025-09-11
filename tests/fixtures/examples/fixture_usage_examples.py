"""Examples demonstrating how to use the test fixtures and data management system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tests.fixtures import (
    get_synthetic_data_generator,
    get_mock_llm_generator,
    get_dataset_manager,
    get_test_config,
    SyntheticDataConfig,
    DatasetType,
    LLMResponseType
)


def example_synthetic_data_generation():
    """Example: Generate synthetic datasets for testing."""
    print("=== Synthetic Data Generation Example ===")
    
    # Get data generator with default config
    generator = get_synthetic_data_generator()
    
    # Generate different types of datasets
    rct_data = generator.generate_rct_data()
    obs_data = generator.generate_observational_data()
    iv_data = generator.generate_iv_data()
    
    print(f"RCT data shape: {rct_data.shape}")
    print(f"RCT treatment effect: {rct_data.attrs['true_treatment_effect']}")
    print(f"Observational data confounders: {obs_data.attrs['confounders']}")
    print(f"IV data instrument: {iv_data.attrs['instrument']}")
    
    # Generate with custom configuration
    custom_config = SyntheticDataConfig(
        n_samples=1000,
        treatment_effect=0.8,
        noise_level=0.05
    )
    generator.config = custom_config
    large_dataset = generator.generate_rct_data()
    
    print(f"Large dataset shape: {large_dataset.shape}")
    print(f"Large dataset treatment effect: {large_dataset.attrs['true_treatment_effect']}")


def example_mock_llm_responses():
    """Example: Use mock LLM responses for testing."""
    print("\n=== Mock LLM Responses Example ===")
    
    # Get LLM response generator
    llm_generator = get_mock_llm_generator()
    
    # Get method selection response
    method_response = llm_generator.get_method_selection_response(
        method="propensity_score",
        confidence=0.85
    )
    
    print(f"Recommended method: {method_response['recommended_method']}")
    print(f"Confidence: {method_response['confidence']}")
    print(f"Assumptions: {method_response['assumptions']}")
    
    # Get dataset analysis response
    dataset_response = llm_generator.get_dataset_analysis_response(
        n_obs=500,
        data_quality=0.9
    )
    
    print(f"Dataset quality score: {dataset_response['data_quality']['quality_score']}")
    print(f"Potential confounders: {dataset_response['variable_analysis']['potential_confounders']}")
    
    # Get structured response object
    structured_response = llm_generator.get_response(LLMResponseType.RESULT_INTERPRETATION)
    print(f"Effect interpretation: {structured_response.content['effect_interpretation']}")


def example_dataset_management():
    """Example: Manage shared datasets for testing."""
    print("\n=== Dataset Management Example ===")
    
    # Get dataset manager
    manager = get_dataset_manager()
    
    # Create benchmark datasets
    benchmark_datasets = manager.create_benchmark_suite()
    print(f"Created {len(benchmark_datasets)} benchmark datasets")
    
    # List available datasets
    available_datasets = manager.list_datasets()
    print(f"Available datasets: {available_datasets[:5]}...")  # Show first 5
    
    # Load a specific dataset
    if available_datasets:
        dataset_name = available_datasets[0]
        data, metadata = manager.load_dataset(dataset_name)
        print(f"Loaded {dataset_name}: {data.shape}, effect={metadata.true_treatment_effect}")
    
    # Get validation datasets
    validation_datasets = manager.get_validation_datasets()
    print(f"Validation datasets: {list(validation_datasets.keys())}")


def example_test_configuration():
    """Example: Use test configuration management."""
    print("\n=== Test Configuration Example ===")
    
    # Get default test configuration
    config = get_test_config()
    
    print(f"Environment: {config.environment.value}")
    print(f"Mock LLM: {config.llm.mock_llm}")
    print(f"Random seed: {config.random_seed}")
    print(f"Coverage threshold: {config.coverage.minimum_coverage_percentage}%")
    
    # Get method-specific configuration
    from tests.fixtures import create_test_config_for_method
    ps_config = create_test_config_for_method("propensity_score")
    
    if "propensity_score" in ps_config.method_configs:
        ps_settings = ps_config.method_configs["propensity_score"]
        print(f"Propensity score max execution time: {ps_settings.get('max_execution_time', 'N/A')}")


def example_integrated_workflow():
    """Example: Complete testing workflow using all fixtures."""
    print("\n=== Integrated Workflow Example ===")
    
    # 1. Setup configuration
    config = get_test_config()
    print(f"Using configuration: {config.environment.value} environment")
    
    # 2. Generate test data
    generator = get_synthetic_data_generator()
    test_data = generator.generate_observational_data()
    print(f"Generated test data: {test_data.shape}")
    
    # 3. Mock LLM analysis
    llm_generator = get_mock_llm_generator()
    analysis_response = llm_generator.get_dataset_analysis_response(
        n_obs=len(test_data),
        data_quality=0.85
    )
    print(f"LLM analysis - Quality score: {analysis_response['data_quality']['quality_score']}")
    
    # 4. Get method recommendation
    method_response = llm_generator.get_method_selection_response(
        method="backdoor_adjustment",
        confidence=0.8
    )
    print(f"Recommended method: {method_response['recommended_method']}")
    
    # 5. Store dataset for reuse
    manager = get_dataset_manager()
    from tests.fixtures.shared_datasets import DatasetMetadata
    
    metadata = DatasetMetadata(
        name="example_workflow_data",
        dataset_type="synthetic",
        n_samples=len(test_data),
        n_features=len([col for col in test_data.columns if col not in ['treatment', 'outcome']]),
        true_treatment_effect=test_data.attrs['true_treatment_effect'],
        description="Example workflow dataset",
        tags=["example", "workflow", "observational"]
    )
    
    try:
        dataset_path = manager.register_dataset("example_workflow_data", test_data, metadata)
        print(f"Dataset stored at: {dataset_path}")
    except ValueError:
        print("Dataset already exists (this is expected in repeated runs)")
    
    print("Workflow completed successfully!")


def example_performance_testing():
    """Example: Performance testing with different dataset sizes."""
    print("\n=== Performance Testing Example ===")
    
    # Get performance datasets
    from tests.fixtures import get_performance_datasets
    perf_datasets = get_performance_datasets()
    
    print("Performance datasets available:")
    for name, data in perf_datasets.items():
        print(f"  {name}: {data.shape}")
    
    # Simulate performance testing
    import time
    
    generator = get_synthetic_data_generator()
    
    sizes = [100, 500, 1000]
    execution_times = []
    
    for size in sizes:
        # Configure for specific size
        generator.config.n_samples = size
        
        # Time data generation
        start_time = time.time()
        data = generator.generate_observational_data()
        end_time = time.time()
        
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        print(f"Size {size}: {execution_time:.4f}s")
    
    print(f"Performance scaling: {execution_times}")


if __name__ == "__main__":
    """Run all examples."""
    print("Causal Agent Test Fixtures - Usage Examples")
    print("=" * 50)
    
    try:
        example_synthetic_data_generation()
        example_mock_llm_responses()
        example_dataset_management()
        example_test_configuration()
        example_integrated_workflow()
        example_performance_testing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
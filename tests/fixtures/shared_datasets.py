"""Shared dataset fixtures for method validation across causal_agent tests."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pickle
import json
from dataclasses import dataclass, asdict
from .synthetic_data import SyntheticDataGenerator, SyntheticDataConfig, DatasetType


@dataclass
class DatasetMetadata:
    """Metadata for shared datasets."""
    name: str
    dataset_type: str
    n_samples: int
    n_features: int
    true_treatment_effect: float
    description: str
    source: str = "synthetic"
    created_date: str = ""
    version: str = "1.0"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SharedDatasetManager:
    """Manager for shared datasets used across tests."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the dataset manager."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Subdirectories for different types of data
        self.synthetic_dir = self.data_dir / "synthetic"
        self.real_dir = self.data_dir / "real"
        self.benchmark_dir = self.data_dir / "benchmark"
        
        for dir_path in [self.synthetic_dir, self.real_dir, self.benchmark_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.data_dir / "dataset_metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, DatasetMetadata]:
        """Load dataset metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                return {
                    name: DatasetMetadata(**meta) 
                    for name, meta in metadata_dict.items()
                }
        return {}
    
    def _save_metadata(self):
        """Save dataset metadata to file."""
        metadata_dict = {
            name: asdict(meta) 
            for name, meta in self._metadata.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def register_dataset(self, 
                        name: str, 
                        dataset: pd.DataFrame,
                        metadata: DatasetMetadata,
                        overwrite: bool = False) -> Path:
        """Register a new dataset with metadata."""
        dataset_path = self._get_dataset_path(name, metadata.dataset_type)
        
        if dataset_path.exists() and not overwrite:
            raise ValueError(f"Dataset {name} already exists. Use overwrite=True to replace.")
        
        # Save dataset
        dataset.to_csv(dataset_path, index=False)
        
        # Save metadata
        self._metadata[name] = metadata
        self._save_metadata()
        
        return dataset_path
    
    def _get_dataset_path(self, name: str, dataset_type: str) -> Path:
        """Get the file path for a dataset."""
        if dataset_type == "synthetic":
            return self.synthetic_dir / f"{name}.csv"
        elif dataset_type == "real":
            return self.real_dir / f"{name}.csv"
        elif dataset_type == "benchmark":
            return self.benchmark_dir / f"{name}.csv"
        else:
            return self.data_dir / f"{name}.csv"
    
    def load_dataset(self, name: str) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load a dataset and its metadata."""
        if name not in self._metadata:
            raise ValueError(f"Dataset {name} not found in registry")
        
        metadata = self._metadata[name]
        dataset_path = self._get_dataset_path(name, metadata.dataset_type)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        dataset = pd.read_csv(dataset_path)
        
        # Restore dataset attributes from metadata
        dataset.attrs['dataset_type'] = metadata.dataset_type
        dataset.attrs['true_treatment_effect'] = metadata.true_treatment_effect
        
        return dataset, metadata
    
    def list_datasets(self, dataset_type: Optional[str] = None, 
                     tags: Optional[List[str]] = None) -> List[str]:
        """List available datasets with optional filtering."""
        datasets = []
        
        for name, metadata in self._metadata.items():
            # Filter by dataset type
            if dataset_type and metadata.dataset_type != dataset_type:
                continue
            
            # Filter by tags
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            datasets.append(name)
        
        return sorted(datasets)
    
    def get_metadata(self, name: str) -> DatasetMetadata:
        """Get metadata for a specific dataset."""
        if name not in self._metadata:
            raise ValueError(f"Dataset {name} not found in registry")
        return self._metadata[name]
    
    def delete_dataset(self, name: str):
        """Delete a dataset and its metadata."""
        if name not in self._metadata:
            raise ValueError(f"Dataset {name} not found in registry")
        
        metadata = self._metadata[name]
        dataset_path = self._get_dataset_path(name, metadata.dataset_type)
        
        # Remove file if it exists
        if dataset_path.exists():
            dataset_path.unlink()
        
        # Remove from metadata
        del self._metadata[name]
        self._save_metadata()
    
    def create_benchmark_suite(self) -> Dict[str, pd.DataFrame]:
        """Create a comprehensive benchmark suite for testing."""
        generator = SyntheticDataGenerator()
        benchmark_datasets = {}
        
        # Define benchmark scenarios
        benchmark_configs = [
            # RCT scenarios
            ("rct_small_effect", DatasetType.RCT, 
             SyntheticDataConfig(n_samples=200, treatment_effect=0.1, noise_level=0.2)),
            ("rct_medium_effect", DatasetType.RCT,
             SyntheticDataConfig(n_samples=500, treatment_effect=0.5, noise_level=0.1)),
            ("rct_large_effect", DatasetType.RCT,
             SyntheticDataConfig(n_samples=1000, treatment_effect=0.8, noise_level=0.05)),
            
            # Observational scenarios
            ("obs_weak_confounding", DatasetType.OBSERVATIONAL,
             SyntheticDataConfig(n_samples=500, treatment_effect=0.4, confounding_strength=0.2)),
            ("obs_strong_confounding", DatasetType.OBSERVATIONAL,
             SyntheticDataConfig(n_samples=500, treatment_effect=0.4, confounding_strength=0.8)),
            ("obs_high_dimensional", DatasetType.OBSERVATIONAL,
             SyntheticDataConfig(n_samples=800, n_features=10, treatment_effect=0.3)),
            
            # IV scenarios
            ("iv_weak_instrument", DatasetType.INSTRUMENTAL_VARIABLE,
             SyntheticDataConfig(n_samples=400, treatment_effect=0.6, instrument_strength=0.3)),
            ("iv_strong_instrument", DatasetType.INSTRUMENTAL_VARIABLE,
             SyntheticDataConfig(n_samples=400, treatment_effect=0.6, instrument_strength=1.0)),
            
            # RDD scenarios
            ("rdd_sharp", DatasetType.REGRESSION_DISCONTINUITY,
             SyntheticDataConfig(n_samples=300, treatment_effect=0.5, bandwidth=1.5)),
            ("rdd_fuzzy", DatasetType.REGRESSION_DISCONTINUITY,
             SyntheticDataConfig(n_samples=500, treatment_effect=0.4, bandwidth=2.0)),
            
            # DiD scenarios
            ("did_balanced_panel", DatasetType.DIFFERENCE_IN_DIFFERENCES,
             SyntheticDataConfig(n_periods=20, n_units=50, treatment_effect=0.3)),
            ("did_unbalanced_panel", DatasetType.DIFFERENCE_IN_DIFFERENCES,
             SyntheticDataConfig(n_periods=30, n_units=100, treatment_effect=0.4)),
        ]
        
        for name, dataset_type, config in benchmark_configs:
            generator.config = config
            dataset = generator.generate_dataset(dataset_type)
            
            # Create metadata
            metadata = DatasetMetadata(
                name=name,
                dataset_type="benchmark",
                n_samples=len(dataset),
                n_features=len([col for col in dataset.columns if col not in ['treatment', 'outcome']]),
                true_treatment_effect=config.treatment_effect,
                description=f"Benchmark dataset for {dataset_type.value} with {name.split('_')[1]} characteristics",
                tags=["benchmark", dataset_type.value, name.split('_')[1]]
            )
            
            # Register dataset
            self.register_dataset(name, dataset, metadata, overwrite=True)
            benchmark_datasets[name] = dataset
        
        return benchmark_datasets
    
    def get_validation_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get datasets specifically designed for method validation."""
        validation_datasets = {}
        
        # Load or create validation datasets
        validation_names = [
            "rct_medium_effect",
            "obs_weak_confounding", 
            "obs_strong_confounding",
            "iv_strong_instrument",
            "rdd_sharp",
            "did_balanced_panel"
        ]
        
        for name in validation_names:
            try:
                dataset, _ = self.load_dataset(name)
                validation_datasets[name] = dataset
            except (ValueError, FileNotFoundError):
                # Create if doesn't exist
                if name not in self.create_benchmark_suite():
                    continue
                dataset, _ = self.load_dataset(name)
                validation_datasets[name] = dataset
        
        return validation_datasets
    
    def get_performance_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get datasets for performance testing with varying sizes."""
        generator = SyntheticDataGenerator()
        performance_datasets = {}
        
        # Different sizes for performance testing
        sizes = [100, 500, 1000, 5000, 10000]
        
        for size in sizes:
            name = f"performance_n{size}"
            
            try:
                dataset, _ = self.load_dataset(name)
            except (ValueError, FileNotFoundError):
                # Create performance dataset
                config = SyntheticDataConfig(
                    n_samples=size,
                    treatment_effect=0.5,
                    n_features=5,
                    random_seed=42
                )
                generator.config = config
                dataset = generator.generate_observational_data()
                
                metadata = DatasetMetadata(
                    name=name,
                    dataset_type="benchmark",
                    n_samples=size,
                    n_features=5,
                    true_treatment_effect=0.5,
                    description=f"Performance testing dataset with {size} samples",
                    tags=["performance", "observational"]
                )
                
                self.register_dataset(name, dataset, metadata, overwrite=True)
            
            performance_datasets[name] = dataset
        
        return performance_datasets


# Global shared dataset manager instance
shared_dataset_manager = SharedDatasetManager()


def get_standard_datasets() -> Dict[str, pd.DataFrame]:
    """Get the standard set of datasets for testing."""
    return shared_dataset_manager.get_validation_datasets()


def get_benchmark_datasets() -> Dict[str, pd.DataFrame]:
    """Get benchmark datasets for comprehensive testing."""
    return shared_dataset_manager.create_benchmark_suite()


def get_performance_datasets() -> Dict[str, pd.DataFrame]:
    """Get datasets for performance testing."""
    return shared_dataset_manager.get_performance_datasets()


def load_real_world_datasets() -> Dict[str, pd.DataFrame]:
    """Load real-world datasets for testing (if available)."""
    # This would load actual research datasets
    # For now, return empty dict as placeholder
    real_datasets = {}
    
    # Example of how real datasets would be loaded:
    # try:
    #     lalonde_data = pd.read_csv("path/to/lalonde.csv")
    #     real_datasets["lalonde"] = lalonde_data
    # except FileNotFoundError:
    #     pass
    
    return real_datasets


def create_method_specific_datasets() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create datasets optimized for testing specific methods."""
    generator = SyntheticDataGenerator()
    
    method_datasets = {
        "backdoor_adjustment": {},
        "propensity_score": {},
        "instrumental_variable": {},
        "regression_discontinuity": {},
        "difference_in_differences": {},
        "linear_regression": {}
    }
    
    # Backdoor adjustment datasets
    for scenario in ["ideal", "weak_overlap", "high_dimensional"]:
        config = SyntheticDataConfig(
            n_samples=500 if scenario != "high_dimensional" else 1000,
            n_features=5 if scenario != "high_dimensional" else 15,
            treatment_effect=0.5,
            confounding_strength=0.3 if scenario != "weak_overlap" else 0.8
        )
        generator.config = config
        dataset = generator.generate_observational_data()
        method_datasets["backdoor_adjustment"][scenario] = dataset
    
    # Propensity score datasets
    for scenario in ["good_overlap", "poor_overlap", "many_confounders"]:
        config = SyntheticDataConfig(
            n_samples=600,
            n_features=8 if scenario == "many_confounders" else 4,
            treatment_effect=0.4,
            confounding_strength=0.4 if scenario != "poor_overlap" else 0.9
        )
        generator.config = config
        dataset = generator.generate_observational_data()
        method_datasets["propensity_score"][scenario] = dataset
    
    # IV datasets
    for scenario in ["strong_instrument", "weak_instrument", "multiple_instruments"]:
        config = SyntheticDataConfig(
            n_samples=400,
            treatment_effect=0.6,
            instrument_strength=0.9 if scenario == "strong_instrument" else 0.2
        )
        generator.config = config
        dataset = generator.generate_iv_data()
        method_datasets["instrumental_variable"][scenario] = dataset
    
    # RDD datasets
    for scenario in ["sharp_cutoff", "fuzzy_cutoff", "nonlinear_trend"]:
        config = SyntheticDataConfig(
            n_samples=350,
            treatment_effect=0.5,
            bandwidth=1.5 if scenario != "fuzzy_cutoff" else 2.5
        )
        generator.config = config
        dataset = generator.generate_rdd_data()
        method_datasets["regression_discontinuity"][scenario] = dataset
    
    # DiD datasets
    for scenario in ["parallel_trends", "violation_trends", "staggered_treatment"]:
        config = SyntheticDataConfig(
            n_periods=25,
            n_units=40,
            treatment_effect=0.3,
            treatment_start_period=12
        )
        generator.config = config
        dataset = generator.generate_did_data()
        method_datasets["difference_in_differences"][scenario] = dataset
    
    return method_datasets
"""Test configuration management system for causal_agent tests."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import tempfile


class TestEnvironment(Enum):
    """Test environment types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"
    CI = "continuous_integration"
    LOCAL = "local"


class LogLevel(Enum):
    """Logging levels for tests."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LLMConfig:
    """Configuration for LLM-related testing."""
    mock_llm: bool = True
    use_real_llm: bool = False
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key_env_var: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
    max_retries: int = 3
    temperature: float = 0.0  # Deterministic for testing
    mock_response_delay: float = 0.1  # Simulate API delay


@dataclass
class DataConfig:
    """Configuration for test data management."""
    use_synthetic_data: bool = True
    synthetic_data_seed: int = 42
    cache_datasets: bool = True
    dataset_cache_dir: Optional[str] = None
    max_dataset_size: int = 10000
    cleanup_temp_data: bool = True
    data_validation: bool = True
    missing_data_threshold: float = 0.1


@dataclass
class PerformanceConfig:
    """Configuration for performance testing."""
    max_execution_time_seconds: float = 10.0
    max_memory_usage_mb: float = 500.0
    enable_profiling: bool = False
    profile_output_dir: Optional[str] = None
    benchmark_iterations: int = 3
    warmup_iterations: int = 1
    statistical_significance_level: float = 0.05


@dataclass
class CoverageConfig:
    """Configuration for test coverage."""
    minimum_coverage_percentage: float = 80.0
    coverage_report_format: List[str] = field(default_factory=lambda: ["html", "xml"])
    coverage_output_dir: Optional[str] = None
    exclude_patterns: List[str] = field(default_factory=lambda: ["*/tests/*", "*/conftest.py"])
    include_patterns: List[str] = field(default_factory=lambda: ["causal_agent/*"])


@dataclass
class CIConfig:
    """Configuration for CI/CD testing."""
    python_versions: List[str] = field(default_factory=lambda: ["3.10", "3.11", "3.12"])
    operating_systems: List[str] = field(default_factory=lambda: ["ubuntu-latest", "macos-latest", "windows-latest"])
    test_matrix_strategy: str = "full"  # "full", "minimal", "custom"
    parallel_jobs: int = 4
    artifact_retention_days: int = 30
    notification_on_failure: bool = True


@dataclass
class CausalAgentTestConfig:
    """Main test configuration class."""
    # Environment settings
    environment: TestEnvironment = TestEnvironment.LOCAL
    debug_mode: bool = False
    verbose_output: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    ci: CIConfig = field(default_factory=CIConfig)
    
    # Test execution settings
    fail_fast: bool = False
    random_seed: int = 42
    test_timeout_seconds: int = 300
    retry_failed_tests: bool = False
    max_test_retries: int = 2
    
    # Output and reporting
    output_dir: Optional[str] = None
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["junit", "html"])
    
    # Method-specific settings
    method_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class TestConfigManager:
    """Manager for test configuration with environment-specific overrides."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.config_file = Path(config_file) if config_file else None
        self.config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> CausalAgentTestConfig:
        """Load configuration from file or create default."""
        if self.config_file and self.config_file.exists():
            return self._load_from_file(self.config_file)
        else:
            return self._create_default_config()
    
    def _load_from_file(self, config_file: Path) -> CausalAgentTestConfig:
        """Load configuration from YAML or JSON file."""
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CausalAgentTestConfig:
        """Convert dictionary to TestConfig object."""
        # Handle nested configurations
        if 'llm' in config_dict:
            config_dict['llm'] = LLMConfig(**config_dict['llm'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'performance' in config_dict:
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        if 'coverage' in config_dict:
            config_dict['coverage'] = CoverageConfig(**config_dict['coverage'])
        if 'ci' in config_dict:
            config_dict['ci'] = CIConfig(**config_dict['ci'])
        
        # Handle enums
        if 'environment' in config_dict:
            config_dict['environment'] = TestEnvironment(config_dict['environment'])
        if 'log_level' in config_dict:
            config_dict['log_level'] = LogLevel(config_dict['log_level'])
        
        return CausalAgentTestConfig(**config_dict)
    
    def _create_default_config(self) -> CausalAgentTestConfig:
        """Create default configuration."""
        config = CausalAgentTestConfig()
        
        # Environment-specific defaults
        if os.getenv('CI'):
            config.environment = TestEnvironment.CI
            config.llm.mock_llm = True
            config.data.cleanup_temp_data = True
            config.performance.enable_profiling = False
        
        # Override with environment variables
        self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: CausalAgentTestConfig):
        """Apply environment variable overrides."""
        env_mappings = {
            'CAUSAL_AGENT_TEST_DEBUG': ('debug_mode', bool),
            'CAUSAL_AGENT_TEST_VERBOSE': ('verbose_output', bool),
            'CAUSAL_AGENT_TEST_MOCK_LLM': ('llm.mock_llm', bool),
            'CAUSAL_AGENT_TEST_SEED': ('random_seed', int),
            'CAUSAL_AGENT_TEST_TIMEOUT': ('test_timeout_seconds', int),
            'CAUSAL_AGENT_TEST_MAX_MEMORY': ('performance.max_memory_usage_mb', float),
            'CAUSAL_AGENT_TEST_MIN_COVERAGE': ('coverage.minimum_coverage_percentage', float),
        }
        
        for env_var, (config_path, config_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert to appropriate type
                    if config_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = config_type(env_value)
                    
                    # Set nested attribute
                    self._set_nested_attr(config, config_path, value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value for {env_var}: {env_value}")
    
    def _set_nested_attr(self, obj: Any, path: str, value: Any):
        """Set nested attribute using dot notation."""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _setup_directories(self):
        """Setup required directories."""
        # Output directory
        if not self.config.output_dir:
            self.config.output_dir = str(Path.cwd() / "test_output")
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Dataset cache directory
        if not self.config.data.dataset_cache_dir:
            self.config.data.dataset_cache_dir = str(output_path / "dataset_cache")
        
        cache_path = Path(self.config.data.dataset_cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        # Coverage output directory
        if not self.config.coverage.coverage_output_dir:
            self.config.coverage.coverage_output_dir = str(output_path / "coverage")
        
        coverage_path = Path(self.config.coverage.coverage_output_dir)
        coverage_path.mkdir(exist_ok=True)
        
        # Profile output directory
        if self.config.performance.enable_profiling and not self.config.performance.profile_output_dir:
            self.config.performance.profile_output_dir = str(output_path / "profiles")
            profile_path = Path(self.config.performance.profile_output_dir)
            profile_path.mkdir(exist_ok=True)
    
    def get_config(self) -> CausalAgentTestConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.custom_settings[key] = value
    
    def save_config(self, output_file: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        if output_file is None:
            output_file = Path(self.config.output_dir) / "test_config.yaml"
        
        output_path = Path(output_file)
        config_dict = self._config_to_dict(self.config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _config_to_dict(self, config: CausalAgentTestConfig) -> Dict[str, Any]:
        """Convert TestConfig to dictionary for serialization."""
        config_dict = asdict(config)
        
        # Handle enums
        config_dict['environment'] = config.environment.value
        config_dict['log_level'] = config.log_level.value
        
        return config_dict
    
    def get_method_config(self, method_name: str) -> Dict[str, Any]:
        """Get configuration for a specific causal method."""
        return self.config.method_configs.get(method_name, {})
    
    def set_method_config(self, method_name: str, method_config: Dict[str, Any]):
        """Set configuration for a specific causal method."""
        self.config.method_configs[method_name] = method_config
    
    def create_temp_config(self, **overrides) -> 'TestConfigManager':
        """Create a temporary configuration with overrides."""
        temp_config = CausalAgentTestConfig(**asdict(self.config))
        
        for key, value in overrides.items():
            if hasattr(temp_config, key):
                setattr(temp_config, key, value)
        
        temp_manager = TestConfigManager()
        temp_manager.config = temp_config
        return temp_manager
    
    def is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        return self.config.environment == TestEnvironment.CI or bool(os.getenv('CI'))
    
    def should_use_real_llm(self) -> bool:
        """Check if real LLM should be used (not mocked)."""
        return (not self.config.llm.mock_llm and 
                self.config.llm.use_real_llm and 
                os.getenv(self.config.llm.api_key_env_var))
    
    def get_test_markers(self) -> List[str]:
        """Get pytest markers based on configuration."""
        markers = []
        
        if self.config.llm.mock_llm:
            markers.append("mock_llm")
        else:
            markers.append("requires_llm")
        
        if self.config.performance.enable_profiling:
            markers.append("profile")
        
        markers.append(f"env_{self.config.environment.value}")
        
        return markers


# Global configuration manager
_config_manager = None


def get_test_config() -> CausalAgentTestConfig:
    """Get the global test configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TestConfigManager()
    return _config_manager.get_config()


def get_config_manager() -> TestConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TestConfigManager()
    return _config_manager


def setup_test_config(config_file: Optional[Union[str, Path]] = None,
                     **overrides) -> TestConfigManager:
    """Setup test configuration with optional file and overrides."""
    global _config_manager
    _config_manager = TestConfigManager(config_file)
    
    if overrides:
        _config_manager.update_config(overrides)
    
    return _config_manager


def create_test_config_for_method(method_name: str, **method_overrides) -> CausalAgentTestConfig:
    """Create test configuration optimized for a specific method."""
    config = get_test_config()
    
    # Method-specific optimizations
    method_configs = {
        "backdoor_adjustment": {
            "max_execution_time": 5.0,
            "required_sample_size": 200,
            "confounders_required": True
        },
        "propensity_score": {
            "max_execution_time": 8.0,
            "required_sample_size": 300,
            "overlap_check": True
        },
        "instrumental_variable": {
            "max_execution_time": 10.0,
            "required_sample_size": 400,
            "instrument_strength_check": True
        },
        "regression_discontinuity": {
            "max_execution_time": 7.0,
            "required_sample_size": 250,
            "bandwidth_optimization": True
        },
        "difference_in_differences": {
            "max_execution_time": 12.0,
            "required_sample_size": 500,
            "parallel_trends_check": True
        }
    }
    
    if method_name in method_configs:
        method_config = method_configs[method_name]
        method_config.update(method_overrides)
        
        # Update performance settings
        config.performance.max_execution_time_seconds = method_config.get(
            "max_execution_time", config.performance.max_execution_time_seconds
        )
        
        # Store method-specific config
        config.method_configs[method_name] = method_config
    
    return config
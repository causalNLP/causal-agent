"""
Tests for causal_agent.tools.data_analyzer module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from causal_agent.tools.data_analyzer import DataAnalyzer


class TestDataAnalyzer:
    """Test DataAnalyzer class"""
    
    def test_data_analyzer_initialization_default(self):
        """Test DataAnalyzer initialization with default parameters"""
        analyzer = DataAnalyzer()
        assert analyzer is not None
        assert analyzer.verbose is False
    
    def test_data_analyzer_initialization_verbose(self):
        """Test DataAnalyzer initialization with verbose=True"""
        analyzer = DataAnalyzer(verbose=True)
        assert analyzer is not None
        assert analyzer.verbose is True
    
    def test_data_analyzer_has_required_attributes(self):
        """Test that DataAnalyzer has required attributes"""
        analyzer = DataAnalyzer()
        assert hasattr(analyzer, 'verbose')
    
    def test_data_analyzer_type(self):
        """Test DataAnalyzer type"""
        analyzer = DataAnalyzer()
        assert isinstance(analyzer, DataAnalyzer)
    
    def test_data_analyzer_verbose_setting(self):
        """Test verbose setting functionality"""
        analyzer_quiet = DataAnalyzer(verbose=False)
        analyzer_verbose = DataAnalyzer(verbose=True)
        
        assert analyzer_quiet.verbose is False
        assert analyzer_verbose.verbose is True
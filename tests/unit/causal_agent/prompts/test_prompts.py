"""
Tests for causal_agent.prompts.prompts module
"""
import pytest
from causal_agent.prompts.prompts import (
    RCT_IDENTIFICATION_PROMPT
)


class TestPrompts:
    """Test prompt constants and functions"""
    
    def test_rct_identification_prompt_exists(self):
        """Test that RCT identification prompt is defined"""
        assert RCT_IDENTIFICATION_PROMPT is not None
        assert isinstance(RCT_IDENTIFICATION_PROMPT, str)
        assert len(RCT_IDENTIFICATION_PROMPT) > 0
    
    def test_rct_prompt_contains_expected_keywords(self):
        """Test that RCT prompt contains expected keywords"""
        prompt_lower = RCT_IDENTIFICATION_PROMPT.lower()
        
        # Should contain RCT-related keywords
        assert any(keyword in prompt_lower for keyword in 
                  ['rct', 'randomized', 'controlled', 'trial'])
        
        # Should contain causal inference keywords
        assert any(keyword in prompt_lower for keyword in 
                  ['causal', 'inference', 'treatment'])
        
        # Should contain JSON format instructions
        assert 'json' in prompt_lower
        assert 'is_rct' in prompt_lower
    
    def test_rct_prompt_format_instructions(self):
        """Test that RCT prompt contains proper format instructions"""
        assert '{description}' in RCT_IDENTIFICATION_PROMPT
        assert '{column_info}' in RCT_IDENTIFICATION_PROMPT
        assert '"is_rct"' in RCT_IDENTIFICATION_PROMPT
    
    def test_rct_prompt_examples(self):
        """Test that RCT prompt contains examples"""
        assert 'Examples:' in RCT_IDENTIFICATION_PROMPT
        assert 'true' in RCT_IDENTIFICATION_PROMPT
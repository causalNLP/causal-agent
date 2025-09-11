"""Mock LLM response fixtures for deterministic testing."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json


class LLMResponseType(Enum):
    """Types of LLM responses that can be mocked."""
    METHOD_SELECTION = "method_selection"
    DATASET_ANALYSIS = "dataset_analysis"
    RESULT_INTERPRETATION = "result_interpretation"
    ASSUMPTION_VALIDATION = "assumption_validation"
    DIAGNOSTIC_ANALYSIS = "diagnostic_analysis"
    QUERY_PARSING = "query_parsing"
    ERROR_EXPLANATION = "error_explanation"


@dataclass
class MockLLMResponse:
    """Structure for mock LLM responses."""
    response_type: LLMResponseType
    content: Dict[str, Any]
    confidence: float = 0.8
    reasoning: str = "Mock reasoning for testing"
    metadata: Optional[Dict[str, Any]] = None


class MockLLMResponseGenerator:
    """Generator for consistent mock LLM responses."""
    
    def __init__(self):
        """Initialize the mock response generator."""
        self._responses = self._initialize_responses()
    
    def _initialize_responses(self) -> Dict[LLMResponseType, List[MockLLMResponse]]:
        """Initialize predefined mock responses for different scenarios."""
        responses = {
            LLMResponseType.METHOD_SELECTION: [
                MockLLMResponse(
                    response_type=LLMResponseType.METHOD_SELECTION,
                    content={
                        "recommended_method": "backdoor_adjustment",
                        "confidence": 0.85,
                        "reasoning": "Dataset appears to have sufficient confounders measured for backdoor adjustment",
                        "alternative_methods": ["propensity_score", "linear_regression"],
                        "assumptions": [
                            "No unmeasured confounders",
                            "Positivity assumption holds",
                            "Consistency assumption"
                        ],
                        "method_parameters": {
                            "adjustment_set": ["confounder_1", "confounder_2"]
                        }
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.METHOD_SELECTION,
                    content={
                        "recommended_method": "propensity_score",
                        "confidence": 0.75,
                        "reasoning": "High-dimensional confounders suggest propensity score methods",
                        "alternative_methods": ["backdoor_adjustment", "matching"],
                        "assumptions": [
                            "Unconfoundedness",
                            "Overlap assumption",
                            "Stable unit treatment value assumption"
                        ],
                        "method_parameters": {
                            "matching_method": "nearest_neighbor",
                            "caliper": 0.1
                        }
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.METHOD_SELECTION,
                    content={
                        "recommended_method": "instrumental_variable",
                        "confidence": 0.9,
                        "reasoning": "Strong instrument detected with good relevance and exogeneity",
                        "alternative_methods": ["two_stage_least_squares"],
                        "assumptions": [
                            "Instrument relevance",
                            "Instrument exogeneity",
                            "Exclusion restriction"
                        ],
                        "method_parameters": {
                            "instrument": "instrument",
                            "first_stage_controls": ["covariate_0", "covariate_1"]
                        }
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.METHOD_SELECTION,
                    content={
                        "recommended_method": "regression_discontinuity",
                        "confidence": 0.8,
                        "reasoning": "Clear discontinuity in treatment assignment detected",
                        "alternative_methods": ["local_linear_regression"],
                        "assumptions": [
                            "Continuity of potential outcomes",
                            "No manipulation of running variable",
                            "Local randomization around cutoff"
                        ],
                        "method_parameters": {
                            "running_variable": "running_var",
                            "cutoff": 0.0,
                            "bandwidth": "optimal"
                        }
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.METHOD_SELECTION,
                    content={
                        "recommended_method": "difference_in_differences",
                        "confidence": 0.7,
                        "reasoning": "Panel data structure with parallel trends assumption plausible",
                        "alternative_methods": ["fixed_effects", "synthetic_control"],
                        "assumptions": [
                            "Parallel trends",
                            "No spillover effects",
                            "Stable composition"
                        ],
                        "method_parameters": {
                            "unit_variable": "unit",
                            "time_variable": "period",
                            "treatment_start": 10
                        }
                    }
                )
            ],
            
            LLMResponseType.DATASET_ANALYSIS: [
                MockLLMResponse(
                    response_type=LLMResponseType.DATASET_ANALYSIS,
                    content={
                        "dataset_summary": {
                            "n_observations": 500,
                            "n_features": 5,
                            "treatment_variable": "treatment",
                            "outcome_variable": "outcome",
                            "data_type": "cross_sectional"
                        },
                        "data_quality": {
                            "missing_data_percentage": 2.5,
                            "outlier_percentage": 1.2,
                            "balance_score": 0.85,
                            "quality_score": 0.9
                        },
                        "variable_analysis": {
                            "potential_confounders": ["confounder_1", "confounder_2"],
                            "potential_instruments": [],
                            "potential_mediators": [],
                            "running_variables": []
                        },
                        "recommendations": [
                            "Consider propensity score matching due to imbalance",
                            "Check for non-linear relationships",
                            "Validate measurement quality of outcome variable"
                        ]
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.DATASET_ANALYSIS,
                    content={
                        "dataset_summary": {
                            "n_observations": 1000,
                            "n_features": 8,
                            "treatment_variable": "treatment",
                            "outcome_variable": "outcome",
                            "data_type": "panel"
                        },
                        "data_quality": {
                            "missing_data_percentage": 0.5,
                            "outlier_percentage": 2.1,
                            "balance_score": 0.92,
                            "quality_score": 0.95
                        },
                        "variable_analysis": {
                            "potential_confounders": ["unit_fixed_effects", "time_trends"],
                            "potential_instruments": ["policy_instrument"],
                            "potential_mediators": ["intermediate_outcome"],
                            "running_variables": []
                        },
                        "recommendations": [
                            "Panel structure suitable for difference-in-differences",
                            "Test parallel trends assumption",
                            "Consider unit-specific time trends"
                        ]
                    }
                )
            ],
            
            LLMResponseType.RESULT_INTERPRETATION: [
                MockLLMResponse(
                    response_type=LLMResponseType.RESULT_INTERPRETATION,
                    content={
                        "effect_interpretation": {
                            "magnitude": "moderate",
                            "direction": "positive",
                            "significance": "statistically_significant",
                            "practical_significance": "meaningful"
                        },
                        "confidence_assessment": {
                            "statistical_confidence": 0.95,
                            "causal_confidence": 0.8,
                            "robustness_score": 0.75
                        },
                        "interpretation_text": "The estimated treatment effect of 0.45 represents a moderate positive impact. The 95% confidence interval [0.2, 0.7] suggests the effect is statistically significant and practically meaningful.",
                        "caveats": [
                            "Results depend on unconfoundedness assumption",
                            "Effect may vary across subpopulations",
                            "Consider sensitivity analysis for unmeasured confounding"
                        ],
                        "recommendations": [
                            "Conduct robustness checks with alternative methods",
                            "Test for heterogeneous treatment effects",
                            "Validate assumptions with domain experts"
                        ]
                    }
                ),
                MockLLMResponse(
                    response_type=LLMResponseType.RESULT_INTERPRETATION,
                    content={
                        "effect_interpretation": {
                            "magnitude": "small",
                            "direction": "negative",
                            "significance": "not_significant",
                            "practical_significance": "negligible"
                        },
                        "confidence_assessment": {
                            "statistical_confidence": 0.6,
                            "causal_confidence": 0.5,
                            "robustness_score": 0.4
                        },
                        "interpretation_text": "The estimated treatment effect of -0.05 is small and not statistically significant. The wide confidence interval [-0.3, 0.2] suggests high uncertainty.",
                        "caveats": [
                            "Low statistical power may mask true effects",
                            "Potential measurement error in key variables",
                            "Sample size may be insufficient"
                        ],
                        "recommendations": [
                            "Increase sample size if possible",
                            "Improve measurement precision",
                            "Consider alternative identification strategies"
                        ]
                    }
                )
            ],
            
            LLMResponseType.ASSUMPTION_VALIDATION: [
                MockLLMResponse(
                    response_type=LLMResponseType.ASSUMPTION_VALIDATION,
                    content={
                        "assumptions_checked": [
                            {
                                "assumption": "unconfoundedness",
                                "status": "plausible",
                                "evidence": "Comprehensive set of pre-treatment variables included",
                                "confidence": 0.8
                            },
                            {
                                "assumption": "positivity",
                                "status": "satisfied",
                                "evidence": "Overlap in propensity scores across treatment groups",
                                "confidence": 0.9
                            },
                            {
                                "assumption": "consistency",
                                "status": "assumed",
                                "evidence": "Single well-defined treatment intervention",
                                "confidence": 0.95
                            }
                        ],
                        "overall_validity": "moderate",
                        "main_concerns": [
                            "Potential unmeasured confounding from socioeconomic factors",
                            "Treatment assignment mechanism not fully understood"
                        ],
                        "sensitivity_analysis_needed": True,
                        "recommendations": [
                            "Conduct sensitivity analysis for unmeasured confounding",
                            "Collect additional pre-treatment variables if possible",
                            "Consider instrumental variable approach as robustness check"
                        ]
                    }
                )
            ],
            
            LLMResponseType.DIAGNOSTIC_ANALYSIS: [
                MockLLMResponse(
                    response_type=LLMResponseType.DIAGNOSTIC_ANALYSIS,
                    content={
                        "balance_diagnostics": {
                            "standardized_mean_differences": {
                                "confounder_1": 0.05,
                                "confounder_2": 0.12,
                                "overall": 0.08
                            },
                            "variance_ratios": {
                                "confounder_1": 1.02,
                                "confounder_2": 0.95,
                                "overall": 0.98
                            },
                            "balance_assessment": "good"
                        },
                        "overlap_diagnostics": {
                            "propensity_score_overlap": 0.92,
                            "common_support_percentage": 95.5,
                            "overlap_assessment": "excellent"
                        },
                        "model_diagnostics": {
                            "r_squared": 0.75,
                            "residual_patterns": "no_obvious_patterns",
                            "model_fit": "good"
                        },
                        "overall_assessment": "diagnostics_passed",
                        "warnings": [],
                        "recommendations": [
                            "Proceed with causal analysis",
                            "Results appear reliable given diagnostic checks"
                        ]
                    }
                )
            ],
            
            LLMResponseType.QUERY_PARSING: [
                MockLLMResponse(
                    response_type=LLMResponseType.QUERY_PARSING,
                    content={
                        "parsed_query": {
                            "causal_question": "effect_estimation",
                            "treatment_variable": "treatment",
                            "outcome_variable": "outcome",
                            "confounders": ["confounder_1", "confounder_2"],
                            "population": "all_units",
                            "estimand": "average_treatment_effect"
                        },
                        "query_type": "standard_causal_inference",
                        "complexity": "moderate",
                        "clarity_score": 0.85,
                        "missing_information": [
                            "Time frame for effect measurement",
                            "Specific subpopulations of interest"
                        ],
                        "suggested_clarifications": [
                            "Specify if interested in immediate or long-term effects",
                            "Clarify if heterogeneous effects are of interest"
                        ]
                    }
                )
            ],
            
            LLMResponseType.ERROR_EXPLANATION: [
                MockLLMResponse(
                    response_type=LLMResponseType.ERROR_EXPLANATION,
                    content={
                        "error_type": "convergence_failure",
                        "error_message": "Optimization algorithm failed to converge",
                        "likely_causes": [
                            "Insufficient sample size",
                            "Multicollinearity in covariates",
                            "Extreme propensity score values"
                        ],
                        "suggested_solutions": [
                            "Increase sample size if possible",
                            "Remove highly correlated variables",
                            "Apply propensity score trimming",
                            "Try alternative estimation method"
                        ],
                        "severity": "moderate",
                        "can_proceed": False,
                        "alternative_methods": ["linear_regression", "matching"]
                    }
                )
            ]
        }
        
        return responses
    
    def get_response(self, 
                    response_type: LLMResponseType, 
                    scenario: str = "default",
                    custom_content: Optional[Dict[str, Any]] = None) -> MockLLMResponse:
        """Get a mock LLM response for the specified type and scenario."""
        if response_type not in self._responses:
            raise ValueError(f"Unknown response type: {response_type}")
        
        responses = self._responses[response_type]
        
        # Select response based on scenario
        if scenario == "default" or len(responses) == 1:
            response = responses[0]
        elif scenario == "alternative" and len(responses) > 1:
            response = responses[1]
        elif scenario == "error" and response_type == LLMResponseType.METHOD_SELECTION:
            # Return error scenario
            response = MockLLMResponse(
                response_type=response_type,
                content={
                    "error": "Unable to determine appropriate method",
                    "reason": "Insufficient information in dataset",
                    "suggestions": ["Collect more data", "Provide domain knowledge"]
                },
                confidence=0.1
            )
        else:
            # Default to first response
            response = responses[0]
        
        # Override with custom content if provided
        if custom_content:
            response.content.update(custom_content)
        
        return response
    
    def get_method_selection_response(self, 
                                    method: str = "backdoor_adjustment",
                                    confidence: float = 0.8) -> Dict[str, Any]:
        """Get method selection response for specific method."""
        method_responses = {
            "backdoor_adjustment": {
                "recommended_method": "backdoor_adjustment",
                "confidence": confidence,
                "reasoning": "Sufficient confounders available for backdoor adjustment",
                "alternative_methods": ["propensity_score", "linear_regression"],
                "assumptions": ["No unmeasured confounders", "Positivity", "Consistency"],
                "method_parameters": {"adjustment_set": ["confounder_1", "confounder_2"]}
            },
            "propensity_score": {
                "recommended_method": "propensity_score",
                "confidence": confidence,
                "reasoning": "High-dimensional confounders favor propensity score methods",
                "alternative_methods": ["backdoor_adjustment", "matching"],
                "assumptions": ["Unconfoundedness", "Overlap", "SUTVA"],
                "method_parameters": {"matching_method": "nearest_neighbor", "caliper": 0.1}
            },
            "instrumental_variable": {
                "recommended_method": "instrumental_variable",
                "confidence": confidence,
                "reasoning": "Valid instrument available for identification",
                "alternative_methods": ["two_stage_least_squares"],
                "assumptions": ["Relevance", "Exogeneity", "Exclusion restriction"],
                "method_parameters": {"instrument": "instrument", "controls": ["covariate_0"]}
            },
            "regression_discontinuity": {
                "recommended_method": "regression_discontinuity",
                "confidence": confidence,
                "reasoning": "Sharp discontinuity in treatment assignment",
                "alternative_methods": ["local_linear_regression"],
                "assumptions": ["Continuity", "No manipulation", "Local randomization"],
                "method_parameters": {"running_variable": "running_var", "cutoff": 0.0}
            },
            "difference_in_differences": {
                "recommended_method": "difference_in_differences",
                "confidence": confidence,
                "reasoning": "Panel structure with plausible parallel trends",
                "alternative_methods": ["fixed_effects", "synthetic_control"],
                "assumptions": ["Parallel trends", "No spillovers", "Stable composition"],
                "method_parameters": {"unit_var": "unit", "time_var": "period"}
            }
        }
        
        return method_responses.get(method, method_responses["backdoor_adjustment"])
    
    def get_dataset_analysis_response(self, 
                                    n_obs: int = 500,
                                    data_quality: float = 0.9) -> Dict[str, Any]:
        """Get dataset analysis response with specified characteristics."""
        return {
            "dataset_summary": {
                "n_observations": n_obs,
                "n_features": 5,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "data_type": "cross_sectional"
            },
            "data_quality": {
                "missing_data_percentage": max(0, (1 - data_quality) * 10),
                "outlier_percentage": max(0, (1 - data_quality) * 5),
                "balance_score": data_quality,
                "quality_score": data_quality
            },
            "variable_analysis": {
                "potential_confounders": ["confounder_1", "confounder_2"],
                "potential_instruments": [],
                "potential_mediators": [],
                "running_variables": []
            },
            "recommendations": [
                "Dataset appears suitable for causal analysis",
                "Consider robustness checks with multiple methods"
            ]
        }
    
    def create_custom_response(self, 
                             response_type: LLMResponseType,
                             content: Dict[str, Any],
                             confidence: float = 0.8,
                             reasoning: str = "Custom mock response") -> MockLLMResponse:
        """Create a custom mock response."""
        return MockLLMResponse(
            response_type=response_type,
            content=content,
            confidence=confidence,
            reasoning=reasoning
        )


# Convenience functions for common mock responses
def mock_method_selection(method: str = "backdoor_adjustment", 
                         confidence: float = 0.8) -> Dict[str, Any]:
    """Quick mock for method selection."""
    generator = MockLLMResponseGenerator()
    return generator.get_method_selection_response(method, confidence)


def mock_dataset_analysis(n_obs: int = 500, 
                         data_quality: float = 0.9) -> Dict[str, Any]:
    """Quick mock for dataset analysis."""
    generator = MockLLMResponseGenerator()
    return generator.get_dataset_analysis_response(n_obs, data_quality)


def mock_result_interpretation(effect_size: float = 0.5,
                             significance: bool = True) -> Dict[str, Any]:
    """Quick mock for result interpretation."""
    return {
        "effect_interpretation": {
            "magnitude": "moderate" if abs(effect_size) > 0.3 else "small",
            "direction": "positive" if effect_size > 0 else "negative",
            "significance": "statistically_significant" if significance else "not_significant",
            "practical_significance": "meaningful" if abs(effect_size) > 0.2 else "negligible"
        },
        "confidence_assessment": {
            "statistical_confidence": 0.95 if significance else 0.6,
            "causal_confidence": 0.8,
            "robustness_score": 0.75
        },
        "interpretation_text": f"The estimated treatment effect of {effect_size:.2f} represents a {'moderate' if abs(effect_size) > 0.3 else 'small'} {'positive' if effect_size > 0 else 'negative'} impact.",
        "caveats": [
            "Results depend on identification assumptions",
            "Effect may vary across subpopulations"
        ],
        "recommendations": [
            "Conduct robustness checks",
            "Test for heterogeneous effects"
        ]
    }


# Global mock response generator instance
mock_llm_generator = MockLLMResponseGenerator()
"""
Prompt templates for dataset analysis functions including identifying 
instrumental variables, assessing variable relationships, and overlap assessment.
"""

# Note: These templates use f-string formatting with dataset-specific variables

INSTRUMENT_IDENTIFICATION_PROMPT = """
You are an expert causal inference assistant helping to identify potential Instrumental Variables (IVs).

I have a dataset with the following characteristics:
- Treatment variable(s): {potential_treatments}
- Outcome variable(s): {potential_outcomes}
- All columns: {all_columns}
- Column types: {column_types}
- Variable relationships: {relationships_info}

An instrumental variable must satisfy three conditions:
1. Relevance: It must be correlated with the treatment variable
2. Exclusion restriction: It must affect the outcome ONLY through the treatment variable (no direct effect)
3. Independence: It must be independent of unobserved confounders affecting the outcome

Based on the column names, types, and relationships, identify potential instrumental variables.
For each potential IV, explain why it might satisfy these conditions.

Return your answer as a list of dictionaries with the following structure:
[
  {{
    "variable": "column_name",
    "reason": "Brief explanation of why this could be an instrumental variable",
    "data_type": "column data type",
    "confidence": "high/medium/low",
    "relevance_assessment": "Brief assessment of condition 1",
    "exclusion_assessment": "Brief assessment of condition 2",
    "independence_assessment": "Brief assessment of condition 3"
  }}
]

If you cannot identify any potential IVs, return an empty list.
"""

OVERLAP_ASSESSMENT_PROMPT = """
You are an expert causal inference assistant helping to assess covariate balance and overlap between treatment and control groups.

Treatment variable: {treatment}
Group sizes:
- Treatment group: {treated_count} observations
- Control group: {control_count} observations

Covariate statistics:
{covariate_stats}

Based on this information, assess:
1. Balance: Are there significant differences in covariates between treatment and control groups?
2. Overlap: Is there sufficient overlap in covariate distributions to make causal comparisons?
3. Sample size: Is the sample size adequate for the analysis?

Your assessment should indicate whether methods like propensity score matching or weighting might be necessary.

Return your assessment as a dictionary with the following structure:
{{
  "balance_assessment": "Good/Moderate/Poor",
  "overlap_assessment": "Good/Moderate/Poor",
  "sample_size_assessment": "Adequate/Limited",
  "problematic_covariates": ["list", "of", "unbalanced", "covariates"],
  "recommendation": "Brief recommendation for addressing any issues",
  "reasoning": "Brief explanation of your assessment"
}}
""" 
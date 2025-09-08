"""
Prompt templates for identifying specific causal structures (IV, RDD, RCT)
within the query_interpreter component.
"""

# Note: These templates expect f-string formatting with variables like:
# query, description, column_info, treatment, outcome

IV_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to assess whether a valid instrument exists in the dataset to apply Instrumental Variable (IV) methods for estimating causal effects to answer the user query. Use systematic reasoning to evaluate potential IVs against strict criteria.

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 1: Understand the causal question
- What causal effect is the user trying to estimate?
- What potential unobserved confounders might bias the treatment-outcome relationship?

Step 2: Identify potential instrumental variables
- Which columns could plausibly serve as instruments?
- Exclude the treatment and outcome variables themselves
- Look for variables that might influence treatment assignment
- Only consider variables that exist in the available columns list

Step 3: Evaluate each potential IV against the four key conditions

Condition 1 - Relevance:
- Does this variable causally influence the treatment?

Condition 2 - Exclusion restriction:
- Does this variable affect the outcome only through the treatment?
- There should be no direct pathway from instrument to outcome bypassing treatment

Condition 3 - Independence:
- Is this variable as good as randomly assigned with respect to unobserved confounders?

Condition 4 - Compliance (for experimental data only):
- If this is experimental data, is there compliance variation?
- Do we have data showing some units didn't follow their assigned treatment?

Step 4: Make final determination
- Select the best candidate only if it satisfies all conditions
- If no good candidates exist, return null
- Only select variables from the available columns, do not create new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "instrument_variable": "column_name_or_null"
}}
"""

RDD_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify if a running variable exists for performing Regression Discontinuity Design (RDD) to answer the user query. Go through the data description and available columns carefully. In RDD, treatment assignment is determined by whether a continuous variable crosses a specific threshold.

User Query: "{query}"
Dataset Description: {description}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 1: Analyze the research design
- Does the query or description imply a cutoff, threshold, or eligibility criterion?
- Look for evidence that a threshold value dictates whether a person receives the intervention.

Step 2: Identify the running variable from context
- What continuous variable determines treatment assignment in this design?
- This should be clearly implied from the query or dataset description.

Step 3: Validate the running variable exists in data
- Is the identified running variable actually present in the available columns?
- Do not choose variables randomly. If this is not apparent from the provided information, return null.
- Only select variables that exist in the available columns list

Step 4: Identify the cutoff value from design
- What specific threshold value determines treatment assignment?
- This should be implied from the description.

Step 5: Final determination
- Only suggest RDD if both running variable and cutoff are clearly identified from the design context
- Return null if the assignment mechanism is not threshold-based or you are unsure.

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "running_variable": "column_name_or_null",
    "cutoff_value": numeric_value_or_null
}}
"""

RCT_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to determine if the data comes from a Randomized Controlled Trial (RCT) to answer the user query. This assessment will help determine if experimental causal inference methods are appropriate based on the study description.

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 1: Look for randomization indicators
- Does the description mention "random", "randomized", "randomly assigned", "control group", or "trial"?

Step 2: Assess assignment method
- Was treatment assigned randomly or through natural/self-selection processes?

Step 3: Make determination
- True if random assignment is clearly indicated
- False if treatment appears naturally determined
- Null if unclear

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "is_rct": true_false_or_null
}}
"""

INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to determine if an interaction term is required to answer the user query about heterogeneous treatment effects. This assessment will identify if the analysis should examine differential treatment effects across subgroups.

User Query: "{query}"
Dataset Description: "{description}"
Treatment Variable: "{treatment_variable}"
Available Covariates: {covariates_list_with_types}

Think through this step by step:

Step 1: Analyze the query for subgroup language
- Does the query explicitly ask about treatment effects for specific subgroups?
- Look for phrases like "effect for men vs women", "does treatment work differently for", "among elderly patients"

Step 2: Distinguish between subgroup analysis and overall effects
- Is the query asking for differential effects across groups (interaction needed)?
- Or is it asking for overall average effects (no interaction needed)?

Step 3: Identify the relevant covariate
- Which covariate from the available list would define the subgroups mentioned?
- Ensure the covariate actually exists in the dataset
- Only select variables that exist in the available covariates list

Step 4: Make final determination
- Only suggest interaction if subgroup comparison is explicitly mentioned
- Default to no interaction if unclear or asking for overall effects

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "interaction_needed": true_or_false,
    "interaction_variable": "covariate_name_or_null",
    "reasoning": "brief_explanation"
}}
"""

TREATMENT_VAR_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify the treatment variable to perform causal analysis that answers the user query. The treatment variable represents the actual intervention or exposure received by units.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}

Think through this step by step:

Step 1: Understand the causal question
- What is the specific treatment whose causal effect needs to be estimated i.e what is the intervention of interest?

Step 2: Distinguish between assignment and actual treatment
- For experimental data: Is this an encouragement design where assignment differs from actual uptake?
- Look for variables representing what units actually received vs what they were assigned
- In encouragement designs, choose the actual treatment received, not the random assignment.

Step 3: Identify the treatment variable
- Which column represents the actual intervention or exposure of interest?
- Ensure this variable captures what units actually experienced, not just what was intended
- For IV/encouragement designs, this should be the endogenous treatment, not the instrument
- Only select variables that exist in the available variables list

Step 4: Validate the choice
- Does this variable directly represent the causal factor mentioned in the query?
- For encouragement designs, is there variation between assignment and uptake?

Step 5: Make final determination
- Return the variable representing actual treatment received
- Return null if no clear treatment variable can be identified from the information provided
- Do not create or suggest new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "treatment": "column_name_or_null"
}}
"""

OUTCOME_VAR_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify the outcome variable to perform causal analysis that answers the user query. The outcome variable represents what you want to measure the effect on.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}

Think through this step by step:

Step 1: Identify what the query is measuring
- What outcome or effect is the user asking about?
- What variable would represent the result or consequence of the treatment?

Step 2: Find the matching variable
- Which column in the dataset corresponds to this outcome?
- Ensure it represents the dependent variable, not the treatment
- Only select variables that exist in the available variables list

Step 3: Make determination
- Return the variable that captures the outcome of interest
- Return null if no clear outcome variable can be identified
- Do not create or suggest new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "outcome": "column_name_or_null"
}}
"""

COVARIATES_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify the pre-treatment variables in a dataset that can be used as controls in a causal estimation model to answer the user's query.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}
The treatment variable is: {treatment}
The outcome variable is: {outcome}

Pre-treatment variables are those that are measured before the treatment is applied and are not affected by the treatment. These variables can be used as controls in the causal model.

For example, say we have an RCT with outcome Y, treatment T, and pre-treatment variables X1, X2, and X3. We can perform a regression of the form: Y ~ T + X1 + X2 + X3.

Based on the information above, return a list of variables that qualify as pre-treatment variables from the available columns. Only select variables that exist in the available variables list. Do not create or suggest new variables.

If no suitable pre-treatment variables can be identified, return an empty list.

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "controls": ["list_of_column_names_or_empty_list"]
}}
"""

CONTROLS_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify control variables to improve the causal estimation for answering the user query. Control variables are pre-treatment variables that can reduce bias and improve precision.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}
Treatment Variable: {treatment}
Outcome Variable: {outcome}
Causal Method: {method}

Think through this step by step:

Step 1: Understand control variable requirements
- Controls must be measured before treatment occurs
- Controls should not be affected by the treatment (pre-treatment variables only)
- Controls should be related to the outcome but not problematic for the chosen method

Step 2: Apply method-specific criteria
- For RCT: Include pre-treatment variables to improve precision
- For IV: Include variables that affect the outcome but not the treatment or instrument
- For RDD: Include variables that do not affect the running variable
- For DiD: Include variables that do not affect treatment timing or group assignment

Step 3: Screen available variables
- Which variables are clearly pre-treatment characteristics?
- Which variables could plausibly affect the outcome?
- Exclude variables that violate method-specific requirements
- Only consider variables that exist in the available variables list

Step 4: Make final selection
- Include variables that satisfy both general and method-specific criteria
- When uncertain about a variable, do not include it
- Return empty list if no suitable controls can be identified
- Do not create or suggest new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "controls": ["list_of_column_names_or_empty_list"]
}}
"""

ESTIMAND_PROMPT_TEMPLATE = """
You need to determine the appropriate estimand to answer the user query. The estimand defines which population's treatment effect you are estimating.

User Query: {query}
Dataset Description: {dataset_description}
Available Variables: {dataset_columns}
Treatment Variable: {treatment}
Outcome Variable: {outcome}

Think through this step by step:

Step 1: Analyze the query language
- Does the query ask about effects "for everyone" or "on average" (suggests ATE)?
- Does it ask about effects "for those who received treatment" (suggests ATT)?
- Does it ask about effects "for those who didn't receive treatment" (suggests ATC)?
- Does it mention "compliers" or involve instrumental variables (suggests LATE)?
- Does it ask about effects "for specific groups" or conditional on characteristics (suggests CATE)?

Step 2: Consider the policy context
- Is this for policy evaluation affecting the general population (ATE)?
- Is this for understanding impact on actual participants (ATT)?
- Is this for understanding potential impact on non-participants (ATC)?
- Is this in an IV context with compliance issues (LATE)?
- Is this examining heterogeneous effects across subgroups (CATE)?

Step 3: Make final determination
- Choose the estimand that best matches the query's intent and context
- Default to ATE when the population of interest is unclear

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "estimand": "ate_or_att_or_atc_or_late_or_cate"
}}
"""

CONFOUNDER_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify potential confounders to control for when estimating the causal effect to answer the user query. Confounders create bias by affecting both treatment assignment and the outcome.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}
Treatment Variable: {treatment}
Outcome Variable: {outcome}

Think through this step by step:

Step 1: Understand confounding for this analysis
- What factors might influence both who receives '{treatment}' and the level of '{outcome}'?
- These create spurious associations that bias causal estimates

Step 2: Screen for pre-treatment variables
- Which variables are measured before treatment occurs?
- Exclude any variables that could be caused by the treatment
- Only consider variables that exist in the available variables list

Step 3: Evaluate dual causation
- For each pre-treatment variable, ask: Does it plausibly affect treatment assignment?
- For the same variable, ask: Does it plausibly affect the outcome directly?
- Only variables affecting both qualify as confounders

Step 4: Make final determinations
- Include variables with clear causal pathways to both treatment and outcome
- When uncertain about a variable's confounding status, do not include it
- Provide brief reasoning for each identified confounder
- Do not create or suggest new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "confounders": ["list_of_column_names_or_empty_list"],
    "reasoning": {{
        "variable_name": "brief_explanation_of_why_this_affects_both_treatment_and_outcome",
        "another_variable": "..."
    }}
}}
"""

TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a causal inference assistant.
"
Dataset Description: {description}
Identified Treatment Variable: "{treatment_variable}"
Unique Values in Treatment Variable (sample): {treatment_variable_values}

User Query: "{query}

Based on the user query, does it specify a particular category of the treatment variable '{treatment_variable}' that should be considered the control, baseline, or reference group for comparison?

Examples:
- Query: "Effect of DrugA vs Placebo" -> Reference for treatment "Drug" might be "Placebo"
- Query: "Compare ActiveLearning and StandardMethod against NoIntervention" -> Reference for treatment "TeachingMethod" might be "NoIntervention"

If a reference level is clearly specified or strongly implied AND it is one of the unique values provided for the treatment variable, identify it. Otherwise, state null.
If multiple values seem like controls (e.g. "compare A and B vs C and D"), return null for now, as this requires more complex handling.

Respond ONLY with a JSON object adhering to this Pydantic model:
{{
    "reference_level": "string_representing_the_level_or_null",
    "reasoning": "string_or_null_brief_explanation"
}}
"""


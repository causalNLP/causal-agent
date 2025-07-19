"""
Prompt templates for identifying specific causal structures (IV, RDD, RCT)
within the query_interpreter component.
"""

# Note: These templates expect f-string formatting with variables like:
# query, description, column_info, treatment, outcome

## TODO: Test is do we need to provide all this information to the LLM or we simply ask find the instrument?
IV_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a causal inference assistant tasked with assessing whether a valid Instrumental Variable (IV) exists in the dataset. A valid IV must satisfy **all** of the following conditions:

1. **Relevance**: It must causally influence the Treatment.
2. **Exclusion Restriction**: It must affect the Outcome only through the Treatment — not directly or indirectly via other paths.
3. **Independence**: It must be as good as randomly assigned with respect to any unobserved confounders affecting the Outcome.
4. **Compliance (for RCTs)**: If the dataset comes from a randomized controlled trial or experiment, IVs are only valid if compliance data is available — i.e., if some units did not follow their assigned treatment. In this case, the random assignment may be a valid IV, and compliance is the actual treatment variable. If compliance related variable is not available, do not select IV.
5. The instrument must be one of the listed dataset columns (not the treatment itself), and must not be assumed or invented.

You should **only suggest an IV if you are confident that all the conditions are satisfied**. Otherwise, return "NULL".

Here is the information about the user query and the dataset:

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}

Available Columns:
{column_info}

Return a JSON object with the structure:
{{ "instrument_variable": "COLUMN_NAME_OR_NULL" }}
"""

DID_TERM_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a causal inference assistant tasked with determining whether a valid Difference-in-Differences (DiD) **interaction term** already exists in the dataset.

This DiD term should be a **binary variable** indicating whether a unit belongs to the **treatment group after treatment was applied**.

For example, if a policy was enacted in 2020 for a particular state, then the DiD term would equal 1 for units from that state in years after 2020, and 0 otherwise.

Here is the information:

User Query: "{query}"

Time variable: {time_variable}
Group variable: {group_variable}

Dataset Description:
{description}

Available Columns:
{column_info}

Column Types:
{column_types}

Return your answer as a valid JSON object with the following format:
{{ "did_term": "COLUMN_NAME_OR_NULL" }}
"""




RDD_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert causal inference assistant helping to determine if Regression Discontinuity Design (RDD) is applicable for quasi-experimental analysis. 
Here is the information about the user query and the dataset:

User Query: "{query}"
Dataset Description: {description}
Identified Treatment (tentative): {treatment}
Identified Outcome (tentative): {outcome}

Available Columns:
{column_info}

Your goal is to check if there is 'Running Variable' i.e. a variable that determines treatment/treatment control. If the variable is above a certain cutoff, the unit is categorized as treat; if below, it is control.
The running variable must be numeric and continuous. Do not use categorical or low-cardinality variables. Additionally, the treatment variable must be binary in this case. If not, RDD is not valid.

Respond ONLY with a valid JSON object matching the required schema. If RDD is not suggested by the context, return null for both fields.
Schema: {{ "running_variable": "COLUMN_NAME_OR_NULL", "cutoff_value": NUMERIC_VALUE_OR_NULL }}
Example: {{ "running_variable": "test_score", "cutoff_value": 70 }} or {{ "running_variable": null, "cutoff_value": null }}
"""

RCT_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert causal inference assistant helping to determine if the data comes from a Randomized Controlled Trial (RCT).
Your goal is to assess if the treatment assignment mechanism described or implied was random. 

Here is the information about the user query and the dataset:

User Query: "{query}"
Dataset Description: {description}
Identified Treatment (tentative): {treatment}
Identified Outcome (tentative): {outcome}

Available Columns:
{column_info}

Based on the above information, determine if the data comes a randmomized experiment / radomized controlled trial.

Respond ONLY with a valid JSON object matching the required schema. Respond with true if RCT is likely, false if observational is likely, and null if unsure.
Schema: {{ "is_rct": BOOLEAN_OR_NULL }}
Example (RCT likely): {{ "is_rct": true }}
Example (Observational likely): {{ "is_rct": false }}
Example (Unsure): {{ "is_rct": null }}
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

INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a causal inference assistant.

Your task is to determine whether the user query suggests the inclusion of an interaction term between the treatment and one covariate, specifically to assess heterogeneous treatment effects (HTE).

User Query:
"{query}"

Dataset Description:
"{description}"

Identified Treatment Variable:
"{treatment_variable}"

Available Covariates (name: type):
{covariates_list_with_types}

Instructions:
- ONLY suggest an interaction if the query explicitly mentions treatment across a subgroup.
- DO NOT suggest an interaction if the query asks for an overall average effect or does not mention subgroup analysis.
- If you're unsure, default to no interaction.

Respond ONLY with a JSON object that follows this schema:

{{
    "interaction_needed": boolean,             // True if subgroup comparison is clearly mentioned
    "interaction_variable": string_or_null,    // Name of covariate to interact with treatment, or null
    "reasoning": string                        // Short explanation
}}
Example (interaction suggested):
{{
    "interaction_needed": true,
    "interaction_variable": "gender",
    "reasoning": "Query asks if the treatment effect if for men."
}}

Example (no interaction suggested):
{{
    "interaction_needed": false,
    "interaction_variable": null,
    "reasoning": "Query asks for the overall average treatment effect, no specific subgroups mentioned for effect heterogeneity."
}}
""" 


## This prompt is used to identify the treatment variable.
TREATMENT_VAR_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to identify the **treatment variable** in a dataset in order to perform a causal analysis that answers the user's query.

User Query:
{query}

Dataset Description:
{description}

List of Available Variables:
{column_info}

Based on the query, dataset description, and available variables, determine which variable is most likely to serve as the treatment variable. 

If a clear treatment variable cannot be determined from the provided information, return null.

Return your response as a valid JSON object in the following format:
{{ "treatment": "COLUMN_NAME_OR_NULL" }}
"""

## This prompt is used to identify the outcome variable. 
OUTCOME_VAR_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to identify the **outcome variable** in a dataset in order to perform a causal analysis that answers the user's query.

User Query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

Based on the query, dataset description, and available variables, determine which variable is most likely to serve as the outcome variable in the causal analysis.

Do not speculate. If a clear outcome variable cannot be identified from the provided information, return null.

Return your response as a valid JSON object in the following format:
{{ "outcome": "COLUMN_NAME_OR_NULL" }}
"""

COVARIATES_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to identify the **pre-treatment variables** in a dataset that can be used as controls in a causal estimation model to answer the user's query.

User Query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

The treatment variable is: {treatment}  
The outcome variable is: {outcome}

Pre-treatment variables are those that are measured **before** the treatment is applied and are **not affected** by the treatment. These variables can be used as controls in the causal model.
For example, say we have an RCT with outcome Y, treatment T, and pre-treatment variables X1, X2, and X3. We can perform a regression of the form: Y ~ T + X1 + X2 + X3. 

Based on the information above, return a list of variables that qualify as pre-treatment variables from the available columns.
If no suitable pre-treatment variables can be identified, return an empty list.

Return your response as a valid JSON object in the following format:
{{ "covariates": ["LIST_OF_COLUMN_NAMES_OR_EMPTY_LIST"] }}
"""


CAUSAL_GRAPH_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to construct a causal graph to help answer a user query.

Here is the user query:
{query}

Dataset Description:
{description}

Here are the treatment and outcome variables:
Treatment: {treatment}
Outcome: {outcome}

Here are the available variables in the dataset:
{column_names}

Based on the query, dataset description, and available variables, list the most relevant direct causal relationships in the dataset. 
Return them as a list in the format "A -> B", where A is the cause and B is the effect. Use only variables present in the dataset. Do not invent or assume any variables. 
Return the result as a Python list of strings, like:
["A -> B", "B -> C", "A -> C"]
"""


ESTIMAND_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to determine the appropriate estimand to answer a given query. 

Here is the user query:
{query}

Additionally, the dataset has the following description:
{dataset_description}

Here are the variables in the dataset:
{dataset_columns}

Likewise, the treatment variable is: {treatment}, and the outcome variable is: {outcome}.

Given this information, decide whether the Average Treatment Effect (ATE) or the Average Treatment Effect on the Treated (ATT) is more appropriate for answering the query.
Only return the estimand name: "att" or "ate"
"""


MEDIATOR_PROMPT_TEMPLATE = """
You are an expert in causal inference. The user is interested in estimating the effect of {treatment} on {outcome}.

Here is the dataset description:
{description}

Taking into account the treatment, outcome, and the description, from the following variables, is there a valid mediator (i.e., affected by {treatment} and affecting {outcome})?
{column_names}

*** This should be a valid mediator. If there is no valid mediator, return "None." ***
Return a single variable name if applicable. If none, return "None."
"""

CONFOUNDER_PROMPT_TEMPLATE = """
You are an expert in causal inference.

The user is interested in estimating the effect of {treatment} on {outcome}.
Here is the dataset description:
{description}

List 3 to 5 variables from the following that are likely confounders (i.e., affect both {treatment} and {outcome}):
{column_names}

Return only a comma-separated list of variable names.
"""


CONFOUNDER_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to identify potential **confounders** in a dataset that should be adjusted for when estimating the causal effect described in the user query.

User Query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

The treatment variable is: {treatment}  
The outcome variable is: {outcome}

A **confounder** is a variable that:
1. **Affects the treatment** (i.e., influences who receives the treatment), and
2. **Affects the outcome**, and
3. **Is not caused by the treatment** (i.e., it must be a pre-treatment variable),
4. Is **not a mediator** between treatment and outcome.

These variables can create spurious associations between treatment and outcome if not adjusted for.

Based on the user query and the dataset description, identify which variables are likely to be confounders. Only include variables that you believe causally affect both treatment and outcome. If you're uncertain, only include variables where the justification is clear from the query or description.

Return your response as a valid JSON object in the following format:
{{ "confounders": ["LIST_OF_COLUMN_NAMES_OR_EMPTY_LIST"] }}
"""



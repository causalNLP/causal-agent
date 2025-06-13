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
User Query: "{query}"
Dataset Description: "{description}"
Identified Treatment Variable: "{treatment_variable}"
Available Covariates (name: type):
{covariates_list_with_types}

Based on the User Query, Dataset Description, identified Treatment Variable, and Available Covariates:
1. Does the query or context strongly suggest that the effect of '{treatment_variable}' on the outcome might *differ* based on the levels/values of any specific covariate listed above? (e.g., "does the drug work differently for men vs. women?", "is the program more effective for younger participants?", "check for heterogeneous effects by region").
2. If yes, identify the single most plausible covariate from the 'Available Covariates' that is suggested to interact with '{treatment_variable}'.
3. Only suggest an interaction if it's clearly implied for exploring heterogeneous treatment effects. Do not suggest interactions for general model improvement if not thematically implied by the query.

Respond ONLY with a JSON object adhering to this Pydantic model. If no interaction is clearly suggested, 'interaction_needed' should be false, and 'interaction_variable' should be null.
Schema:
{{
    "interaction_needed": boolean, /* True if an interaction is strongly suggested, False otherwise */
    "interaction_variable": "string_or_null", /* The name of the single covariate to interact with treatment, or null */
    "reasoning": "string_or_null" /* Brief reasoning for your decision */
}}

Example (interaction suggested):
{{
    "interaction_needed": true,
    "interaction_variable": "gender",
    "reasoning": "Query asks if the treatment effect differs between men and women."
}}

Example (no interaction suggested):
{{
    "interaction_needed": false,
    "interaction_variable": null,
    "reasoning": "Query asks for the overall average treatment effect, no specific subgroups mentioned for effect heterogeneity."
}}
""" 
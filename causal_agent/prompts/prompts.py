## This prompt is used to identify whether a dataset comes from a randomized trial or not. 
RCT_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether an input data comes from a **randomized controlled trial (RCT) / randomized experiment **.

Here is the dataset description:
{description}

Here are the variables included in the dataset:
{column_info}

Based solely on the dataset description, assess whether the treatment was randomly assigned (i.e., whether the data comes from an RCT).
RCTs are characterized by random assignment of treatment across the participating units.

Do not speculate. If the description does not provide enough information to decide, return null.

Return your response as a valid JSON object in the following format:
{{ "is_rct": BOOLEAN_OR_NULL }}

Examples:
- RCT likely → {{ "is_rct": true }}
- Observational likely → {{ "is_rct": false }}
- Unclear or not enough information → {{ "is_rct": null }}
"""

## This prompt is used to identify the outcome variable. 
OUTCOME_VAR_IDENTIFICATION_PROMPT = """
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
{{ "outcome_variable": "COLUMN_NAME_OR_NULL" }}
"""

## This prompt is used to identify the treatment variable.
TREATMENT_VAR_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to identify the **treatment variable** in a dataset in order to perform a causal analysis that answers the user's query.

User Query:
{query}

Dataset Description:
{description}

List of Available Variables:
{column_info}

Based on the query, dataset description, and available variables, determine which variable is most likely to serve as the treatment in the causal analysis.

If a clear treatment variable cannot be determined from the provided information, return null.

Return your response as a valid JSON object in the following format:
{{ "treatment_variable": "COLUMN_NAME_OR_NULL" }}
"""


## This prompt is used to identify whether the dataset comes from an encouragement design, which is a type of randomized experiment where individuals are 
## randomly encouraged to take a treatment, but not all who are encouraged actually comply. For instance, we could randomly selected inviduals and encourage them to 
## take a vaccine. However, we cannot guarantee that all individuals who were encouraged actually took the vaccine. In such case, the mechanism is describe as 
## Z -> T -> Y, where Z is the encouragement variable, T is the treatment variable, and Y is the outcome variable.
ENCOURAGEMENT_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether the dataset follows an encouragement design.

Here is the dataset description:
{description}

Here are the variables included in the dataset:
{column_info}
Recall that an encouragement design is a type of experiment where individuals are randomly encouraged to take a treatment, but not all who are encouraged actually comply. 

To identify such a design, the dataset must include both:
1. A variable indicating whether a unit was encouraged (randomized assignment), and
2. A variable indicating whether the unit actually received the treatment.

If either of these variables is missing, or the description is insufficient, you should return null.

Do not speculate. Base your decision strictly on the provided information.

Return your response as a valid JSON object in the following format:
{{ "is_encouragement": BOOLEAN_OR_NULL }}

Examples:
- Encouragement design likely → {{ "is_encouragement": true }}
- Not an encouragement design → {{ "is_encouragement": false }}
- Unclear or insufficient information → {{ "is_encouragement": null }}
"""

## This prompt is used to identify the encouragement variable and the treatment variable in an encouragement design.
ENCOURAGEMENT_VAR_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to identify variables for performing an encouragement design (Instrumental Variable) analysis to answer the user’s query.

User Query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

Based on the query, dataset description, and listed variables, identify:

1. The **encouragement variable** — a randomized variable indicating whether a unit was encouraged to take the treatment.
2. The **treatment variable** — indicating whether the unit actually took the the treatment.

Do not speculate. If either the encouragement or the treatment variable cannot be clearly identified from the information provided, return null for the respective field.

Return your response as a valid JSON object in the following format:
{{ "encouragement_variable": "COLUMN_NAME_OR_NULL", "treatment_variable": "COLUMN_NAME_OR_NULL" }}
"""

## This prompt is used to identify pre-treatment variables that can be used as control in a causal estimation model. 
PRE_TREAT_VAR_IDENTIFICATION_PROMPT = """
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
{{ "pre_treat_variables": ["LIST_OF_COLUMN_NAMES_OR_EMPTY_LIST"] }}
"""

## This prompt is used to identify whether a Difference-in-Differences (DiD) analysis is appropriate for the dataset in relation to the user query.
DiD_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether a difference-in-differences (DiD) analysis is appropriate for analyzing the given dataset to answer a user query.

Here is the user query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

Recall that, DiD is used to estimate causal effects by comparing outcome changes over time between treated and control groups.This requires information on outcomes both before and after treatment.
There are two common types of DiD designs:
1. **Canonical DiD**: Two groups (treated and control) and two time periods (pre-treatment and post-treatment).
2. **Staggered DiD**: Multiple groups and multiple time periods, with treatment staggered across groups over time.

Based on the provided information, first determine whether DiD is applicable. If it is, indicate whether the design is canonical or staggered.

Do not speculate. If the information is insufficient to make a determination, return null values.

Return your response as a valid JSON object in the following format:
{{ 
  "is_did_applicable": BOOLEAN_OR_NULL, 
  "is_canonical_did": BOOLEAN_OR_NULL 
}}
"""

## This prompt is used to identify the temporal variable necessary for performing a Difference-in-Differences (DiD) analysis.
## The temporal variable must indicate when the observation was recorded or it could be used to construct a post-treatment indicator.

TEMPORAL_VAR_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether the dataset contains **temporal information relevant to treatment timing**, 
which is necessary to perform a Difference-in-Differences (DiD) analysis to answer a user's query.

User Query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

To apply a DiD analysis, the dataset must allow comparison of outcomes **before and after treatment**. 
This requires temporal variables that help determine **when the treatment occurred** and **when each observation was recorded**. There are three possible cases:

1. **Post-treatment indicator is directly available**:
   - A binary variable (e.g., `post_treatment`) indicates whether the observation occurred after the treatment.

2. **Post-treatment indicator can be constructed using a reference value, which represents the time when the treatment was applied**:
   - The dataset contains a **time variable** (e.g., `year`, `date`) indicating when each observation occurred.
   - A single **treatment time** (not a column) can be inferred from the description or query.
   - From this, a post-treatment indicator can be constructed: `post = 1{year ≥ treatment_time}`.

3. **Treatment is staggered across units i.e. this is a two-way fixed effects model**:
   - The dataset includes a **time variable** indicating when each observation occurred.
   - From these, we can construct post indicators or event time variables, such 

Only identify these variables if they are relevant for conducting a DiD-style analysis. Do **not** select time-related variables that are unrelated to treatment timing or if the query does not support a before-after causal comparison.

Based on the query, dataset description, and available variables, return:

- "post_treatment_variable": Name of the binary post-treatment indicator, if it exists.
- "time_variable": Name of the variable indicating when the observation occurred (e.g., `year`, `date`).
- "treatment_reference_time": A single reference period (not a column) when treatment was introduced, if inferable. Note that this is useful for canonical DiD analysis with two groups and two period: pre-and post-treatment. For stagged DiD, this is not needed. Return NULL. 

If any of these cannot be identified, return `null` for that field.

Return your response as a valid JSON object in the following format:
{{ 
  "post_treatment_variable": "COLUMN_NAME_OR_NULL", 
  "time_variable": "COLUMN_NAME_OR_NULL", 
  "treatment_reference_time": YEAR_OR_NULL 
}}
"""

## This prompt is used to identify the group variable necessary for performing a Difference-in-Differences (DiD) analysis.
## The group variable must indicate the treatment and control groups, or it could be a categorical variable in case of staggered DiD.
STATE_VAR_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether the dataset contains a ** group variable*** necessary to perform a Difference-in-Differences (DiD) analysis to answer a user's query.

Here is the user query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

To apply a DiD analysis, the dataset must allow comparison of outcomes between different groups (treatment vs control) over time. This requires a group variable that represents entites that are either treated or not treated.
There are two possible cases:

1. **Group variable is a binary indicator**:
   - The data contains a binary variable indicating whether the observation belongs to the treatment or control group. 
2. **Group variable is categorical**:
   - The data contains a categorical variable indicating different groups. 

Based on the query, dataset description, and available variables, return:
- "group_variable": Name of the group variable that indicates the treatment and control groups or a categorical variable representing different groups in case of staggered DiD.
- "group_reference": A single reference group (not a column) that corresponds to the treatment group. This is used in canonical DiD only. For example, say a policy was enacted in New Jersey, but not in other states. 
The reference group would be "New Jersey" or "NJ", since the policy was enacted there. We can contruct, TREAT variable as `TREAT = 1 if state == "New Jersey" else 0`.

If a suitable group variable cannot be identified, return null for the "group_variable" field.
Return your response as a valid JSON object in the following format:
{{ 
  "group_variable": "COLUMN_NAME_OR_NULL", 
  "group_reference": "GROUP_NAME_OR_NULL" 
}}
"""

## This prompt is used to identify whether a Regression Discontinuity Design (RDD) is appropriate for the dataset in relation to the user query.
## The goal is to identify the running variable, cutoff value, and treatment variable for RDD analysis.

RDD_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether a Regression Discontinuity Design (RDD) is appropriate for analyzing the dataset in relation to the user query.

Here is the user query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

Recall that RDD requires a numeric, non-binary running variable that determines treatment assignment based on a cutoff—i.e., treatment = 1 if value > cutoff. 
In many cases, the treatment variable may already be included in the dataset and does not need to be computed. In such cases, return the name of the treatment variable as well.

Based on the query, dataset description, and available variables, return:
- "running_variable": Name of the running variable that determines treatment assignment based on a cutoff.
- "cutoff_value": The numeric cutoff value that determines treatment assignment.
- "treatment_variable": Name of the treatment variable indicating whether the unit received treatment (1) or control (0).

Do not speculate. If the information is insufficient, return null for the relevant fields.

Return your response as a valid JSON object in the following format:
{{ 
  "running_variable": "COLUMN_NAME_OR_NULL", 
  "cutoff_value": NUMERIC_VALUE_OR_NULL, 
  "treatment_variable": "COLUMN_NAME_OR_NULL" 
}}
"""

## This prompt is used to identify whether an Instrumental Variable (IV) is appropriate for the dataset in relation to the user query.
## The output includes the instrument variable and the treatment variable that the instrument influences. We can check whether the treatment variable selected here is 
## the same as the treatment variable selected by the TREATMENT_VAR_IDENTIFICATION_PROMPT.

INSTRUMENT_VAR_IDENTIFICATION_PROMPT = """
You are an expert in causal inference. Your task is to determine whether an Instrumental Variable (IV) is appropriate for analyzing the dataset in relation to the user query.

Here is the user query:
{query}

Dataset Description:
{description}

Available Variables:
{column_info}

Recall that a valid instrument must satisfy all of the following conditions:
1. **Relevance**: It must causally influence the treatment variable.
2. **Exclusion Restriction**: It must affect the outcome only through the treatment—not directly or through other pathways.
3. **Independence**: It must be as good as randomly assigned, independent of unobserved confounders affecting the outcome.

The instrument must be one of the variables listed in the dataset (not the treatment itself), and must not be assumed or invented.

Based on the query, dataset description, and available variables, return:
- "instrument_variable": Name of the variable that can serve as a valid instrument.
- "treatment_variable": Name of the treatment variable that the instrument influences.

Do not speculate. Only suggest an IV if you are confident that all conditions are satisfied. Otherwise, return null.

Return your response as a valid JSON object in the following format:
{{ 
  "instrument_variable": "COLUMN_NAME_OR_NULL", 
  "treatment_variable": "COLUMN_NAME_OR_NULL" 
}}
"""

## This prompt is used to construct a causal graph based on the user query, dataset description, treatment, outcome, and available variables.
## Once the graph is constructed, we can use dowhy to identify frontdoor, backdoor adjustment sets. 

CAUSAL_GRAPH_PROMPT = """
You are an expert in causal inference. Your task is to construct a causal graph to help answer a user query.

Here is the user query:
{query}

Dataset Description:
{description}

Here are the treatment and outcome variables:
Treatment: {treatment}
Outcome: {outcome}

Here are the available variables in the dataset:
{column_info}

Based on the query, dataset description, and available variables, construct a causal graph that captures the relationships between the treatment, outcome, and other relevant variables.

Use only variables present in the dataset. Do not invent or assume any variables. However, not all variables need to be included—only those that are relevant to the causal relationships should appear in the graph.

Return the causal graph in DOT format. The DOT format should include:
- Nodes for each included variable.
- Directed edges representing causal relationships among variables.

Also return the list of edges in the format "A -> B", where A and B are variable names.

Here is an example of the DOT format:
digraph G {
    A -> B;
    B -> C;
    A -> C;
}

And the corresponding list of edges:
["A -> B", "B -> C", "A -> C"]

Return your response as a valid JSON object in the following format:
{{ 
  "causal_graph": "DOT_FORMAT_STRING",
  "edges": ["EDGE_1", "EDGE_2", ...] 
}}
"""

## This prompt is used to determine whether an interaction term between the treatment variable and any covariate is needed in the causal model 
## In case, we need to look at the coefficient of the interaction term to answer the user query, set the boolean "interaction_term_query" to true.
## If the interaction term is not needed, but its inclusion may still be statistically or substantively justified, set "interaction_term_query" to true. In this case, 
## we include the itneraction term in the mode, but we do not need to look at the coefficient of the interaction term. We use the coefficient of the treatment variable

INTERACTION_TERM_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. Your task is to determine whether an interaction term between the treatment variable and any covariate is needed in the causal model to answer the user query.

Here is the user query:
{query}

Dataset Description:
{description}

Identified Treatment Variable: "{treatment_variable}"
Available Covariates (with types): "{covariates_list_with_types}"
Outcome Variable: "{outcome_variable}"

Recall that an interaction term is needed when:
- To answer the query, we need to examine the **coefficient of the interaction term**, in which case the interaction is strictly necessary.
- The goal is to estimate **heterogeneous treatment effects** — i.e., whether the effect of the treatment on the outcome differs based on the level or value of a covariate.
- In other cases, the interaction may not be required to answer the query, but its inclusion may still be statistically or substantively justified.

Based on the information above, determine whether an interaction term between the treatment and any covariate is needed to answer the user query.

Return your response as a valid JSON object in the following format:
{{ 
  "interaction_needed": BOOLEAN,               // True if an interaction term is needed or beneficial
  "interaction_variable": "COLUMN_NAME_OR_NULL", // Covariate to interact with treatment, or null if not needed
  "reasoning": "REASONING_STRING",             // Brief explanation of your decision
  "interaction_term_query": BOOLEAN            // True if the interaction is essential to answering the query
}}
"""

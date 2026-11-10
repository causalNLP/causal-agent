"""
Prompt templates for identifying specific causal structures (IV, RDD, RCT)
within the query_interpreter component.
"""

# Note: These templates expect f-string formatting with variables like:
# query, description, column_info, treatment, outcome

IV_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference.
Your task is to NOMINATE instrumental-variable candidates: columns that plausibly
Determine who receives the treatment -- or how much treatment they receive -- based
on the study design described in the text.

This is a nomination task, not a validity verdict. Do NOT try to certify untestable
conditions (exclusion restriction, independence): they are argued separately, and
the program verifies instrument strength statistically from the data and discards
weak candidates. Nominating a plausible design column is helpful; withholding it
for untestable reasons is not.

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 0: Classify the uptake regime FIRST -- this decides whether an instrument
can exist at all.
- DRIVEN: the description says some factor determined or drove who received
  treatment -- an assignment or eligibility rule, an offer or invitation, a
  provider or system decision, funding, a lottery -- OR receipt is described as
  tied to a factor OUTSIDE the unit's own control: a provider attribute
  (experience, staffing), a system or area condition (local unemployment or
  prices, budgets, policies, exposure rules). Uptake need not be formally
  assigned; what matters is an outside driver of who gets treated.
- VOLUNTARY/SELF-SELECTED: units effectively chose their own uptake ('chose to
  participate', 'opted in', 'enrolled themselves', uptake described as a personal
  decision) and no outside driver is described. The unit's own characteristics
  (its income, residence, health, history) are NOT outside drivers -- described
  differences between participants and non-participants are exactly the
  confounding that matching or regression adjusts for, not an instrument.
- If VOLUNTARY/SELF-SELECTED, return null for ALL fields. An instrument requires
  an outside driver of receipt; in a self-selected design there is none, and
  adjustment methods (matching, regression) are the right tools. Abstaining here
  is the correct answer, not a failure.

Step 1: Understand the causal question
- What causal effect is the user trying to estimate?
- What might confound treatment and outcome? (Context only; not your job to fix here.)

Step 2: Nominate candidates (only if the regime is ASSIGNED/ENCOURAGED)
- Find columns that plausibly determine treatment assignment or uptake.
- Patterns that suggest a candidate: assignment-like flags ('assigned', 'offered',
  'invited', 'enrolled', 'push'); design determinants the description names
  (eligibility rules, income thresholds, unemployment rates, program funding,
  provider experience, lottery components).
- NEVER nominate: the treatment or outcome columns themselves; time or period
  flags (columns marking pre/post periods such as 'post_intervention',
  'post_policy_period', year or wave dummies) -- they occur at or after treatment
  time and cannot determine assignment.
- Only nominate variables that exist in the available columns list.

Step 3: Ground each candidate
- Quote the sentence from the dataset description that led you to the candidate,
  verbatim, or null if no sentence supports it.
- Set 'basis' to 'design-stated' when the quoted sentence ties the candidate to
  the treatment decision; otherwise 'name-only' (trusted for its meaning, to be
  verified by the program from data).

Step 4: Decide 'instrument_variable'
- Return the best candidate in 'instrument_variable'; list up to three in 'candidates'.
- Only select variables from the available columns; do not create new variables.
- If the regime is voluntary or nothing in the design plausibly determines
  treatment, return null for all fields rather than guessing from unrelated columns.

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "instrument_variable": "best_candidate_or_null",
    "candidates": ["up_to_three_candidates_or_null"],
    "basis": "design-stated_or_name-only_or_null",
    "quote": "verbatim_sentence_or_null"
}}
"""

RDD_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. 
You need to identify if a running variable exists for performing Regression Discontinuity Design (RDD) to answer the user query. 
Go through the data description and available columns carefully. You need to be strict with the assessment.
In RDD, treatment assignment (for analysis) is determined by whether a continuous variable crosses a specific threshold.
Whether the treatment is assigned above or below the cutoff can be determined by the user query or dataset description.

User Query: "{query}"
Dataset Description: {description}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 1: Analyze the research design
- Does the query or description imply a cutoff, threshold, or eligibility criterion?
- Check if the design implies that a cutoff/threshold determines if the unit receives the intervention or not.

Step 2: If there is a running variable implied by the context, identify the running variable:
- What is the variable associated with the cutoff, i.e., what variable determines whether a unit is receiving the intervention/treatment or not?

Step 3: Identify the cutoff value from design
- What specific threshold value determines treatment assignment?

Step 4: Identify if the treatment is assigned above or below the cutoff
- Is the treatment assigned above the cutoff? Store it as a boolean value in 'treat_above_cutoff' in the JSON.
- Store null if you are unsure.

Step 5: Final determination
- Only suggest RDD if both running variable and cutoff value can be identified
- Return null if the assignment mechanism is not threshold-based or you are unsure.

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "running_variable": "column_name_or_null",
    "cutoff_value": numeric_value_or_null,
    "treat_above_cutoff": true_false_or_null,
}}
"""

RCT_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. 
You need to determine if the data comes from a Randomized Controlled Trial (RCT). 
Based on the assessment, we will decide if RCT methods are appropriate to answer the user query.

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}
Available Columns: {column_info}

Think through this step by step:

Step 1: Look for randomization indicators
- Does the description mention "random", "randomized", "randomly assigned", "control group", or "trial"?

Step 2: Assess assignment method
- Was treatment assigned randomly?

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
You are an expert in causal inference. 
You need to determine if an interaction term is required in the causal model built to answer the user query. 
Be strict with your assessment. 

User Query: "{query}"
Dataset Description: "{description}"
Treatment Variable: "{treatment_variable}"
Available Covariates: {covariates_list_with_types}

Think through this step by step:

Step 1: Distinguish between subgroup analysis and overall effects
- Is the query asking for differential effects across groups (interaction needed)?
- Or is it asking for overall average effects (no interaction needed)?

Step 2: Analyze the query and see if it implies subgroup effects
- Does the query explicitly ask about treatment effects for specific subgroups?

Step 3: Identify the relevant interaction covariate
- Which covariate from the available list would define the subgroups mentioned?
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
You are an expert in causal inference. 
You need to identify the treatment variable to perform causal analysis that answers the user query. 
The treatment variable corresponds to the variable representing the treatment or intervention of interest. 

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}

Think through this step by step:

Step 1: Understand the causal question
- What is the specific treatment whose causal effect needs to be estimated, i.e., what is the intervention of interest?

Step 2: Distinguish between assignment and actual treatment
- For experimental data: Is this an encouragement design where assignment differs from actual uptake?
- Look for variables representing what units actually received vs what they were assigned
- In encouragement designs, choose the actual treatment received, not the random assignment.

Step 3: Identify the treatment variable
- Which column represents the actual intervention or treatment of interest?
- The variable should represent the actual treatment received. 
- Only select variables that exist in the available variables list

Step 4: Make final determination
- Return the variable representing actual treatment received
- Return null if no clear treatment variable can be identified from the information provided

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "treatment": "column_name_or_null"
}}
"""

OUTCOME_VAR_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. 
You need to identify the outcome variable to perform causal analysis that answers the user query. 
The outcome variable corresponds to the variable representing the effect of an intervention or treatment of interest.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}

Think through this step by step:

Step 1: Identify what the query is measuring
- What outcome or effect is the user asking about?

Step 2: Find the related variable
- Which column in the dataset corresponds to this outcome?
- Only select variables that exist in the available variables list

Step 3: Make final determination
- Return the variable that captures the outcome of interest
- Return null if no clear outcome variable can be identified
- Do not create or suggest new variables

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "outcome": "column_name_or_null"
}}
"""

COVARIATES_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference. 
You need to identify the pre-treatment variables in a dataset that can be used as controls in a causal estimation model to answer the user's query.
Be strict with your assessment.

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}
The treatment variable is: {treatment}
The outcome variable is: {outcome}

Pre-treatment variables are those that are measured before the treatment is applied and are not affected by the treatment. 
These variables can be used as controls in the causal model to improve precision. 

Based on the information above, return a list of variables that qualify as pre-treatment variables from the available columns. Only select variables that exist in the available variables list. 
Do not create or suggest new variables. 

If no suitable pre-treatment variables can be identified, return an empty list.

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "covariates": ["list_of_column_names_or_empty_list"]
}}
"""

CONTROLS_IDENTIFICATION_PROMPT_TEMPLATE = """
You need to identify control variables to improve the causal estimation for answering the user query. 
Control variables are pre-treatment variables that can reduce bias and improve precision. 
Please do not consider confounders. 

Here are the basic information. 
User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}

Likewise, we have selected {method} as the causal causal inference method to perform the analysis. Similarly, 
the identified treatment and outcome variables are as follows:
Treatment Variable: {treatment}
Outcome Variable: {outcome}

Think through the following step to select the appropriate control variables:

Step 1: Understand control variable requirements
- Controls must be measured before treatment occurs
- Controls should not be affected by the treatment (pre-treatment variables only)

Step 2: Apply method-specific criteria
- For RCT: Include pre-treatment variables to improve precision
- For IV: Include variables that affect the outcome but not the treatment or instrument
- For RDD: Include variables that do not affect the running variable
- For DiD: Include variables that do not affect treatment timing or group assignment

Step 3: Make final selection
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
You are an expert in causal inference. 
You need to determine the appropriate estimand to answer the user query. 
The estimand defines which population's treatment effect you are estimating.

User Query: {query}
Dataset Description: {dataset_description}
Available Variables: {dataset_columns}
Treatment Variable: {treatment}
Outcome Variable: {outcome}

Think through this step by step:

Step 1: Analyze the query language
- Does the query imply effects "for everyone" or "on average" (suggests ATE)?
- Does it ask about effects "for those who received treatment" (suggests ATT)?
- Does it ask about effects "for those who didn't receive treatment" (suggests ATC)?
- Does it mention "compliers" or involve instrumental variables (suggests LATE)?
- Does it ask about effects "for specific groups" or conditional on characteristics (suggests CATE)?

Step 2: Consider the context
- Are we interested in evaluating the effect on the general population (ATE)?
- Are we interested in the effect on actual participants (ATT)?
- Are we interested in understanding the impact on non-participants (ATC)?
- Are we in an IV context with compliance issues (LATE)?
- Are we examining heterogeneous effects across subgroups (CATE)?

Step 3: Make final determination
- Choose the estimand that best matches the query's intent and context
- Default to ATE when the population of interest is unclear

Important: Return only valid JSON. No explanations, reasoning, or markdown formatting.

{{
    "estimand": "ate_or_att_or_atc_or_late_or_cate"
}}
"""

CONFOUNDER_IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert in causal inference.
You need to identify potential confounders to control for when estimating the causal effect to answer the user query. 
Confounders are those variables that influence both the treatment and the outcome. This especially happens in observational studies. Be strict with your assessment.
For RCTs, we do not need to control for confounders as randomization takes care of that. 

User Query: {query}
Dataset Description: {description}
Available Variables: {column_info}
Treatment Variable: {treatment}
Outcome Variable: {outcome}

Think through this step by step:

Step 1: Understand confounding 
- What factors might influence both treatment: '{treatment}' and the outcome: '{outcome}'?

Step 2: Exclusions
- Exclude any variables that could be caused by the treatment
- Only consider variables that exist in the available variables list. Do not create new variables.

Step 3: Assess influence on both treatment and outcome
- Which variables plausibly affect the treatment assignment and the outcome?
- Only variables affecting both qualify as confounders. 

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

Dataset Description: {description}
Identified Treatment Variable: "{treatment_variable}"
Unique Values in Treatment Variable (sample): {treatment_variable_values}

User Query: "{query}"

Based on the user query, does it specify a particular category of the treatment variable '{treatment_variable}' that should be considered the control, baseline, or reference group for comparison?

Examples:
- Query: "Effect of DrugA vs Placebo" -> Reference for treatment "Drug" might be "Placebo"
- Query: "Compare ActiveLearning and StandardMethod against NoIntervention" -> Reference for treatment "TeachingMethod" might be "NoIntervention"

If a reference level is clearly specified or strongly implied AND it is one of the unique values provided for the treatment variable, identify it. Otherwise, state null.
If multiple values seem like controls (e.g., "compare A and B vs C and D"), return null for now, as this requires more complex handling.

Respond ONLY with a JSON object adhering to this Pydantic model:
{{
    "reference_level": "string_representing_the_level_or_null",
    "reasoning": "string_or_null_brief_explanation"
}}
"""
DESIGN_CONTEXT_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a causal inference expert.
You need to extract study-design facts from the user query and dataset
description. These facts are semantic (they depend on what the variables
MEAN, not on their statistical properties), so only report what the query or
description explicitly supports. Never guess: if the text does not settle a
field, return null for it.

User Query: "{query}"
Dataset Description: {description}
Treatment: {treatment}
Outcome: {outcome}
Available Columns: {column_info}

Fields to extract:

1. mediator_variable
   A column that plausibly carries the treatment's effect to the outcome
   (a causal pathway T -> M -> Y). It must be an available column and must
   not be the treatment or the outcome itself. Only report it if the query
   or description describes such a mechanism. Return null otherwise.
   Example: "ad exposure" mediating the effect of a marketing campaign on sales.

2. treatment_period_start
   When the treatment/policy begins, in the same units as the time column
   (e.g. a year like 1988). Only if stated or clearly implied by the text.

3. placebo_period_start
   The start of a pre-treatment period during which the outcome should not
   yet respond to the treatment (an anticipation-free placebo window).
   Only if the text states or clearly implies one.

4. is_encouragement_design
   true ONLY if units were randomized into an encouragement or nudge, while
   the actual treatment uptake was left to the units (encouragement differs
   from uptake). false if the design is plain randomization or clearly not
   randomized. null otherwise. True encouragement designs are rare.

Respond ONLY with a JSON object adhering to this Pydantic model:
{{
    "mediator_variable": "column_name_or_null",
    "treatment_period_start": "number_or_null",
    "placebo_period_start": "number_or_null",
    "is_encouragement_design": true_or_false_or_null,
    "reasoning": "brief_justification_or_why_nothing_was_extracted"
}}
"""

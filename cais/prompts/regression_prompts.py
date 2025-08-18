STATSMODELS_PARAMS_IDENTIFICATION_PROMPT_TEMPLATE = """
You are a statistical assistant. Given a list of parameter names from a regression model,
the user's query, and context about the treatment variable, identify the parameter names and their original indices
that are relevant for answering the query or for providing a general overview of the treatment effect.

User Query: "{user_query}"
Treatment variable in formula (Patsy term): "{treatment_patsy_term}"
Original treatment column name: "{treatment_col_name}"
Is treatment multi-level categorical with a reference: {is_multilevel_case}
Reference level (if multi-level): "{reference_level_for_prompt}"

Available Parameter Names (with their original 0-based index):
{indexed_param_names_str}

Instructions:
-Respond with best matching param or params in case multiple matches with their index/s 
-Exclude interaction terms (those containing ':') unless the query *specifically* asks for an interaction effect. This task is focused on main treatment effects.

Respond ONLY with a valid JSON object matching this Pydantic model schema:
{llm_response_schema_json}
""" 
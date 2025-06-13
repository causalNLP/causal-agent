"""
Prompts for interpreting statistical model results in the context of a specific user query.
"""

QUERY_SPECIFIC_INTERPRETATION_PROMPT_TEMPLATE = """
You are an AI assistant. Your task is to analyze the results of a statistical model and extract the specific information that answers the user's query.

User Query: "{user_query}"

Context from Model Execution:
- Treatment Variable: "{treatment_variable}"
- Reference Level for Treatment (if any): "{reference_level}"
- Model Formula: "{formula}"
- Estimated Effects by Treatment Level (compared to reference, if applicable):
{effects_by_level_str}
- Information on Interaction Term (if any):
{interaction_info_str}

Full Model Summary (for additional context if needed, prefer structured 'Estimated Effects' above):
---
{model_summary_text}
---

Instructions:
1.  Carefully read the User Query to understand what specific treatment effect or comparison they are interested in.
2.  Examine the 'Estimated Effects by Treatment Level' to find the statistics (estimate, p-value, confidence interval, std_err) for the treatment level or comparison most relevant to the query.
3.  If the query refers to a specific treatment level (e.g., "Civic Duty" when treatment variable is "treatment" with levels "Control", "Civic Duty", etc.), focus on that level's comparison to the reference.
4.  Determine if the identified effect is statistically significant (p-value < 0.05).
5.  If a significant interaction is noted in 'Information on Interaction Term' and it involves the identified treatment level, briefly state how it modifies the main effect in your interpretation. Do not perform complex calculations; just state the presence and direction if clear.
6.  Construct a concise 'interpretation_summary' that directly answers the User Query using the extracted statistics.
7.  If the query cannot be directly answered (e.g., the specific level isn't in the results, or the query is too abstract for the given data), explain this in 'unanswered_query_reason'.

Respond ONLY with a valid JSON object matching this Pydantic model schema:
{llm_response_schema_json}
""" 
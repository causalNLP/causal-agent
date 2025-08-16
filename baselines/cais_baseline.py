import pandas as pd
import re
import json

from .chatbot import Chatbot
from .query_formats import QueryFormat, CausalQueryFormat,SequentialCausalThinking
from .coderunner import CodeRunner

from typing import Optional


def print_color(text, color):
    """Print text in color"""
    print(f"\033[{color}m{text}\033[0m")


def find_code(reply, language="python"):
    """Find python code in a string `reply`"""
    code_start = reply.find("```{}".format(language))
    if code_start == -1:
        return None
    code_start += len("```python")
    code_end = reply[code_start:].find("```")
    if code_end == -1:
        return None
    code_end += code_start
    code = reply[code_start:code_end]
    return code


class CAISBaseline:
    """A conversational chatbot that has access to a dataset"""

    def __init__(self, chatbot: Chatbot, safe_exec=True, persistent=False, session_timeout=3600, max_retries=5) -> None:
        self.chatbot = chatbot
        self.code_runner = CodeRunner(safe_exec=safe_exec, persistent=persistent, session_timeout=session_timeout)
        self.max_retries = max_retries
        self.persistent = persistent

    def get_final_result(self):
        """Get the final result from the chatbot in a structured JSON format."""
        prompt = """
Please provide a final summary of the analysis in a single, well-formed JSON object. The JSON object should have the following keys. If a field is not applicable, use `null`.

- `method`: The name of the primary causal inference method used (e.g., "Propensity Score Weighting", "Difference-in-Differences").
- `causal_effect`: The estimated causal effect (e.g., ATE, ATT). Provide this as a numerical value.
- `standard_deviation`: The standard deviation of the causal effect estimate, if available.
- `treatment_variable`: The name of the treatment variable.
- `outcome_variable`: The name of the outcome variable.
- `covariates`: A list of covariate variable names used in the model.
- `instrument_variable`: The name of the instrumental variable, if applicable.
- `running_variable`: The name of the running variable for Regression Discontinuity, if applicable.
- `temporal_variable`: The name of the time variable for Difference-in-Differences, if applicable.
- `statistical_test_results`: A summary of key statistical test results, like p-values or confidence intervals.
- `explanation_for_model_choice`: A brief explanation for why the chosen causal method was appropriate for this analysis.
- `regression_equation`: The exact regression equation if a regression model was used.

Please output ONLY the JSON object and nothing else.
"""
        json_reply = self.chatbot.ask(prompt)

        # Clean the response to extract only the JSON part
        try:
            # Find the start and end of the JSON object
            json_start = json_reply.find('{')
            json_end = json_reply.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = json_reply[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not find a valid JSON object in the response.", "raw_response": json_reply}
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON from the response.", "raw_response": json_reply}

    def get_variable_value(self, variable_name):
        """Get the value of a variable in the persistent environment."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.get_variable_value(variable_name)

    def get_defined_variables(self):
        """Get a list of all defined variables in the environment."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.get_defined_variables()

    def start_persistent_session(self):
        """Start a persistent Python session."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.start_persistent_container()

    def stop_persistent_session(self):
        """Stop the persistent Python session."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.stop_persistent_container()

    def is_session_active(self):
        """Check if the persistent session is active."""
        if not self.persistent:
            return False
        return self.code_runner.is_container_running()

    def upload_file(self, local_path, container_path=None):
        """Upload a file from the local machine to the container."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.upload_file(local_path, container_path)

    def download_file(self, container_path, local_path=None):
        """Download a file from the container to the local machine."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.download_file(container_path, local_path)

    def list_files(self, directory='.'):
        """List files in a directory in the container."""
        if not self.persistent:
            return "Error: Not in persistent mode"
        return self.code_runner.list_files(directory)

    def answer(self, query, dataset_path, dataset_description="", qf=CausalQueryFormat, post_steps=False):
        """Answer a causal query using the dataset path (a df)"""
        
        self.chatbot.delete_history()
        
        # Initialize the query format
        query_format = qf(query, dataset_path, dataset_description)
        
        # Check if we're using sequential thinking
        is_sequential = isinstance(query_format, SequentialCausalThinking)
        
        if not is_sequential:
            # Original implementation for non-sequential formats
            queries = query_format.get_query_format()
            
            reply = ""
            for q in queries["pre"]:
                # Pre-analysis queries
                print_color(q, 32)
                reply = self.chatbot.ask(q)
                print_color(reply, 33)
            
            codes = []
            code_outputs = []
            
            # Run code while the model outputs code
            for _ in range(self.max_retries):
                code = find_code(reply)
                if code is None:
                    break

                codes.append(code)

                code_output = self.code_runner.run_code(code)
                code_outputs.append(code_output)

                # Ask the chatbot to analyze the results
                analysis_query = query_format.get_analysis_format(code_output)
                print_color(analysis_query, 32)
                reply = self.chatbot.ask(analysis_query)
                print_color(reply, 33)

            # Post-analysis queries
            if post_steps and "post" in queries:
                for q in queries["post"]:
                    print_color(q, 32)
                    reply = self.chatbot.ask(q)
                    print_color(reply, 33)

            final_result = self.get_final_result()
            chat_history = self.chatbot.conversation_history
            
        else:
            # Sequential thinking implementation
            codes = []
            code_outputs = []
            chat_history = []
            
            # Start with step 1
            current_step = 1
            max_steps = 7
            
            # Track dataframes for each step
            dataframes = {
                "original_df": None,
                "cleaned_df": None,
                "analyzed_df": None,
                "model_df": None,
                "posthoc_df": None
            }
            
            while current_step <= max_steps:
                # Get the prompt for the current step
                step_prompt = query_format.get_query_format()["pre"][current_step - 1]
                
                # Add context from previous steps if we're beyond step 1
                if current_step > 1:
                    # Create a context that includes summaries from previous steps
                    previous_steps_context = "Here's a summary of our previous steps:\n\n"
                    
                    # Add summaries from all previous steps
                    for prev_step in range(1, current_step):
                        summary = query_format.get_step_summary(prev_step)
                        if summary:
                            previous_steps_context += f"STEP {prev_step} SUMMARY:\n{summary}\n\n"
                    
                    # Add information about available resources
                    available_resources = self._identify_available_resources(query_format, current_step)
                    if available_resources:
                        previous_steps_context += "\n" + available_resources + "\n\n"
                    
                    # Append the previous steps context to the current step prompt
                    step_prompt = previous_steps_context + step_prompt
                
                # Ask the LLM
                print_color(f"STEP {current_step}: {step_prompt}", 32)
                reply = self.chatbot.ask(step_prompt)
                print_color(reply, 33)
                
                # Store the step result
                query_format.store_step_result(current_step, {"prompt": step_prompt, "reply": reply})
                
                # Check if there's code to execute
                code = find_code(reply)
                if code:
                    # Execute the code
                    codes.append(code)
                    
                    # Update the code to use the correct dataframe variables based on the step
                    updated_code = self._update_code_for_step(code, current_step, dataframes)
                    
                    # Execute the updated code
                    code_output = self.code_runner.run_code(updated_code)
                    code_outputs.append(code_output)
                    
                    # Extract dataframe variables from the code execution
                    self._extract_dataframes(current_step, dataframes)
                    
                    # Store the code output in the step result
                    query_format.step_results[current_step]["code"] = code
                    query_format.step_results[current_step]["code_output"] = code_output
                    
                    # Update the code context
                    query_format.update_code_context(code, {})
                    
                    # Update available resources based on the step
                    self._update_step_resources(query_format, current_step, code, code_output)
                    
                    # Get step-specific analysis
                    analysis_query = query_format.get_analysis_format(code_output, step=current_step)
                    print_color(analysis_query, 32)
                    analysis_reply = self.chatbot.ask(analysis_query)
                    print_color(analysis_reply, 33)
                    
                    # Store the analysis in the step result
                    query_format.step_results[current_step]["analysis"] = analysis_reply
                    
                    # Extract summary from the analysis reply
                    analysis_summary = query_format.extract_summary(analysis_reply)
                    if analysis_summary and analysis_summary != "No explicit summary provided.":
                        # If we got a good summary from the analysis, use it instead
                        query_format.step_summaries[current_step] = analysis_summary
                    
                    # Check if there's more code in the analysis (for corrections)
                    correction_code = find_code(analysis_reply)
                    if correction_code:
                        # Execute the corrected code
                        codes.append(correction_code)
                        
                        # Update the correction code to use the correct dataframe variables
                        updated_correction_code = self._update_code_for_step(correction_code, current_step, dataframes)
                        
                        # Execute the updated correction code
                        correction_output = self.code_runner.run_code(updated_correction_code)
                        code_outputs.append(correction_output)
                        
                        # Extract dataframe variables from the correction code execution
                        self._extract_dataframes(current_step, dataframes)
                        
                        # Store the correction in the step result
                        query_format.step_results[current_step]["correction_code"] = correction_code
                        query_format.step_results[current_step]["correction_output"] = correction_output
                        
                        # Update the code context with the corrected code
                        query_format.update_code_context(correction_code, {})
                        
                        # Update available resources based on the corrected code
                        self._update_step_resources(query_format, current_step, correction_code, correction_output, is_correction=True)
                        
                        # Get analysis of the correction
                        correction_analysis_query = query_format.get_analysis_format(correction_output, step=current_step)
                        print_color(correction_analysis_query, 32)
                        correction_analysis_reply = self.chatbot.ask(correction_analysis_query)
                        print_color(correction_analysis_reply, 33)
                        
                        # Store the correction analysis in the step result
                        query_format.step_results[current_step]["correction_analysis"] = correction_analysis_reply
                        
                        # Extract summary from the correction analysis reply
                        correction_summary = query_format.extract_summary(correction_analysis_reply)
                        if correction_summary and correction_summary != "No explicit summary provided.":
                            # If we got a good summary from the correction analysis, use it
                            query_format.step_summaries[current_step] = correction_summary
                
                # Advance to the next step
                current_step += 1
            
            # After all steps are complete, get the final result using the existing method
            final_result = self.get_final_result()
            chat_history = self.chatbot.conversation_history
        
        # Return the results in the same format as the original function
        return {
            "query": query,
            "codes": codes,
            "code_outputs": code_outputs,
            "chat_history": chat_history,
            "retries": len(codes)-4,
            "final_result": final_result,
        }

    def _identify_available_resources(self, query_format, current_step):
        """
        Identifies and formats available resources for the current step.
        
        Args:
            query_format: The SequentialCausalThinking instance
            current_step: The current step number
            
        Returns:
            Formatted string of available resources
        """
        if not isinstance(query_format, SequentialCausalThinking):
            return ""
        
        resources_text = "AVAILABLE RESOURCES FOR THIS STEP:\n"
        
        # Add information about the dataset
        if current_step >= 2:  # From step 2 onwards, the original dataset is available
            resources_text += "- Original Dataset: The original dataset is available as 'original_df'\n"
        
        # Add step-specific resource information
        if current_step >= 3:  # From step 3 onwards, the cleaned dataset is available
            resources_text += "- Cleaned Dataset: The cleaned dataset from step 3 is available as 'cleaned_df'\n"
        
        if current_step >= 4:  # From step 4 onwards, any analyzed data might be available
            resources_text += "- Analyzed Data: Any transformed data from step 4 might be available as 'analyzed_df' (if created)\n"
        
        if current_step >= 5:  # From step 5 onwards, the modeling data is available
            resources_text += "- Model Data: The dataset used for modeling from step 5 is available as 'model_df'\n"
            resources_text += "- You can use dowhy, scikit-learn, or statsmodels for causal inference\n"
        
        if current_step >= 6:  # From step 6 onwards, post-hoc analysis data is available
            resources_text += "- Post-hoc Data: The dataset used for post-hoc analysis is available as 'posthoc_df'\n"
            resources_text += "- You can perform additional tests and analyses on the model results\n"
        
        if current_step == 7:  # For Interpretation
            resources_text += "- All Previous Results: Results from all previous steps are available for interpretation\n"
        
        # Add information about variables from code context
        code_context = query_format.get_code_context()
        if "variables" in code_context and code_context["variables"]:
            resources_text += "\nKEY VARIABLES AVAILABLE:\n"
            for var_name, var_info in code_context["variables"].items():
                resources_text += f"- {var_name}: {var_info}\n"
        
        # Add custom resources from previous steps
        custom_resources = query_format.format_available_resources(current_step - 1)
        if custom_resources and custom_resources != "No resources available from previous steps.":
            resources_text += f"\n{custom_resources}\n"
        
        return resources_text
    
    def _update_step_resources(self, query_format, step, code, code_output, is_correction=False):
        """
        Updates the available resources based on the code executed in a step.
        
        Args:
            query_format: The SequentialCausalThinking instance
            step: The current step number
            code: The code that was executed
            code_output: The output of the code execution
            is_correction: Whether this is a correction of previous code
        """
        if not isinstance(query_format, SequentialCausalThinking):
            return
        
        # Initialize resources for this step if not already present
        if step not in query_format.available_resources:
            query_format.available_resources[step] = {}
        
        # Extract variable names from the code
        # This is a simple approach - in a real implementation, you might want to use AST parsing
        import re
        
        # Look for variable assignments
        assignments = re.findall(r'(\w+)\s*=', code)
        
        # Look for DataFrame operations
        df_ops = re.findall(r'(\w+_df)\[[\'\"](\w+)[\'\"]\]', code)
        
        # Add resources based on the step
        resources = query_format.available_resources[step]
        
        if step == 3:  # Data cleaning
            if "original_df" in assignments or any("original_df" in line for line in code.split("\n")):
                resources["original_dataset"] = "Original dataset available as 'original_df'"
            
            if "cleaned_df" in assignments or any("cleaned_df" in line for line in code.split("\n")):
                resources["cleaned_dataset"] = "Cleaned dataset available as 'cleaned_df'"
            
            # Look for specific cleaning operations
            if "fillna" in code:
                resources["missing_values"] = "Missing values have been handled in cleaned_df"
            if "drop" in code:
                resources["dropped_columns"] = "Some columns may have been dropped from cleaned_df"
            if "encode" in code or "get_dummies" in code:
                resources["encoded_variables"] = "Categorical variables have been encoded in cleaned_df"
        
        elif step == 4:  # EDA
            if "cleaned_df" in assignments or any("cleaned_df" in line for line in code.split("\n")):
                resources["cleaned_dataset"] = "Cleaned dataset used for EDA as 'cleaned_df'"
            
            if "analyzed_df" in assignments or any("analyzed_df" in line for line in code.split("\n")):
                resources["analyzed_dataset"] = "Analyzed dataset available as 'analyzed_df'"
            
            # Look for specific EDA operations
            if "correlation" in code or "corr()" in code:
                resources["correlation_analysis"] = "Correlation analysis has been performed"
            if "describe()" in code:
                resources["descriptive_stats"] = "Descriptive statistics have been calculated"
            if "groupby" in code:
                resources["grouped_analysis"] = "Grouped analysis has been performed"
        
        elif step == 5:  # Modeling
            if "cleaned_df" in assignments or any("cleaned_df" in line for line in code.split("\n")):
                resources["cleaned_dataset"] = "Cleaned dataset used for modeling as 'cleaned_df'"
            
            if "model_df" in assignments or any("model_df" in line for line in code.split("\n")):
                resources["model_dataset"] = "Modeling dataset available as 'model_df'"
            
            # Identify the causal inference method used
            methods = {
                "propensity_score": "Propensity score methods have been used",
                "matching": "Matching methods have been used",
                "regression": "Regression methods have been used",
                "instrumental": "Instrumental variable methods have been used",
                "difference_in_differences": "Difference-in-differences has been used",
                "regression_discontinuity": "Regression discontinuity design has been used",
                "causal_model": "DoWhy causal model has been created"
            }
            
            for keyword, description in methods.items():
                if keyword in code.lower():
                    resources["causal_method"] = description
            
            # Look for causal effect estimates
            if "effect" in code.lower() or "ate" in code.lower() or "att" in code.lower():
                resources["causal_effect"] = "Causal effect has been estimated"
            
            # Look for model objects
            model_objects = re.findall(r'(\w+)\s*=\s*\w+\(\)', code)
            if model_objects:
                resources["model_objects"] = f"Model objects created: {', '.join(model_objects)}"
        
        elif step == 6:  # Post-hoc analysis
            if "model_df" in assignments or any("model_df" in line for line in code.split("\n")):
                resources["model_dataset"] = "Model dataset used for post-hoc analysis as 'model_df'"
            
            if "posthoc_df" in assignments or any("posthoc_df" in line for line in code.split("\n")):
                resources["posthoc_dataset"] = "Post-hoc analysis dataset available as 'posthoc_df'"
            
            if "subgroup" in code.lower() or "group" in code.lower():
                resources["subgroup_analysis"] = "Subgroup analysis has been performed"
            if "test" in code.lower() or "p_value" in code.lower() or "significance" in code.lower():
                resources["statistical_tests"] = "Statistical tests have been performed"
            if "sensitivity" in code.lower() or "robust" in code.lower():
                resources["sensitivity_analysis"] = "Sensitivity analysis has been performed"
        
        # Update the resources in the query format
        query_format.available_resources[step] = resources

    def _update_code_for_step(self, code, step, dataframes):
        """
        Updates the code to use the correct dataframe variables based on the step.
        
        Args:
            code: The code to update
            step: The current step number
            dataframes: Dictionary of dataframes from previous steps
            
        Returns:
            Updated code with correct dataframe references
        """
        # No need to update for step 1 and 2
        if step <= 2:
            return code
        
        updated_code = code
        
        # For step 3 (Data cleaning), we don't need to modify anything as it's the first step with code
        
        # For step 4 (EDA), ensure it uses cleaned_df from step 3
        if step == 4 and "cleaned_df" not in code and "df" in code:
            # Replace generic df references with cleaned_df
            updated_code = re.sub(r'\bdf\b', 'cleaned_df', code)
        
        # For step 5 (Modeling), ensure it uses cleaned_df or analyzed_df from previous steps
        elif step == 5:
            if "cleaned_df" not in code and "analyzed_df" not in code and "df" in code:
                # If analyzed_df was created in step 4, use it, otherwise use cleaned_df
                if dataframes["analyzed_df"] is not None:
                    updated_code = re.sub(r'\bdf\b', 'analyzed_df', code)
                else:
                    updated_code = re.sub(r'\bdf\b', 'cleaned_df', code)
        
        # For step 6 (Post-hoc analysis), ensure it uses model_df from step 5
        elif step == 6:
            if "model_df" not in code and "df" in code:
                updated_code = re.sub(r'\bdf\b', 'model_df', code)
        
        # For step 7 (Interpretation), ensure it uses appropriate dataframes
        elif step == 7:
            if "posthoc_df" not in code and "model_df" not in code and "df" in code:
                if dataframes["posthoc_df"] is not None:
                    updated_code = re.sub(r'\bdf\b', 'posthoc_df', code)
                else:
                    updated_code = re.sub(r'\bdf\b', 'model_df', code)
        
        return updated_code
    
    def _extract_dataframes(self, step, dataframes):
        """
        Extracts dataframe variables from the code execution environment.
        
        Args:
            step: The current step number
            dataframes: Dictionary to store the extracted dataframes
        """
        # Get the variables from the code runner's execution environment
        variables = self.code_runner.get_variables()
        
        # Extract dataframes based on their names
        if "original_df" in variables:
            dataframes["original_df"] = True
        
        if "cleaned_df" in variables:
            dataframes["cleaned_df"] = True
        
        if "analyzed_df" in variables:
            dataframes["analyzed_df"] = True
        
        if "model_df" in variables:
            dataframes["model_df"] = True
        
        if "posthoc_df" in variables:
            dataframes["posthoc_df"] = True

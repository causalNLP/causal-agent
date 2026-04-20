import pandas as pd


class QueryFormat:
    """A format of a query"""

    def __init__(self, query, dataset_path, dataset_description):
        self.query = query
        self.dataset_path = dataset_path
        self.dataset_description = dataset_description

    def get_query_format(self) -> str:
        pass

    def get_analysis_format(self, code_output: str) -> str:
        pass


class CausalQueryFormat(QueryFormat):
    def get_query_format(self, include_method_explanation=False):
        # Create a causal query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        if include_method_explanation:
            # Load prompt file relative to this module's directory
            from pathlib import Path
            prompt_path = Path(__file__).resolve().parent / "method_explanations.txt"
            with open(prompt_path) as file:
                method_explanation = file.read()
        else:
            method_explanation = ""

        query = f"""You are an expert in statistics and causal reasoning. You will answer a causal question on a tabular dataset.

The dataset is located at {self.dataset_path}.

The dataset has the following description:
```
{self.dataset_description}
```

To help you understand it, here is the result of df.describe():
```
{df_info}
```
Here are the columns and their types:
```
{columns_and_types}
```

Here are the first 5 rows of the dataset:
```
{df.head()}
```

If there are less than 10 columns, here is the result of df.cov():
```
{(df.cov(numeric_only=True) if len(df.columns) < 10 else "Too many columns to compute covariance")}
```

Finally, here is the output of df.isnull().sum(axis = 0):
```
{nan_per_column}
```

The causal question I would like you to answer is:
```
{self.query}
```

Here are some examples methods, you can choose one from them: [
'propensity_score_weighting' # output the ATE
'propensity_score_matching_treatment_to_control' # output the ATT
'linear_regression' # output the coefficient of the variable of interest
'instrumental_variable' # output the coefficient of the variable of interest
'matching' # output the ATE
'difference_in_differences' # Output the coefficient
'regression_discontinuity_design' # output the coefficient
'matching_treatment_to_control' # output the ATT
'linear_regression / difference_in_means' # output the coefficient / DiM
].

{method_explanation}

Using the descriptions and information from the dataset, implement a python code to answer the causal question. Remember the dataset is located at {self.dataset_path}.
In the case you need to preprocess the data, please do so in the code. The following libraries are available to you: dowhy, pandas, numpy, scipy, scikit-learn, statsmodels. Use the methods from the libraries as best as you can.
Don't code yourself that which is already implemented in the libraries. Do not create random data. Make sure it outputs the quantitative value in the comments of the example method.
The code you output will be executed and you will receive the output. Please make sure to output only one block of code, and make sure the code prints the result you are looking for at the end.
Everything between your first codeblock: ''```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code."""

        return {"pre": [query]}

    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
Can you please provide an analysis of the results?
If the code returns an error, please also provide a corrected version of the code.
Output the entirety of the code, not only the part that needs to be corrected.
Use a single code block. If the code succeeded, don't add any new code, just provide the analysis.
"""
        return query


class CausalQueryVeridicalFormat(QueryFormat):
    def get_query_format(self):
        # Create a causal query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)

        queries = [
            f"""
You are an expert in statistics and causal reasoning. You will use a rigorous scientific framework (Veridical Data Science) to answer a causal question on a tabular dataset. You will answer in multiple steps.
In each step, a series of checklist questions will be asked. You will answer these questions in a clear and concise manner as best as you can. If the question is not appropriate for the task, you can skip it.
Try to keep your answers as simple as possible, but not simpler.

Our discussion's goal is to answer the following problem:
```
{self.query}
```
Keep it in mind as you answer the questions.

Step 1: Domain question, problem formulation.

Checklist questions:
What is the real-world question you are trying to answer? Why is it interesting or important?
How could differences in the formulation affect the final result?
""",
            f"""
Step 2: Data collection and storage.

The dataset is located at {self.dataset_path}.

It has the following description:
```
{self.dataset_description}
```

Here is the result of df.describe():
```
{df_info}
```

Here are the columns and their types:
```
{columns_and_types}
```

Here are the first 5 rows of the dataset:
```
{df.head()}
```

If there are less than 10 columns, here is the result of df.cov():
```
{(df.cov(numeric_only=True) if len(df.columns) < 10 else "Too many columns to compute covariance")}
```

Here is the output of df.isnull().sum(axis = 0):
```
{nan_per_column}
```

Checklist questions:
How was the data generated? What are the experimental design principles? Why is the data
relevant to answer the domain question? Are the conditions affecting data collection stable?

What is the data format? What are the columns? What are their types? What are the units of measurement?
Are the variables categorical or quantitative? How are the variables represented?
What are the potential sources of error in the data? Is the data raw, or has it already been processed?
            """,
            f"""

Step 3: Exploratory data analysis

Do not use code for this step.
Checklist questions:
What are the key statistics of the data? Think of the potential confounders, mediators, and moderators of the data. What are the potential biases in the data?
Should you suspect endogeneity? What are the potential sources of endogeneity? What are the potential instruments?
Are there any variables which are strongly correlated? How might it affect our analysis? Are the confounders measured and adjusted for? What are relationships between the variables that need to be considered to answer the question? What are some direct and indirect cause and effect relations?
            """,
            f"""
Step 4: Modeling

Here are some example methods you can choose from:
            1)'propensity_score_weighting' # output the ATE / ATT / ATC
            When to use: Use propensity score weighting to estimate the Average Treatment Effect (ATE) when you suspect observed confounders are influencing both the treatment and the outcome. The goal is to address the sampling bias in treated and control groups that arises from the relationship between the confounders and the treatment

            2) 'propensity_score_matching_treatment_to_control' # output the ATT
            When to use: Employ propensity score matching treatment-to-control to estimate the Average Treatment Effect on the Treated (ATT) when you specifically want to assess the impact of a treatment on those who actually received it, and you have observed confounders. This method is useful when you want to compare treated units with similar control units based on propensity scores

            3) 'linear_regression' # output the coefficient of the variable of interest
            When to Use: Linear regression can be used to estimate the average treatment effect (ATE) when you assume the relationship between the treatment, confounders, and outcome is linear. It's a special case of outcome model adjustment and is applicable when the identification conditions for confounder adjustment are met.

            4)'matching' # output the ATE
            When to use: Use matching when you want to estimate the average treatment effect (ATE) and you have a set of observed confounding variables. Matching is particularly useful when you can assume that there are no unobserved confounders and you want to create comparable groups of treated and untreated participants.

            5) 'difference_in_differences' # Output the coefficient
            When to use: Use Difference in Differences when you want to estimate the average treatment effect on the treated (ATT) using panel data (i.e., data observed at multiple time points), especially when there might be unobserved unit-specific and time-specific effects. This method is appropriate when you have a group that receives a treatment and a control group that does not, and you observe both groups before and after the treatment.

            6) 'regression_discontinuity_design' # output the coefficient
            When to use: Use regression discontinuity designs when there is a sharp, arbitrary non-linearity in treatment assignment. This means that eligibility for a treatment or intervention is determined by whether a unit's value on some continuous variable (the "running variable" or "assignment variable") falls above or below a specific threshold

            7) 'matching_treatment_to_control' # output the ATT
            When to use: Use this method when you want to estimate the average treatment effect on the treated (ATT) and have a set of observed confounding variables. This is especially useful when members of the population are unlikely to receive treatment, but the treated units had a reasonably high probability of receiving the control. It is also helpful when sampling control units from the general population, but treatment units self-selected into treatment from a smaller subpopulation

            8) 'linear_regression / difference_in_means' # output the coefficient / DiM
            When to use: Use linear regression or difference in means when you want to estimate the association between two variables and you are willing to make strong assumptions about the underlying data generating process. Specifically, this approach is most applicable when:
You assume the identification conditions for confounder adjustment are met.
You can confidently assume a linear relationship between the treatment and the outcome, as well as between the covariates and the outcome.
You want a simple and interpretable method for estimating the treatment effect, especially when dealing with linear parametric assumptions

            9) 'instrumental_variable' # output the coefficient of the variable of interest
            When to use: Employ instrumental variable (IV) strategies when you want to identify causal effects in the presence of unobserved confounders. This approach is valuable when you have a variable (the instrument) that affects the treatment but influences the outcome only through its effect on the treatment
			
			Select 3 most appropriate method  from above to use to answer the question "{self.query}" based on below checklist and argue which one to use among those 3.
Checklist questions:
What are the assumptions of the method? Why is it appropriate for this dataset and this question?
Which library is the best to use for this method, and why? How do you use the right method for that library?
What are some problematic steps that you need to be careful about? How do you avoid them?
What are the key steps of the analysis and what is the key quantitative output value to obtain?
What are the computational constraints on the modeling decision? Is the algorithm
efficient or scalable? What refutation methods are available for this method? What happens
if the refutation method fails? What are the key assumptions of the refutation method?

Here are some examples methods, you can choose one from them: [
'propensity_score_weighting' # output the ATE / ATT / ATC
'propensity_score_matching_treatment_to_control' # output the ATT
'linear_regression' # output the coefficient of the variable of interest
'instrumental_variable' # output the coefficient of the variable of interest
'matching' # output the ATE
'difference_in_differences' # Output the coefficient
'regression_discontinuity_design' # output the coefficient
'matching_treatment_to_control' # output the ATT
'linear_regression / difference_in_means' # output the coefficient / DiM
].

Then, after you explain the method, implement a python code to answer this question. Remember the dataset is located at {self.dataset_path}.
In the case you need to preprocess the data, please do so in the code. The following libraries are available to you: dowhy, pandas, numpy, scipy, scikit-learn, statsmodels. Use the methods from the libraries as best as you can.
Don't code yourself that which is already implemented in the libraries. Do not create random data. Make sure it outputs the quantitative value in the comments of the example method.
The code you output will be executed and you will receive the output. Please make sure to output only one block of code, and make sure the code prints the result you are looking for at the end.
Everything between your first codeblock: ''```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code.
            """,
            """
Step 5: Post hoc analysis
What relationships or results are revealed with post hoc analysis? Are these relationships stable?
If necessary, specify null hypotheses.
            """,
            """
Step 6: Interpretation of results
Final Answer: The final answer to the original input question. Please provide a structured response including the following information. If a field is not applicable, use "NA".
- Method: [The method used]
- Causal Effect: [The causal effect]
- Standard Deviation: [The standard deviation]
- Treatment Variable: [The treatment variable]
- Outcome Variable: [The outcome variable]
- Covariates: [List of covariates]
- Instrument / Running Variable / Temporal variable: [The relevant variable, if applicable]
- Results of statistical test: [Key statistical results, if applicable]
- Explanation for model choice: [Explanation, if applicable]
- Regression equation: [The regression equation, if applicable]
            """,
        ]

        return {"pre": queries[0:4], "post": queries[4:]}

    def get_analysis_format(self, code_output: str, remaining_tries=1) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
If the code returns an error, please also provide a corrected version of the code.
The timeout for this task is 180 seconds.
Output the entirety of the code, not only the part that needs to be corrected.
Use a single code block. If you are confident, then reply "No corrections needed".
Make sure the quantitative value is output, and that the method and code are correct.
You have {remaining_tries} attempts to correct the code.
"""
        return query

class ReActFormat(QueryFormat):
    def get_query_format(self):
        # Create a ReAct query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)

        format = f"""
Data Description:
{self.dataset_description}. The dataset is located at {self.dataset_path}.

You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

You should use the tools below to answer the question posed of you:
`python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.


### Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be python_repl_ast
Action Input: the input to the action, should be the code to execute
Observation: the result of the action
...(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question. Please provide a structured response including the following information. If a field is not applicable, use "NA".
- Method: [The method used]
- Causal Effect: [The causal effect]
- Standard Deviation: [The standard deviation]
- Treatment Variable: [The treatment variable]
- Outcome Variable: [The outcome variable]
- Covariates: [List of covariates]
- Instrument / Running Variable / Temporal variable: [The relevant variable, if applicable]
- Results of statistical test: [Key statistical results, if applicable]
- Explanation for model choice: [Explanation, if applicable]
- Regression equation: [The regression equation, if applicable]


Note that you should import the libraries if needed.
DO NOT create any plotting.
For all outputs in code, THE `print()` function MUST be called.
If you use Action in this step, stop after generating the Action Input and await the execution outcome from `python_repl_ast`.
If you output the Final Answer in this step, do not use Action.

Here is an example of using the `python_repl_ast`:
Action: python_repl_ast
Action Input:
 ```python
            # Your code goes here
            import pandas as pd
            print(df.head())
            ```
This is the result:
| | question | response | prob_random_correct |
|---|--------------------------------------------|-----------|----------------------|
| 0 | children_with_1_or_more_vaccination         | correct   | 0.333333             |
| 1 | children_with_1_or_more_vaccination         | correct   | 0.333333             |
| 2 | children_with_1_or_more_vaccination         | incorrect | 0.333333             |
| 3 | children_with_1_or_more_vaccination         | incorrect | 0.333333             |
| 4 | children_with_1_or_more_vaccination         | incorrect | 0.333333             |

Begin!

Question:
{self.query}

Here are some examples methods, you can choose one from them:
- propensity_score_weighting: output the ATE
- propensity_score_matching_treatment_to_control: output the ATT
- linear_regression: output the coefficient of the variable of interest
- instrumental_variable: output the coefficient of the variable of interest
- matching: output the ATE
- difference_in_differences: output the coefficient
- regression_discontinuity_design: output the coefficient
- matching_treatment_to_control: output the ATT
- linear_regression / difference_in_means: output the coefficient / DiM

        """
        return {"pre": [format]}

    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
"""
        return query
class ProgramOfThoughtsFormat(QueryFormat):
    def get_query_format(self):
        # Create a chain of thought query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)

        format = f"""
You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write python code to analyze the whole table. You can use any python library. The returned value of the program is supposed to be the answer. The format of the code should be

```python
def solution():
    # import libraries if needed
    # load data
    # write code to get the answer
    # return answer
print(solution())
```

Data Description:
{self.dataset_description}
The dataset is located at {self.dataset_path}.

First 10 rows of the data:
{df.head(10)}

Question:
{self.query}

Here are some examples methods, you can choose one from them:
- propensity_score_weighting: output the ATE
- propensity_score_matching_treatment_to_control: output the ATT
- linear_regression: output the coefficient of the variable of interest
- instrumental_variable: output the coefficient of the variable of interest
- matching: output the ATE
- difference_in_differences: output the coefficient
- regression_discontinuity_design: output the coefficient
- matching_treatment_to_control: output the ATT
- linear_regression / difference_in_means: output the coefficient / DiM

Response: The final answer to the original input question. Please provide a structured response including the following information. If a field is not applicable, use "NA".
- Method: [The method used]
- Causal Effect: [The causal effect]
- Standard Deviation: [The standard deviation]
- Treatment Variable: [The treatment variable]
- Outcome Variable: [The outcome variable]
- Covariates: [List of covariates]
- Instrument / Running Variable / Temporal variable: [The relevant variable, if applicable]
- Results of statistical test: [Key statistical results, if applicable]
- Explanation for model choice: [Explanation, if applicable]
- Regression equation: [The regression equation, if applicable]

"""
        return {"pre": [format]}

    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
Can you please provide an analysis of the results?
If the code returns an error, please also provide a corrected version of the code.
Output the entirety of the code, not only the part that needs to be corrected.
Use a single code block. If the code succeeded, don't add any new code, just provide the analysis.
"""
        return query

class SequentialCausalThinking(QueryFormat):
    """
    A class that implements a sequential thinking flow for LLMs to use code in persistent mode
    and execute step by step with proper planning for causal analysis.
    
    This approach breaks down causal analysis into 8 manageable steps:
    1. Domain question or problem formulation
    2. Data collection
    3. Data cleaning and preprocessing
    4. Exploratory data analysis
    5. Modeling
    6. Post-hoc analysis
    7. Interpretation of results
    8. Update domain knowledge
    
    Each step builds on the previous ones, creating a structured and thorough approach to causal analysis.
    """
    
    def __init__(self, query, dataset_path, dataset_description):
        super().__init__(query, dataset_path, dataset_description)
        self.current_step = 1
        self.step_results = {}  # Store results from each step
        self.step_summaries = {}  # Store concise summaries for each step
        self.code_context = {}  # Store code and variables for persistent execution
        self.available_resources = {}  # Track available resources from each step
    
    def get_query_format(self):
        """
        Returns a series of prompts that guide the LLM through the 8-step sequential thinking process.
        Each step has specific questions or tasks for the LLM to address.
        """
        # Load dataset information
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        
        # Define the prompts for each step
        step_prompts = [
            # Step 1: Domain question or problem formulation
            f"""
            You are an expert in statistics and causal reasoning. You will use a sequential thinking approach to answer a causal question on a tabular dataset.
            
            Our goal is to answer the following causal question:
            ```
            {self.query}
            ```
            
            STEP 1: DOMAIN QUESTION OR PROBLEM FORMULATION
            
            In this step, you need to clearly formulate the domain question and understand how it can be answered with causal analysis.
            
            Please address the following questions:
            1. What is the real-world question you are trying to answer? Why is it interesting or important?
            2. How can this question be formulated as a causal inference problem?
            3. What are the key variables involved (treatment, outcome, potential confounders)?
            4. How might differences in the formulation affect the final result?
            5. What assumptions might be necessary to answer this question?
            
            DO NOT proceed to coding yet. Focus on understanding and formulating the problem clearly.
            
            At the end of your response, please provide a 5-10 line SUMMARY that captures all the key insights from this step.Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.

            Format your summary as follows:
            
            SUMMARY:
            [Your 5-10 line summary here]
            """,
            
            # Step 2: Data collection
            f"""
            STEP 2: DATA COLLECTION
            
            Now let's examine the dataset we'll be working with.
            
            The dataset is located at {self.dataset_path}.
            
            It has the following description:
            ```
            {self.dataset_description}
            ```
            
            Here is the result of df.describe():
            ```
            {df_info}
            ```
            
            Here are the columns and their types:
            ```
            {columns_and_types}
            ```
            
            Here are the first 5 rows of the dataset:
            ```
            {df.head()}
            ```
            
            If there are less than 10 columns, here is the result of df.cov():
            ```
            {(df.cov(numeric_only=True) if len(df.columns) < 10 else "Too many columns to compute covariance")}
            ```
            
            Here is the output of df.isnull().sum(axis = 0):
            ```
            {nan_per_column}
            ```
            
            Please address the following questions:
            1. How was the data collected? What are the design principles?
            2. Why is this data relevant to answer the domain question?
            3. Are there any limitations in the data collection process that might affect our analysis?
            4. Are the conditions affecting data collection stable?
            5. What are the key variables in the dataset and how do they relate to our causal question?
            
            DO NOT proceed to coding yet. Focus on understanding the data collection process and its implications for our analysis.
            
            At the end of your response, please provide a 10-15 line SUMMARY that captures all the key insights from this step, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 10-15 line summary here]
            """,
            
            # Step 3: Data cleaning and preprocessing
            f"""
            STEP 3: DATA CLEANING AND PREPROCESSING
            
            Now it's time to plan and implement the data cleaning and preprocessing steps.
            
            Please address the following questions first:
            1. What steps and methods are needed to convert the data to a format suitable for causal analysis?
            2. Are there missing values that need to be handled? If so, how?
            3. Are there outliers that need to be addressed? If so, how?
            4. Do any variables need to be transformed or encoded?
            5. Will these preprocessing steps add bias to the data? How can we minimize this?
            
            After addressing these questions, implement the necessary data cleaning and preprocessing steps.
            You can use pandas, numpy, and other libraries as needed.
            
            ```python
            # Your code for data cleaning and preprocessing goes here
            # Remember that this code will be executed and the results will be available in subsequent steps
            # Start by loading the dataset
            import pandas as pd
            import numpy as np
            
            # Load the original dataset
            original_df = pd.read_csv("{self.dataset_path}")
            
            # Create a copy for cleaning to preserve the original data
            cleaned_df = original_df.copy()
            
            # Implement your data cleaning and preprocessing steps on cleaned_df
            # ...
            
            # Display the first few rows of the cleaned dataset
            print("First few rows of the cleaned dataset:")
            print(cleaned_df.head())
            
            # Display summary statistics of the cleaned dataset
            print("\\nSummary statistics of the cleaned dataset:")
            print(cleaned_df.describe())
            
            # Display missing values in the cleaned dataset
            print("\\nMissing values in the cleaned dataset:")
            print(cleaned_df.isnull().sum())
            ```
            
            At the end of your response, please provide a 10-15  line SUMMARY that captures all the key insights and preprocessing steps from this step, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 10-15 line summary here]
            """,
            
            # Step 4: Exploratory data analysis
            """
            STEP 4: EXPLORATORY DATA ANALYSIS
            
            Now let's explore the data to identify interesting relationships before modeling.
            
            Please address the following questions first:
            1. What are the key statistics and distributions of the variables?
            2. Are there any interesting correlations or patterns in the data?
            3. Can we identify potential confounders, mediators, or moderators?
            4. What are the potential biases in the data?
            5. Are there any variables that are strongly correlated? How might this affect our analysis?
            
            After addressing these questions, implement exploratory data analysis to understand the data.
            You can use pandas, numpy, and other libraries as needed.
            DO NOT use visualization libraries.
            
            ```python
            # Your code for exploratory data analysis goes here
            # Remember that this code will be executed and the results will be available in subsequent steps
            # The cleaned dataset from the previous step is available as 'cleaned_df' 
            
            # Implement your exploratory data analysis on cleaned_df
            # ...
            
            
            # Example: Examine relationships between treatment, outcome, and potential confounders
            # ...
            
            
            # Calculate and display key statistics
            # ...
            
            # You can create a new dataframe for any transformed data if needed
            # analyzed_df = cleaned_df.copy()
            # [your transformations here]
            ```
            
            At the end of your response, please provide a 10-15  line SUMMARY that captures all the key insights from the exploratory data analysis, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            SUMMARY:
            [Your 10-15 line summary here]
            """,
            
            # Step 5: Modeling
            f"""
            STEP 5: MODELING
            
            Here are some example methods you can choose from:
            1)'propensity_score_weighting' # output the ATE / ATT / ATC
            When to use: Use propensity score weighting to estimate the Average Treatment Effect (ATE) when you suspect observed confounders are influencing both the treatment and the outcome. The goal is to address the sampling bias in treated and control groups that arises from the relationship between the confounders and the treatment

            2) 'propensity_score_matching_treatment_to_control' # output the ATT
            When to use: Employ propensity score matching treatment-to-control to estimate the Average Treatment Effect on the Treated (ATT) when you specifically want to assess the impact of a treatment on those who actually received it, and you have observed confounders. This method is useful when you want to compare treated units with similar control units based on propensity scores

            3) 'linear_regression' # output the coefficient of the variable of interest
            When to Use: Linear regression can be used to estimate the average treatment effect (ATE) when you assume the relationship between the treatment, confounders, and outcome is linear. It's a special case of outcome model adjustment and is applicable when the identification conditions for confounder adjustment are met.

            4)'matching' # output the ATE
            When to use: Use matching when you want to estimate the average treatment effect (ATE) and you have a set of observed confounding variables. Matching is particularly useful when you can assume that there are no unobserved confounders and you want to create comparable groups of treated and untreated participants.

            5) 'difference_in_differences' # Output the coefficient
            When to use: Use Difference in Differences when you want to estimate the average treatment effect on the treated (ATT) using panel data (i.e., data observed at multiple time points), especially when there might be unobserved unit-specific and time-specific effects. This method is appropriate when you have a group that receives a treatment and a control group that does not, and you observe both groups before and after the treatment.

            6) 'regression_discontinuity_design' # output the coefficient
            When to use: Use regression discontinuity designs when there is a sharp, arbitrary non-linearity in treatment assignment. This means that eligibility for a treatment or intervention is determined by whether a unit's value on some continuous variable (the "running variable" or "assignment variable") falls above or below a specific threshold

            7) 'matching_treatment_to_control' # output the ATT
            When to use: Use this method when you want to estimate the average treatment effect on the treated (ATT) and have a set of observed confounding variables. This is especially useful when members of the population are unlikely to receive treatment, but the treated units had a reasonably high probability of receiving the control. It is also helpful when sampling control units from the general population, but treatment units self-selected into treatment from a smaller subpopulation

            8) 'linear_regression / difference_in_means' # output the coefficient / DiM
            When to use: Use linear regression or difference in means when you want to estimate the association between two variables and you are willing to make strong assumptions about the underlying data generating process. Specifically, this approach is most applicable when:
You assume the identification conditions for confounder adjustment are met.
You can confidently assume a linear relationship between the treatment and the outcome, as well as between the covariates and the outcome.
You want a simple and interpretable method for estimating the treatment effect, especially when dealing with linear parametric assumptions

            9) 'instrumental_variable' # output the coefficient of the variable of interest
            When to use: Employ instrumental variable (IV) strategies when you want to identify causal effects in the presence of unobserved confounders. This approach is valuable when you have a variable (the instrument) that affects the treatment but influences the outcome only through its effect on the treatment
			
			Select 3 most appropriate method  from above to use to answer the question "{self.query}" based on below checklist and argue which one to use among those 3.
            
            Checklist questions:
            1. What are the assumptions of the method? Why is it appropriate for this dataset and this question?
            2. Which library is the best to use for this method, and why? How do you use the right method for that library?
            3. What are some problematic steps that you need to be careful about? How do you avoid them?
            4. What are the key steps of the analysis and what is the key quantitative output value to obtain?
            5. What are the computational constraints on the modeling decision? Is the algorithm
            efficient or scalable? What refutation methods are available for this method? What happens
            if the refutation method fails? What are the key assumptions of the refutation method?
            
            
            After addressing these questions, implement the selected causal inference method.
            You can use dowhy, pandas, numpy, scipy, scikit-learn, statsmodels, and other libraries as needed.
            DO NOT use Seaborn.
            
            
            ```python
            # Your code for causal inference modeling goes here
            # Remember that this code will be executed and the results will be available in subsequent steps
            # The cleaned dataset from step 3 is available as 'cleaned_df'
            # Any transformed data from step 4 might be available if you created it
            
            # Import necessary libraries
            # Example: from dowhy import CausalModel
            # Example: from sklearn.linear_model import LogisticRegression
            # ...
            
            # Create a copy for modeling to preserve the cleaned data
            model_df = cleaned_df.copy()
            
            # Implement your causal inference method on model_df
            # ...
            
            # Calculate and display the causal effect estimate
            # ...
            
            # Perform sensitivity analysis or robustness checks if appropriate
            # ...
            
            # Store the key results for later steps
            # Example: causal_effect = 0.25  # Store the estimated causal effect
            # Example: print(f"Estimated causal effect: 0.25")
            ```
            
            At the end of your response, please provide a 10-15 line SUMMARY that captures all the key modeling decisions and results, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 10-15 line summary here]
            """,
            
            # Step 6: Post-hoc analysis
            """
            STEP 6: POST-HOC ANALYSIS
            
            Now let's perform post-hoc analysis to reveal additional relationships or results.
            
            Please address the following questions first:
            1. What additional relationships or results are revealed with post-hoc analysis?
            2. Are these relationships stable?
            3. What null hypotheses can we specify and test?
            4. Are there any subgroup effects or heterogeneous treatment effects?
            
            After addressing these questions, implement post-hoc analysis to explore additional insights.
            You can use the results from previous steps and additional statistical tests as needed.
            
            ```python
            # Your code for post-hoc analysis goes here
            # Remember that this code will be executed and the results will be available in subsequent steps
            # The model results from step 5 are available
            # You can access the cleaned data as 'cleaned_df' and model data as 'model_df'
            
            # Create a copy for post-hoc analysis
            posthoc_df = model_df.copy()
            
            # Implement your post-hoc analysis
            # ...
            
            # Example: Test for heterogeneous treatment effects across subgroups
            # ...
            
            # Example: Perform additional statistical tests
            # ...
            
            # Display your findings
            # ...
            ```
            
            At the end of your response, please provide a 10-15 line SUMMARY that captures all the key insights from the post-hoc analysis, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 10-15 line summary here]
            """,
            
            # Step 7: Interpretation of results
            """
            STEP 7: INTERPRETATION OF RESULTS
            
            Now let's interpret the results of our causal analysis.
            
            Please address the following questions:
            1. What do the results mean in the context of our domain question?
            2. What are the practical implications of our findings?
            3. What are the limitations of our analysis?
            4. How confident are we in our causal effect estimate?
            5. What alternative explanations might exist for our findings?
            
            Provide a clear and concise interpretation of the results, focusing on the causal effect estimate and its implications.
            At the end of your response, please provide a 10-15 line SUMMARY that captures all the key interpretations and implications, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 10-15 line summary here]
            """
        ]
        
        return {"pre": step_prompts}
    
    def get_analysis_format(self, code_output: str, step=None) -> str:
        """
        Returns a prompt for analyzing the results of a specific step in the sequential thinking process.
        If step is not provided, it analyzes the final results.
        
        Args:
            code_output: The output of the code execution
            step: The step number to analyze (optional)
        
        Returns:
            A prompt for analyzing the results
        """
        if step is None:
            # Analyze the final results
            query = f"""
            The code you provided has been executed, here is the output:
            ```
            {code_output}
            ```
            
            Please provide a comprehensive analysis of the results, addressing the following:
            1. What is the estimated causal effect and its interpretation?
            2. How confident are we in this estimate?
            3. What are the key limitations and assumptions of our analysis?
            4. What practical recommendations can we make based on our findings?
            5. What follow-up analyses would be valuable?
            
            If the code returns an error, please provide a corrected version of the code.
            Output the entirety of the code, not only the part that needs to be corrected.
            Use a single code block. If the code succeeded, don't add any new code, just provide the analysis.
            
            At the end of your response, please provide a 5-10 line SUMMARY that captures all the key insights from this analysis, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 5-10 line summary here]
            """
        else:
            # Analyze the results of a specific step
            query = f"""
            The code for Step {step} has been executed, here is the output:
            ```
            {code_output}
            ```
            
            Please analyze the results of this step and prepare for the next step.
            
            If the code returns an error, please provide a corrected version of the code.
            Output the entirety of the code, not only the part that needs to be corrected.
            Use a single code block. If the code succeeded, don't add any new code, just provide the analysis.
            
            After analyzing the results, we will proceed to Step {step + 1}.
            
            At the end of your response, please provide a 5-10  line SUMMARY that captures all the key insights from this step, Include as much
            information that might be relevant for next steps of anwering the questions asked programatically.
            Format your summary as follows:
            
            SUMMARY:
            [Your 5-10 line summary here]
            """
        
        return query
    
    def extract_summary(self, text):
        """
        Extracts the summary from the LLM's response.
        
        Args:
            text: The LLM's response text
            
        Returns:
            The extracted summary or a default message if no summary is found
        """
        # Look for the summary section
        summary_start = text.find("SUMMARY:")
        if summary_start == -1:
            # Try alternative formats
            summary_start = text.find("Summary:")
            if summary_start == -1:
                return "No explicit summary provided."
        
        # Extract the summary text
        summary_text = text[summary_start:].strip()
        
        # Find the end of the summary (next double newline or end of text)
        end_idx = summary_text.find("\n\n")
        if end_idx != -1:
            summary_text = summary_text[:end_idx].strip()
        
        # Remove the "SUMMARY:" prefix
        if summary_text.startswith("SUMMARY:"):
            summary_text = summary_text[len("SUMMARY:"):].strip()
        elif summary_text.startswith("Summary:"):
            summary_text = summary_text[len("Summary:"):].strip()
        
        return summary_text if summary_text else "No explicit summary provided."
    
    def advance_step(self):
        """
        Advances to the next step in the sequential thinking process.
        
        Returns:
            The prompt for the next step
        """
        self.current_step += 1
        if self.current_step <= 8:
            return self.get_query_format()["pre"][self.current_step - 1]
        else:
            return None
    
    def store_step_result(self, step, result):
        """
        Stores the result of a specific step.
        
        Args:
            step: The step number
            result: The result of the step
        """
        self.step_results[step] = result
        
        # Extract and store the summary if a reply is available
        if "reply" in result:
            summary = self.extract_summary(result["reply"])
            self.step_summaries[step] = summary
    
    def get_step_result(self, step):
        """
        Gets the result of a specific step.
        
        Args:
            step: The step number
        
        Returns:
            The result of the step
        """
        return self.step_results.get(step)
    
    def get_step_summary(self, step):
        """
        Gets the summary of a specific step.
        
        Args:
            step: The step number
        
        Returns:
            The summary of the step or None if not available
        """
        return self.step_summaries.get(step, None)
    
    def update_code_context(self, code, variables):
        """
        Updates the code context with new code and variables.
        
        Args:
            code: The code executed
            variables: The variables created or updated by the code
        """
        self.code_context["code"] = self.code_context.get("code", "") + "\n" + code
        self.code_context["variables"] = {**self.code_context.get("variables", {}), **variables}
    
    def get_code_context(self):
        """
        Gets the current code context.
        
        Returns:
            The current code context
        """
        return self.code_context
    
    def update_available_resources(self, step, resources):
        """
        Updates the available resources after a step is completed.
        
        Args:
            step: The step number
            resources: Dictionary of resources created or updated in this step
        """
        self.available_resources[step] = resources
    
    def get_available_resources(self, up_to_step=None):
        """
        Gets the available resources up to a specific step.
        
        Args:
            up_to_step: The step number to get resources up to (inclusive)
                       If None, returns all available resources
        
        Returns:
            Dictionary of available resources
        """
        if up_to_step is None:
            return self.available_resources
        
        resources = {}
        for step in range(1, up_to_step + 1):
            if step in self.available_resources:
                resources[step] = self.available_resources[step]
        
        return resources
    
    def format_available_resources(self, up_to_step):
        """
        Formats the available resources up to a specific step as a string.
        
        Args:
            up_to_step: The step number to get resources up to (inclusive)
        
        Returns:
            Formatted string of available resources
        """
        resources = self.get_available_resources(up_to_step)
        if not resources:
            return "No resources available from previous steps."
        
        formatted_resources = "AVAILABLE RESOURCES FROM PREVIOUS STEPS:\n"
        
        for step, step_resources in resources.items():
            formatted_resources += f"From Step {step}:\n"
            
            if isinstance(step_resources, dict):
                for name, description in step_resources.items():
                    formatted_resources += f"- {name}: {description}\n"
            else:
                formatted_resources += f"- {step_resources}\n"
            
            formatted_resources += "\n"
        
        return formatted_resources.strip()


if __name__ == "__main__":
    query = "what is effect?"
    dataset_path = "/Users/Vishal/Project/fork_/causalscientist/data/qrdata/ak91.csv"
    dataset_description = "The dataset contains blablabla"
    query_format = SequentialCausalThinking(query, dataset_path, dataset_description)
    
    print(query_format.get_query_format())

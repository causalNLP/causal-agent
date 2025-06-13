## This file contains the methods that can be used to generate contexts for synethetic data using OpenAI API

def generate_data_summary(df, n_cont_vars, n_bin_vars, method, cutoff=None) -> str:
    """
    Generate a summary of the DataFrame.
    """


    continuous_vars = [f"X{i}" for i in range(1, n_cont_vars + 1)]
    binary_vars = [f"X{i}" for i in range(n_cont_vars + 1, n_cont_vars + n_bin_vars + 1)]
    
    information = "The dataset contains the following **continuous covariates**: " + ", ".join(continuous_vars) + ".\n"
    information += "The dataset contains the following **binary covariates**: " + ", ".join(binary_vars) + ".\n"
    information += "The **outcome variable** is Y.\n"
    information += "The **treatment variable** is D.\n"

    if method == "encouragement":
        information += "This is an encouragement design where Z is the instrument, i.e., the \
            , the initial treatment assignment \ n"
    elif method == "IV":
        information += "This is an IV design where Z is the instrument \n"
    elif method == "rdd":
        information += "The running variable is running_X, and the cutoff is {}\n".format(cutoff)
    elif method ==  "did_twfe":
        information += "This is a staggered Difference in Difference where D indicates whether or not the unit is treated \
            at time t. Similarly, year denotes the time at which the data was measured.\n"
    elif method == "did_canonical":
        information += "This is a canonical Difference in Difference where D indicates whether or not the unit is treated \
            at time t. Similarly, post is a binary variable indicating post / pre-intervention time points \
            , post = 1 indicates post-intervention time points.\n"
    
    information += "Here is the statistical summary of the variables: \n " + str(df.describe(include='all')) + "\n"

    return information

        
def create_prompt(summary, method, history):
    
    """
    creates a prompt for the OpenAI API to generate a context for the dataset 
    """

    method_names = {"encouragement": "Encouragement Design", "did_twfe": "Difference in Differences with Two-Way Fixed Effects",
                    "did_canonical": "Canonical Difference in Differences", "IV": "Instrumental Variable",
                    "multi_rct": "Multi-Treatment Randomized Control Trial", "rdd": "Regression Discontinuity Design",
                    "observational": "Observational", "rct": "Randomized Control Trial"}
    
    
    prompt = f"""
You are a helpful assistant helping me generate realistic contexts for synthetic datasets
The synthetic dataset is generated for ***{method_names[method]} *** studies.
 ** Dataset Summary **
    {summary}

Likewise, here are the contexts that have already been used. **Avoid overlap**. 
 ** Previous Contexts **
    {history}

  ***Your tasks***:
 1. Invent a real-world scenario for the dataset. The scenario should be relatable to ***{method_names[method]} studies***. 
    If data comes from a randomized trial, mention that.
    Each dataset must have a **unique and realistic context** and **must not overlap** with previously used contexts.
 2. Assign **realistic headings** for each column in the dataset. Make the column names concise. For example, "Hours Worked per Week" could be writted as "hours_worked". You can provide more details in the description section.
 3. Write a paragraph describing the dataset, how it was collected, the motivation for the study. Then, write a
    one-line description of each column.
 4. Write a **natural language query** that can be answered by performing a causal analysis on the dataset.
      IMPORTANT NOTES:
    - Do not mention any variable names in the question explicitly
    - Do not describe covariates or confounders explicitly or hint the column names.
    The question should relate the treatment and outcome variables and not explicitly mention them. Information about the treatment and outcome 
 should be inferable from the description.   

 5. Do not mention any causal inference or statistical methods and terminologies.
 6. Include a one-sentence **summary of the dataset** that captures its essence.


 Return your output as a JSON object with the following keys:
 - "variable_labels": {{ "X1": "education_years", ... }}
 - "description": "<paragraph>"
 - "question": "<causal question>"
 - "context":<context>
    """

    return prompt 

def filter_question(question):
    """
    Filter the question to remove context.
    """

    prompt = """
    You are a helpful assistant. Help me filter this causal query. 

    The query is: {}
    The query should not provide information on what variables one needs to consider in course of causal analysis.
    For example, 
    Bad question: "What is the effect of the training program on job outcomes considering education and experience?" is 
    a bad question because it mentions the use of specific variables in the analysis. 
    One can change the question to "What is the effect of the training program on job outcomes?"

    If the question is already filtered, return the question as it is.

    In the output, return only the filtered query. For the above example, the output should be:
    "What is the effect of the training program on job outcomes?"
    """

    return prompt.format(question)





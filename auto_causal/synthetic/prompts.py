## This file contains the functions that can be used to create prompts for generating synthetic data contexts.

def generate_data_summary(df, n_cont_vars, n_bin_vars, method, cutoff=None) -> str:
    """
    Generate a summary of the input dataset. The summary includes information about column headings
    for continuuous, binary, treatment, and outcome variables. Additionally, it also includes information on the method
    used to generate the dataset and the basic statistical summary.

    Args:
        df (pd.DataFrame): The input dataset.
        n_cont_vars (int): Number of continuous variables in the dataset
        n_bin_vars (int): Number of binary variables in the dataset
        method (str): The method used to generate the dataset
        cutff (float, None): The cutoff value for RDD data

    Returns:
        str: Summary of the (raw) dataset.

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


def create_prompt(summary, method, domain, history):

    """
    Creates a prompt for the OpenAI API to generate a context for the given dataset

    Args:
        summary (str): Summary of the dataset
        method (str): The method used to generate the dataset
        domain (str): The domain of the dataset
        history (str): Previous contexts that have been used. We use this to avoid overlap in contexts

 """

    method_names = {"encouragement": "Encouragement Design", "did_twfe": "Difference in Differences with Two-Way Fixed Effects",
                    "did_canonical": "Canonical Difference in Differences", "IV": "Instrumental Variable",
                    "multi_rct": "Multi-Treatment Randomized Control Trial", "rdd": "Regression Discontinuity Design",
                    "observational": "Observational", "rct": "Randomized Control Trial"}


    prompt = f"""
You are a helpful assistant generating realistic, domain-specific contexts for synthetic datasets.

The current dataset is designed for **{method_names[method]}** studies in the **{domain}** domain.

Below is a statistical summary of the dataset:
**Dataset Summary**
{summary}

Here are summaries of previously used contexts. **Avoid overlap with these**:
**Previous Contexts**
{history}

### Your Tasks:
1. Propose a **realistic real-world scenario for this dataset. It should suit a **{method_names[method]}** study. If the data is from a randomized trial, state this clearly.
2. Assign **realistic and concise column names**. For example, "Hours Worked per Week" can be renamed to `hours_worked`. Use snake_case formatting. 
In the description section, give a one-line description of each column i.e. describe what each columns represents. For example, "hours_worked: the number of hours worked per week by the individual." Use newline for each column description.
3. Write a paragraph describing the dataset, including its background, how it was collected, and the motivation for the study.
4. Propose a **natural language causal question** that is answerable using the dataset.
 **Important notes**:
 - Do not mention column names explicitly.
 - Do not describe covariates or confounders directly.
 - The question should relate the treatment and outcome, but implicitly. These are inferred from the description.
4. Avoid all causal inference or statistical terminologies in the context and question.
5. Write a **one-to-two-sentence summary** that captures the essence of the dataset.

 Return your output as a JSON object with the following keys:
 - "variable_labels": {{ "X1": "education_years", ... }}
 - "description": "<paragraph>"
 - "question": "<causal question>"
 - "summary": <summary>
 - "domain": "<domain>"
 """

    return prompt

def filter_question(question):
    """
    Filter the question to remove explicit mentions of variables.

    Args:
        question (str): The original causal query

    Returns:
        str: The filtered causal query
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
    No need to say "The filtered query is" or anything like that. Return the query only.
    """

    return prompt.format(question)
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
                    "observational": "Observational", "rct": "Randomized Control Trial", "frontdoor": "Front-Door Causal Inference"}

    domain_guides = {
        "education": "Education data often includes student performance, school-level features, socioeconomic background, and intervention types like tutoring or online classes.",
        "healthcare": "Healthcare data may include treatments, diagnoses, hospital visits, recovery outcomes, or demographic details.",
        "labor": "Labor datasets typically include income, education, job type, employment history, and training programs.",
        "policy": "Policy evaluation data may track program participation, regional differences, economic impact, and public outcomes like housing, safety, or benefits.",
        "psychology": "Psychology data often covers cognitive tests, personality traits, mental health measures, behavioral interventions, and clinical outcomes.",
        "law": "Law-related data may include court cases, trial outcomes, sentencing decisions, legal representation, and demographic information of defendants or plaintiffs.",
        "political_science": "Political science datasets often track elections, voter demographics, political attitudes, policy preferences, campaign effects, and governance outcomes.",
        "economics": "Economics data typically includes household income, consumption, trade, inflation, economic growth indicators, and market-level shocks or interventions.",
        "sociology": "Sociology datasets may involve social networks, inequality, family structure, demographic patterns, community studies, and social mobility indicators."
    }
    
    prompt = f"""
You are a helpful assistant generating realistic, domain-specific contexts for synthetic datasets.

The current dataset is designed for **{method_names[method]}** studies in the **{domain}** domain.

### Dataset Summary
{summary}

### Previously Used Contexts (avoid duplication)
{history}

### Domain-Specific Guidance
{domain_guides.get(domain, '')}


---

### Your Tasks:
1. Propose a **realistic real-world scenario** that fits a {method_names[method]} study in the {domain} domain. Mention whether the data was collected from a randomized trial, policy rollout, or real-world observation.
2a. Assign **realistic and concise variable names** in snake_case. Map original variable names like `"X1"` to names like `"education_years"`.
2b. Provide a **one-line natural-language description for each variable** (e.g., `education_years: total years of formal schooling completed by the individual.`). Use newline-separated key-value format.
3. Write a **paragraph** describing the dataset's background: who collected it, what was studied, why, and how. Then, provide a clear description of each variable in the dataset, explaining what it represents and, where relevant, 
its type (e.g., continuous, binary, categorical). For binary or categorical variables, specify what the values mean.
4. Write a **natural language causal question** the dataset could answer. The question should:
   - Relate implicitly to the treatment and outcome
   - Avoid any statistical or causal terminology
   - Avoid naming variables directly
   - Feel like it belongs in a news article or report
5. Write a **1-2 sentence summary** capturing the dataset's overall intent and contents.

---


 Return your output as a JSON object with the following keys:
 - "variable_labels": {{ "X1": "education_years", ... }}
 - "description": "<paragraph>"
 - "question": "<causal question>"
 - "summary": <summary>
 - "domain": "<domain>"

 Return only a valid JSON object. Do not include any markdown, explanations, or extra text.
 """



    return prompt

def filter_description(description):
    """
    Filter the dataset description to remove explicit mentions of methods.

    Args:
        description (str): The original dataset description

    Returns:
        str: The filtered dataset description
    """

    prompt = """
    You are a helpful assistant. Help me filter this dataset description.

    The dataset description is: {}
    The dataset description should not provide information on what method one needs to consider in course of causal analysis.
    For example,
    Bad dataset description: "This dataset was collected by a group of social scientists interested in evaluating the effects of a job training program on job satisfaction among adults. They have gathered information on a group of individuals including their age, education level, gender, and socioeconomic status. The job training program was not mandatory and individuals were invited to participate, thus creating a randomized experiment with the invitation to the program as the instrument. The variable 'age' refers to the age of the participant, measured in years (continuous). 'Education_level' represents the highest level of education attained by the participant, treated as an ordinal categorical variable. 'Gender' is a binary variable where 1 represents male and 0 represents female. 'Socioeconomic_status' is a categorical variable that classifies individuals based on their income and occupation into groups such as low, middle, and high. 'Invitation_to_training' is a binary variable indicating whether a participant was invited to take part in the program (1 for invited, 0 for not invited). Finally, 'Job_satisfaction' is the outcome variable, measured on a scale where higher values indicate greater satisfaction with employment."
    Good dataset description: "This dataset was collected by a group of social scientists interested in evaluating the effects of a job training program on job satisfaction among adults. They have gathered information on a group of individuals including their age, education level, gender, and socioeconomic status. The job training program was not mandatory and individuals were invited to participate, thus creating a randomized experiment. The variable 'age' refers to the age of the participant, measured in years (continuous). 'Education_level' represents the highest level of education attained by the participant, treated as an ordinal categorical variable. 'Gender' is a binary variable where 1 represents male and 0 represents female. 'Socioeconomic_status' is a categorical variable that classifies individuals based on their income and occupation into groups such as low, middle, and high. 'Invitation_to_training' is a binary variable indicating whether a participant was invited to take part in the program (1 for invited, 0 for not invited). Finally, 'Job_satisfaction' is the outcome variable, measured on a scale where higher values indicate greater satisfaction with employment."

    If the dataset description is already filtered, return it as is.
    Return only the filtered dataset description. Do not say anything else.
    """

    return prompt.format(description)

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
    Bad question: "What is the effect of the training program on job outcomes considering education and experience?"
    Good question: "What is the effect of the training program on job outcomes?"

    If the question is already filtered, return it as is.
    Return only the filtered query. Do not say anything else.
    """

    return prompt.format(question)

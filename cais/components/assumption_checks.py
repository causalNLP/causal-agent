import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from rddensity import rddensity
from dowhy import CausalModel

from cais.config import get_llm_client
from langchain_core.messages import SystemMessage, HumanMessage

import logging
logger = logging.getLogger(__name__)

ASSUMPTION_PROMPT = """You are CausalAI an expert in Causal Analysis. To run a robust causal analysis, we need to check the assumptions we make about our data to see whether a proposed causal method is appropriate. You are tasked with running those checks.
"""

def format_prompt(user_prompt):
    return [
        SystemMessage(content=ASSUMPTION_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    

class Test:
    """A base class for testing the validity of assumptions of causal inference methods"""

    def __init__(self, data, query, description):

        self.data = data # the data in csv form 
        self.query = query # the query of interest
        self.description = description # the description of the dataset 
        self.llm = get_llm_client()

        self.encode_strings() # convert object (string) columns to numeric values; temporary fix for calculating sample statistics
    
    def run_tests(self):
        """
        Runs tests to validate the assumptions of the causal inference method 
        """

    def encode_strings(self):
        """
        Converts any strings into numeric values for statistical testing. 
        !!! This is a temporary fix; we will need to either move dataset cleaning earlier into the pipeline or come up with some other safety nets for statsmodels
        """
        
        string_columns = self.data.select_dtypes("object").columns.tolist()
        for column in string_columns:
            codes, _ = pd.factorize(self.data[column])
            self.data[column] = codes

    def invoke(self, prompt):
        return self.llm.invoke(format_prompt(prompt)).content

    def sutva_test(self, treat_var, outcome_var, description, question):
        """
        Test for the Stable Unit Treatment Value Assumption (SUTVA), which is perhaps the most common assumption in causal inference. 
        It has two main components: no interference and consistency. 
        This is again an untestable assumptions.
        """
        
        sutva_prompt = f"""I am considering using a causal inference method to estimate the causal effect of {treat_var} on {outcome_var}.
                        The goal is to answer the question: {question}. The dataset and its variables is described as follows: {description}.
                        I need to assess whether the Stable Unit Treatment Value Assumption (SUTVA) holds in this context.
                        SUTVA has two main components: no interference and consistency.
                        No interference means that the treatment status of one unit does not affect the potential outcome of another unit. 
                        Consistency means that the observed outcome for a unit under a particular treatment is equal to the potential outcome 
                        under that treatment i.e. if treatment a was assigned, then Y = Y(a)

                        Based on the description of the dataset and the variable, is it plausible that SUTVA holds here?
                        First, say yes or not. Then, explain your reasoning. 
                        """
        
        return self.invoke(sutva_prompt)


class IVTest(Test):
    """A class to test for IV assumptions in a dataset"""

    def __init__(self, data, query, description, treat_var, outcome_var, iv_var, 
                 covariates=None, is_rct=False):

        super().__init__(data, query, description)
        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.iv_var = iv_var
        self.covariates = covariates if covariates is not None else []
        self.is_rct = is_rct

        self.results = {}

    def relevance_test(self):
        """
        Test for relevance of the instrument i.e. whether or not the instrument is correlated with the treatment variable.
        This is based on the F-statistic from the first stage regression. 
        The rule of thum is that F-statistics should be greater than 10. We can either set this rule explicitly or 
        have an LLM interpret the results as a whole
        """

        first_stage_reg = smf.ols(f"{self.treat_var} ~ {self.iv_var} + {' + '.join(self.covariates)}", 
                                  data=self.data).fit()
        summary = first_stage_reg.summary()
        f_stat = summary.tables[1].data[1][3]  

        return f_stat
    
    def exclusion_test(self):
        """
        Test for the exclusion restriction i.e. it tests if the instrument affects the outcome variable only through the treatment variable. 
        This cannot be tested directly, and is the most controversial assumption of IV analysis. We can argue this quanlitatively. 
        """

        sample_prompt = f"""I am considering using instrumental variable analysis to estimate the causal effect of {self.treat_var} on 
                            {self.outcome_var} using {self.iv_var} as an instrument. The goal is to answer the query: {self.query}.
                            The dataset and its variables is described as follows: {self.description}.

                            I need to assess whether the instrument {self.iv_var} satisfies the exclusion restriction, which states that 
                            the instrument affects the outcome variable {self.outcome_var} only through the treatment variable {self.treat_var} i.e.
                            there is no direct effect of {self.iv_var} on {self.outcome_var}.
                            Based on the description of the dataset and variables, does it seem plausible that {self.iv_var} satisfies the exclusion restriction?

                            Explain your reasoning. Be critical of your assessment. 
                        """
        
        result = self.invoke(sample_prompt) ## whatever the function is to invoke an LLM. 

        return result
    
    def unconfoundedness_test(self):
        """
        Test for unconfoundedness of the instrument i.e. whether or not the instrument is independent of unobserved confounders that affect both the treatment and outcome variables.
        This cannot be tested directly, and is another controversial assumption of IV analysis. We can argue this quanlitatively. 
        """

        sample_prompt = f"""I am considering using instrumental variable analysis to estimate the causal effect of {self.treat_var} on 
                            {self.outcome_var} using {self.iv_var} as an instrument. The goal is to answer the query: {self.query}.
                            The dataset and its variables is described as follows: {self.description}.

                            I need to assess whether the instrument {self.iv_var} satisfies the unconfoundedness assumption, which states that 
                            the instrument is independent of unobserved confounders that affect both the treatment variable {self.treat_var} 
                            and the outcome variable {self.outcome_var}.

                            Based on the description of the dataset and variables, does it seem plausible that {self.iv_var} satisfies the unconfoundedness assumption?

                            Explain your reasoning. Be critical of your assessment. 
                        """
        
        result = self.invoke(sample_prompt) ## whatever the function is to invoke an LLM. 

        return result
    
    def defier_test(self):
        """
        Test if the no-defier assumption holds i.e. there are no individuals who would do the opposite of their assigned treatment 
        Note that, here treatment assigned is not the same as actual treatment uptake. Actual treatment is opposed to assigned treatment.
        This is again an untestable assumption, and we can argue this qualitatively. In general, qualitative arguments for this assumption
        is easy to justify compared to exclusion restriction and unconfoundedness.
        """

        sample_prompt = f"""I am considering using instrumental variable analysis to estimate the causal effect of {self.treat_var} on 
                            {self.outcome_var} using {self.iv_var} as an instrument. The goal is to answer the query: {self.query}.
                            The dataset and its variables is described as follows: {self.description}.

                            I need to assess whether the instrument {self.iv_var} satisfies the no-defier assumption, which states that 
                            there are no individuals who would always do the opposite of their assigned treatment based on the instrument.

                            Based on the description of the dataset and variables, does it seem plausible that the the no-defier assumption 
                            is satisfied here?

                            Explain your reasoning. Be critical of your assessment. 
                        """
        
        result = self.invoke(sample_prompt) ## whatever the function is to invoke an LLM. 

        return result


    def run_tests(self):
        """
        Runs the IV assumptions tests
        """

        # Test for relevance 
        relevance_result = self.relevance_test()

        ## Test for exclusion restriction 
        exclusion_result = self.exclusion_test()

        ## uncounfoundedness 
        unconfoundedness_result = self.unconfoundedness_test()

        ## test for defier (this is useful only in RCTs where non-compliance is an issue)
        if self.is_rct:
            defier_result = "Not applicable since this is not an RCT"
        else:
            defier_result = self.defier_test()
        
        final_llm_prompt = f"""You need to assess the validity of the instrumental variable {self.iv_var} for estimating the 
                                causal effect of {self.treat_var} on {self.outcome_var}. 

                                The goal is to answer the query: {self.query}. For reference, here is the description of the dataset and its variables: {self.description}.
                                Here are the results for each of the IV assumptions tests:

                                1. F-stat from Relevance test: {relevance_result}
                                2. Exclusion restriction test: {exclusion_result}
                                3. Unconfoundedness test: {unconfoundedness_result}
                                4. No-defier assumption test: {defier_result}

                                Based on these results, provide whether the instrumental variable regression is valid or not? 
                                First, say yes or no. Then, justify your answer. 
                            """
        
        final_result = self.invoke(final_llm_prompt)
        return final_result


class DiDTest(Test):
    """A class to test for DiD assumptions in panel datasets"""

    def __init__(self, data, query, description, treat_var, outcome_var, time_var, group_var, 
                    covariates=None):

        super().__init__(data, query, description)
        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.time_var = time_var
        self.group_var = group_var
        self.covariates = covariates if covariates is not None else []
    
    def no_anticipation_test(self):
        """
        Test for no-anticipation assumption in DiD i.e. effect of the treatment does not occur before the treatment is actually applied. 
        This is again an untestable assumption.
        """

        no_anticipation_prompt = f"""I am considering using Difference-in-Differences (DiD) to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                    The goal is to answer the question: {self.query}. The dataset and its variables is described as follows: {self.description}.
                                    Likewise, the time variable is {self.time_var} and the group variable is {self.group_var}.

                                    For the validity of DiD, I need to assess whether the no-anticipation assumption holds or not in this context. 
                                    No-anticipation means that the effect of the treatment does not occur before the treatment is actually applied.
                                    Based on the description of the dataset and the variable, is it plausible that the no-anticipation assumption holds here?
                                    First, say yes or not. Then, explain your reasoning.
                                    """
        result = self.invoke(no_anticipation_prompt)

        return result
    
    def parallel_trends_test(self):
        """
        Test for parallel trends assumption in DiD i.e. in the absence of treatment, the average change in outcome for treated and control groups would have been the same over time.
        If we have data available for multiple time periods before the treatment, we can test this visually. This involves plotting the outcomes for 
        variables over time for both the treated and control groups to see if the outcomes are roughly parallel. 
        This test is easier for the classical DiD setup with two groups (treated and control) and two time periods (pre and post treatment).
        If we know the exact nature of the data, we can run this ourselves. Otherwise, we might have to invoke an LLM to help with this. 

        Moreover, if we don't have data for periods before treatment, we cannot test this assumption. 
        For now let's argue qualitatively using an LLM.
        """

        ## later we can code + ask LLM to visually inspect the trends
        parallel_trends_prompt = f"""I am considering using Difference-in-Differences (DiD) to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                    The goal is to answer the question: {self.query}. The dataset and its variables is described as follows: {self.description}.
                                    Likewise, the time variable is {self.time_var}. 
                                    Based on the description, is it plausible that parallel trends assumption holds here?
                                    Parallel trends means that in the absence of treatment, the change in outcome for the treated and control groups 
                                    before and after treatment would have been the same over time. First, say yes or not. 
                                    Then,justify your answer.
                                    """
        parallel_trends_result = self.invoke(parallel_trends_prompt)

        return parallel_trends_result
    
    def run_tests(self):
        """
        Runs tests to validate the assumptions of DiD
        """

        # 1. SUTVA test
        sutva_result = self.sutva_test(self.treat_var, self.outcome_var, self.description, self.query)

        # 2. No-anticipation test
        no_anticipation_result = self.no_anticipation_test()

        # 3. Parallel trends test
        parallel_trends_result = self.parallel_trends_test()

        final_llm_prompt = f"""You need to check if differeence-in-differences (DiD) is a valid method to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                The results of the analysis will be used to answer the query: {self.query}. 
                                For reference, here is the description of the dataset and its variables: {self.description}.

                                Here are the results for each of the DiD assumptions tests:
                                1. SUTVA test: {sutva_result}
                                2. No-anticipation test: {no_anticipation_result}
                                3. Parallel trends test: {parallel_trends_result}
                                Based on these results, provide whether the difference-in-differences method is valid or not?
                                First, say yes or no. Then, justify your answer.
                            """
        
        return self.invoke(final_llm_prompt)

class RDDTest(Test):
    """A class to test for RDD assumptions in datasets"""

    def __init__(self, data, query, description, treat_var, outcome_var, running_var, cutoff, 
                    covariates=None):

        super().__init__(data, query, description)
        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.running_var = running_var
        self.cutoff = cutoff
        self.covariates = covariates if covariates is not None else []

    
    def visual_inspection_test(self):
        """This test visually inspects the running variable around the cutoff to check for the presence of discontinuity. 
            For RDD, to be valid, there should be a discontinuity in the outcome values around the cutoff point"""
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        treat_data = self.data[self.data[self.running_var] > self.cutoff]
        control_data = self.data[self.data[self.running_var] < self.cutoff]
        ax.scatter(treat_data[self.running_var], treat_data[self.outcome_var], color='blue', label='Treated')
        ax.scatter(control_data[self.running_var], control_data[self.outcome_var], color='red', label='Control')
        ax.axvline(x=self.cutoff, color='black', linestyle='--', label='Cutoff')
        ax.set_xlabel(self.running_var)
        ax.set_ylabel(self.outcome_var)
        ax.set_title('Visual Inspection of RDD')
        
        ax.legend()

        llm_prompt = f"""I am considering using Regression Disconntinuity Design in the process of answering the query: {self.query}.
                        The dataset and its variables is described as follows: {self.description}.
                        The running variable is {self.running_var} with a cutoff at {self.cutoff}, and the outcome variable is {self.outcome_var}.
                        I have plotted the outcome against the running variable. Based on the plot, do you observe a discontinuity in the outcome variable around the cutoff point?
                        First, answer yes or no. Then explain your reasoning.
                    """
        ## if calling LLM with image is not possible, we can either skip this or convey some information about the discontinuity numerically.
        result = self.invoke_image(llm_prompt, ax)

        return result
    
    def mccrary_test(self):
        """Note that for RDD, we need treatment assignment to be as good as random around the cutoff. 
        Hence, we can check for manipulation of the running variable about the cutoff using the McCrary test """

        test = rddensity(self.data[self.running_var], self.cutoff)

        return test 

    
    
    def run_tests(self):
        """
        Runs tests to validate the assumptions of RDD
        """

        ## 1 Visual inspection test
        visual_inspection_result = self.visual_inspection_test()

        ## 2. McCrary test
        mccrary_result = self.mccrary_test()

        final_llm_prompt = f"""You need to check if regression discontinuity design (RDD) is a valid method to answer the query: {self.query}. 
                            For reference, here is the description of the dataset and its variables: {self.description}.
                            The running variable is {self.running_var} with a cutoff at {self.cutoff}, and the outcome variable is {self.outcome_var}.
                                Here are the results for each of the RDD assumptions tests:
                                1. Visual inspection test: {visual_inspection_result}
                                2. McCrary test: {mccrary_result}
                                Based on these results, provide whether the regression discontinuity design is valid or not?
                                First, say yes or no. Then, justify your answer.
                        """
        
        return self.invoke(final_llm_prompt)
    
class ObervationalTest(Test):
    """
        A class to test for assumptions of observational causal inference methods in datasets. The focus is on SUTVA, conditional ignorability, and positivity. 
        Our focus is on propensity score based methods at the moment. This includes PS matching and IPW, and OLS for randomized data. 
    """


    def __init__(self, data, query, description, treat_var, outcome_var, confounders=None):

        super().__init__(data, query, description)
        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.confounders = confounders if confounders is not None else []
    
    def no_confounder_test(self):
        """
        In this test, we qualitatively argue whether or not there are any uboserved confounders that affect both treatment and outcome variable. 
        This is again an untestable assumption. We will first argue this qualitatively. 
        """

        no_confounder_prompt = f"""I am considering using an observational causal inference method to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                    The goal is to answer the question: {self.query}. The dataset and its variables is described as follows: {self.description}.
                                    I need to assess whether there are any unobserved confounders that affect both the treatment variable {self.treat_var} and the outcome variable {self.outcome_var}.
                                    Based on the description of the dataset and variables, does it seem plausible that there are no unobserved confounders here? 
                                    The confounders under consideration are: {', '.join(self.confounders)}.
                                    First, say yes or no. Then, explain your reasoning.
                                    """
        result = self.invoke(no_confounder_prompt)

        return result
    
    def covariate_balance_test(self):
        """
        In this test, we will test for covariate balance between the treated and control groups. 
        This is to assess whether or not the observed confounders are balanced between the treated and control groups.  
        """

        balance_results = {}
        treated = self.data[self.data[self.treat_var] == 1]
        control = self.data[self.data[self.treat_var] == 0]

        for confounder in self.confounders:
            treated_mean = treated[confounder].mean()
            control_mean = control[confounder].mean()
            treated_std = treated[confounder].std()
            control_std = control[confounder].std()
            smd = abs(treated_mean - control_mean) / np.sqrt((treated_std ** 2 + control_std ** 2) / 2)
            balance_results[confounder] = smd
        
        balance_string = ", ".join([f"{confounder}: {smd:.3f}" for confounder, smd in balance_results.items()])

        balance_prompt = f"""I am considering using an causal inference method for observational studies to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                Thus, we need to check if conditional ignorability is plausible or not. 
                                The main goal is to answer the query: {self.query}. The dataset and its variables is described as follows: {self.description}.
                                To assess the plausibility of conditional ignorability, we can check for the distribution of observed confounders. 
                                To this end, I have calculated the standardized mean difference (SMD) for each of the observed confounders between the treated and control groups.
                                The SMD values for the confounders are as follows: {balance_string}.
                            
                                Based on these results, does it seem plausible that conditional ignorability holds here?
                                First, say yes or no. Then, explain your reasoning.
                        """
        
        result = self.invoke(balance_prompt)

        return result
    
    def positivity_test(self):
        """
        In this test, we will check for positivity assumption. This means that given the observed coufounders, each unit has a positive probability of receiving each level of treatment. 
            We will look at the distribution of propensity scores. Additionally, this also checks for propensity score overlap. 
            We do not want only one group to have propensity scores close to 0 or 1. Ideally, we would do this visually, but for now we will use a simple heursitic. 
        """

        treated_data = self.data[self.data[self.treat_var] == 1]
        control_data = self.data[self.data[self.treat_var] == 0]

        log_model = LogisticRegression()
        log_model.fit(self.data[self.confounders], self.data[self.treat_var])
        propensity_scores = log_model.predict_proba(self.data[self.confounders])[:, 1]

        treated_propensity = propensity_scores[self.data[self.treat_var] == 1]
        control_propensity = propensity_scores[self.data[self.treat_var] == 0]

        mean_treated_propensity = treated_propensity.mean()
        mean_control_propensity = control_propensity.mean()

        std_treated_propensity = treated_propensity.std()
        std_control_propensity = control_propensity.std()
        var = (std_treated_propensity ** 2 + std_control_propensity ** 2) / 2

        smd_propensity = abs(mean_treated_propensity - mean_control_propensity) / np.sqrt(var)

        propensity_prompt = f"""I am considering using an causal inference method for observational studies to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                Thus, we need to check if positivity assumption is plausible or not. We will do this using propensity scores. 
                                At the same time, we will also check for propensity score overlap.
                                The main goal is to answer the query: {self.query}. The dataset and its variables is described as follows: {self.description}.
                                We compute propensity scores using a logistic regression model with the following observed confounders: {', '.join(self.confounders)}.
                                The standardized mean difference (SMD) of the propensity scores between the treated and control groups is {smd_propensity:.3f}.
                                The distribution of propensity scores for the treatment group is,
                                mean: {mean_treated_propensity:.3f}, std: {std_treated_propensity:.3f}, max: {treated_propensity.max():.3f}, min: {treated_propensity.min():.3f}.
                                median: {np.median(treated_propensity):.3f}.
                                The distribution of propensity scores for the control group is,
                                mean: {mean_control_propensity:.3f}, std: {std_control_propensity:.3f}, max: {control_propensity.max():.3f}, min: {control_propensity.min():.3f}.
                                median: {np.median(control_propensity):.3f}.
                                Based on these results, does it seem plausible that positivity assumption and propensity score overlap holds here?
                                First, say yes or no. Then, explain your reasoning.
                        """
        result = self.invoke(propensity_prompt)

        

        
    def run_tests(self):
        
        ## 1. SUTVA test
        sutva_result = self.sutva_test(self.treat_var, self.outcome_var, self.description, self.query)

        ## 2. No unobserved confounder test
        no_confounder_result = self.no_confounder_test()

        ## 3. Covariate balance test
        covariate_balance_result = self.covariate_balance_test()

        ## 4. Positivity test
        positivity_result = self.positivity_test()


        final_llm_prompt = f"""You need to check if an observational causal inference method is valid to estimate the causal effect of {self.treat_var} on {self.outcome_var}.
                                The results of the analysis will be used to answer the query: {self.query}. 
                                For reference, here is the description of the dataset and its variables: {self.description}.
                                Here are the results for each of the assumptions tests:
                                1. SUTVA test: {sutva_result}
                                2. No unobserved confounder test: {no_confounder_result}
                                3. Covariate balance test: {covariate_balance_result}
                                4. Positivity test: {positivity_result}
                                Based on these results, provide whether the observational causal inference method is valid or not?
                                First, say yes or no. Then, justify your answer.
                            """
        
        return self.invoke(final_llm_prompt)
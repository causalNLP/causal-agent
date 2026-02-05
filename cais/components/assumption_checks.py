import numpy as np 
import statsmodels.formula.api as smf
from rdddensity import rddensity
from dowhy import CausalModel
from cais.config import get_llm_client
from langchain_core.messages import SystemMessage, HumanMessage

ASSUMPTION_PROMPT = """You are CausalAI an expert in Causal Analysis. To run a robust causal analysis, we need to check the assumptions we make about our data to see whether a proposed causal method is appropriate. You are tasked with running those checks.
"""

def format_prompt(user_prompt):
    return [
        SystemMessage(content=ASSUMPTION_PROMPT),
        HumanMessage(content=user_prompt)
    ]

def sutva_test(treat_var, outcome_var, description, question):
    """
    Test for the Stable Unit Treatment Value Assumption (SUTVA), which is perhaps the most common assumption in causal inference. 
    It has two main components: no interference and consistency. 
    This is again an untestable assumptions.
    """

    llm = get_llm_client()

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
    
    result = llm.invoke(format_prompt(sutva_prompt))

    return result
    

class Test:
    """A base class for testing the validity of assumptions of causal inference methods"""

    def __init__(self, data, query, description):

        self.data = data # the data in csv form 
        self.query = query # the query of interest
        self.description = description # the description of the dataset 
    
    def run_tests(self):
        """
        Runs tests to validate the assumptions of the causal inference method 
        """


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

        self.llm = get_llm_client()

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
        
        result = self.llm.invoke(format_prompt(sample_prompt)) ## whatever the function is to invoke an LLM. 

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
        
        result = self.llm.invoke(format_prompt(sample_prompt)) ## whatever the function is to invoke an LLM. 

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
        
        result = self.llm.invoke(format_prompt(sample_prompt)) ## whatever the function is to invoke an LLM. 

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
        
        final_result = self.llm.invoke(format_prompt(final_llm_prompt))

class RCTTest(Test):

    def __init__(self, data, query, description):
        super().__init__(data, query, description)


    def run_tests(self):
        return super().run_tests()
    
class RDDTest(Test):

    def __init__(self, data, query, description):
        super().__init__(data, query, description)


    def run_tests(self):
        return super().run_tests()
    

class MatchingTest(Test):

    def __init__(self, data, query, description):
        super().__init__(data, query, description)


    def run_tests(self):
        return super().run_tests()
    

class DiDTest(Test):

    def __init__(self, data, query, description):
        super().__init__(data, query, description)


    def run_tests(self):
        return super().run_tests()
    

from doubleml.datasets import make_did_SZ2020
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.datasets import make_iivm_data
from linearmodels.iv import IV2SLS
from dowhy import CausalModel
from dowhy import datasets as dset
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class DataGenerator:
    """
    Base class for generating synthetic data

    Attributes:
        n_observations (int): Number of observations
        n_continuous_covars (int): Number of covariates
        n_treatments (int): Number of treatments
        true_effect (float): True effect size
        seed (int): Random seed for reproducibility
        data (pd.DataFrame): Generated data
        info (dict): Dictionary to store additional information about the data
        method (str): the causal inference method assocated with the synthetic
    """
    
    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, 
                 covar = None, n_treatments=1, true_effect=0 ,seed=111, heterogeneity=0):
        
        np.random.seed(seed)
        self.n_observations = n_observations
        self.n_continuous_covars = n_continuous_covars
        self.n_covars = n_continuous_covars + n_binary_covars
        self.n_treatments = n_treatments
        self.n_binary_covars = n_binary_covars
        self.data = None
        self.seed = seed
        self.true_effect = true_effect
        self.method = None
        self.mean = mean
        self.covar = covar
        if mean is None:
            self.mean = np.random.randint(3, 20, size=self.n_continuous_covars)
        if self.covar is None:
            self.covar = np.identity(self.n_continuous_covars)
        self.heterogeneity = heterogeneity
    
    def generate_data(self):
        """
        Generate synthetic data
        """
        raise NotImplementedError("Invoke the method in the subclass")
    
    def save_data(self, folder, filename):
        """
        Saves the generated data as a CSV file
        Args:
            folder (str): path to the folder where the data is saved
            filename (str): name of the file
        """

        if self.data is None:
            raise ValueError("Data not generated yet. Please generate data first.")
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.csv'):
            filename += '.csv'
        self.data.to_csv(path / filename, index=False)

    def test_data(self, print_=False):
        """
        Test the generated data
        """

        raise NotImplementedError("This method should be overridden by subclasses")
    
    def generate_covariates(self):
        """
        Generate covariates
        """
        X_c = np.random.multivariate_normal(mean=self.mean, cov=self.covar, 
                                                   size=self.n_observations)
        p = np.random.uniform(0.3, 0.7)
        X_b = np.random.binomial(1, p, size=(self.n_observations, self.n_binary_covars)).astype(int)
        covariates = np.hstack((X_c, X_b))
        covariates = covariates.astype(int)

        return covariates
    
class MultiTreatRCTGenerator(DataGenerator):
    """
    Base class for generating synthetic data for multi-treatment RCTs
    """
    def __init__(self, n_observations, n_continuous_covars, n_treatments, n_binary_covars=2, 
                 mean=None, covar=None, true_effect=1.0, true_effect_vec = None, 
                 seed=111, heterogeneity=0):
        
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars,
                         mean=mean, covar=covar, true_effect=true_effect, seed=seed,
                         heterogeneity=heterogeneity, n_treatments=n_treatments)
        
        self.method = "MultiTreatRCT"
        self.true_effect_vec = true_effect_vec
        if true_effect_vec is None:
            self.true_effect_vec = np.zeros(n_treatments)
            for i in range(1, n_treatments):
                self.true_effect_vec[i] = self.true_effect
    
    def generate_data(self):

        X = self.generate_covariates()
        cols = [f"X{i+1}" for i in range(self.n_covars)]
        df = pd.DataFrame(X, columns=cols)
        
        
        df['D'] = np.random.randint(0, self.n_treatments+1, size=self.n_observations)
        vec = np.random.uniform(0, 1, size=self.n_covars)
        intercept = np.random.normal(50, 3)
        noise = np.random.normal(0, 1, size=self.n_observations)

        # Apply appropriate treatment effect per treatment arm
        treatment_effects = np.array(self.true_effect_vec)
        df['treat_effect'] = treatment_effects[df['D']]

        df['Y'] = intercept + X.dot(vec) + df['treat_effect'] + noise

        df.drop(columns='treat_effect', inplace=True)
        self.data = df

        return df
    

    def test_data(self, print_=False):
        if self.data is None:
            raise ValueError("Data not generated yet. Please generate data first.")
        
        model = smf.ols('Y ~ C(D)', data=self.data).fit()

        result = model.summary()
        if print_:
            print(result)
        return result

class ObservationalDataGenerator(DataGenerator):
    """
    Generate synthetic data for observational studies
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, covar=None, 
                 true_effect=1.0, seed=111, heterogeneity=0):

        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, mean=mean, covar=covar, 
                         true_effect=true_effect, seed=seed, heterogeneity=heterogeneity)
        self.heterogeneity = heterogeneity

    def generate_data(self):

        X = self.generate_covariates()

        cols = [f"X{i+1}" for i in range(self.n_covars)]
        df = pd.DataFrame(X, columns=cols)
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

        vec1 = np.random.normal(0, 0.5, size=self.n_covars)
        lin = X_norm @ vec1 + np.random.normal(0, 1, self.n_observations)
        ps = 1 / (1 + np.exp(-lin))
        ps = np.clip(ps, 1e-3, 1 -1e-3) 
        df['D'] = np.random.binomial(1, ps).astype(int)  
        vec2 = np.random.normal(0, 0.5, size=self.n_covars)
        intercept = np.random.normal(50, 3)
        noise = np.random.normal(0, 1, size=self.n_observations)
        df['Y'] = intercept + X @ vec2 + self.true_effect * df['D'] + noise

        self.propensity = ps
        self.weights = np.where(df['D'] == 1, 1 / ps, 1 / (1 - ps))
        self.data = df

        return self.data

class PSMGenerator(ObservationalDataGenerator):
    """
    Generate synthetic data for Propensity Score Matching (PSM)
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, covar=None, 
                 true_effect=1.0, seed=111, heterogeneity=0):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, mean=mean, covar=covar,
                         true_effect=true_effect, seed=seed, heterogeneity=heterogeneity)
        self.method = "PSM"

    def test_data(self, print_=False):
        """
        Test the generated data
        """
        if self.data is None:
            raise ValueError("Data not generated yet. Please generate data first.")
            
        lr = LogisticRegression(solver='lbfgs')
        X = self.data[[f"X{i+1}" for i in range(self.n_covars)]]
        lr.fit(X, self.data['D'])
        ps_hat = lr.predict_proba(X)[:, 1]
        treated = self.data[self.data['D'] == 1]
        control = self.data[self.data['D'] == 0]
        match_idxs = [np.abs(ps_hat[control.index] - ps_hat[i]).argmin() for i in treated.index]
        matches = control.iloc[match_idxs]
        att = treated['Y'].mean() - matches['Y'].mean()

        result = f"Estimated ATT (matching): {att:.3f} | True: {self.true_effect}"
        if print_:
            print(result)
        return result

class PSWGenerator(ObservationalDataGenerator):
    """
    Generate synthetic data for Propensity Score Weighting (PSW)
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, covar=None, 
                 true_effect=1.0, seed=111, heterogeneity=0):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, mean=mean, covar=covar,
                         true_effect=true_effect, seed=seed, heterogeneity=heterogeneity)
        self.method = "PSW"

    def test_data(self, print_=False):
        """
        Test the generated data
        """
        if self.data is None:
            raise ValueError("Data not generated yet. Please generate data first.")

        df = self.data.copy()
        D = df['D']
        Y = df['Y']
        
        treated = D == 1
        control = D == 0

        w = np.zeros(self.n_observations)
        w[control] = self.propensity[control]# / (1 - self.propensity[control])
        w[treated] = 1 
        
        Y1 = Y[treated].mean()
        Y0_weighted = np.average(Y[control], weights=w[control])
        
        att = Y1 - Y0_weighted
        result = f"Estimated ATE (IPW): {att:.3f} | True: {self.true_effect}"
        if print_:
            print(result)

        return result

    
class RCTGenerator(DataGenerator):
    """
    Generate synthetic data for Randomized Controlled Trials (RCT)
    """
    
    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, 
                covar=None, true_effect=1.0, seed=111, heterogeneity=0):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, 
                             mean=mean, covar=covar, true_effect=true_effect, seed=seed, 
                             heterogeneity=heterogeneity)
        self.method = "RCT"

    def generate_data(self):
        
        X = self.generate_covariates()
        cols = [f"X{i+1}" for i in range(self.n_covars)]
        df = pd.DataFrame(X, columns=cols)
        df['D'] = np.random.binomial(1, 0.5, size=self.n_observations)
        vec = np.random.uniform(0, 1, size=self.n_covars)
        intercept = np.random.normal(50, 3)
        noise = np.random.normal(0, 1, size=self.n_observations)
        df['Y'] = (intercept + X.dot(vec) + self.true_effect * df['D'] + noise)
        self.data = df

    def test_data(self, print=False):
        if self.data is None:
            raise ValueError("Data not generated yet. Please generate data first.")
        model = smf.ols('Y ~ D', data=self.data).fit()
        result = model.summary()
        if print:
            print(result)
        est = model.params['D']
        conf_int = model.conf_int().loc['D']
        result = f"TRUE ATE: {self.true_effect:.3f}, ESTIMATED ATE: {est:.3f}, \
            95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        
        return result

class IVGenerator(DataGenerator):
    """
    Generate synthetic data for Instrumental Variables (IV) analysis
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, 
                 covar=None, true_effect=1.0, seed=111, heterogeneity=0, alpha=0.5, encouragement=False):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, 
                         mean=mean, covar=covar, true_effect=true_effect, seed=seed, 
                         heterogeneity=heterogeneity)
        self.method = "IV"
        self.alpha = alpha
        self.encouragement = encouragement

    def generate_data(self):
        X = self.generate_covariates()

        mean = np.random.randint(8, 13)
        Z = np.random.normal(mean, 2, size=self.n_observations).astype(int)
        vec1 = np.random.normal(0, 0.5, size=self.n_covars)
        intercept1 = np.random.normal(30, 2)
        D = self.alpha * Z + X @ vec1 + np.random.normal(size=self.n_observations) + intercept1
        if self.encouragement:
            D = (D > np.mean(D)).astype(int)
        D = D.astype(int)  

        intercept2 = np.random.normal(50, 3)
        vec2 = np.random.normal(0, 0.5, size=self.n_covars)
        Y = self.true_effect * D + X @ vec2 + np.random.normal(size=self.n_observations) + intercept2
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(self.n_covars)])
        df['Z'] = Z
        df['D'] = D
        df['Y'] = Y
        self.data = df

        return self.data

    
    def test_data(self, print_=False):

        if self.data is None:
            raise ValueError("Data not generated yet.")
        model = IV2SLS.from_formula('Y ~ 1 + [D ~ Z]', data=self.data).fit()
        est = model.params['D']
        conf_int = model.conf_int().loc['D']
        result = f"TRUE LATE: {self.true_effect:.3f}, ESTIMATED LATE: {est:.3f}, \
            95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        if print_:
            print(result)
        return result

class RDDGenerator(DataGenerator):
    """
    Generate synthetic data for Regression Discontinuity Design (RDD)
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, mean=None, plot=False,
                 covar=None, true_effect=1.0, seed=111, heterogeneity=0, cutoff=10, bandwidth=0.1):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars,
                         mean=mean, covar=covar, true_effect=true_effect, seed=seed,
                         heterogeneity=heterogeneity)
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.method = "RDD"
        self.plot=plot

        print("self.plot", self.plot)

    def generate_data(self):
        
        X = self.generate_covariates()
        cols = [f"X{i+1}" for i in range(self.n_covars)]
        df = pd.DataFrame(X, columns=cols)
        
        df['running_X'] = np.random.normal(0, 2, size=self.n_observations) + self.cutoff
        
  
        df['D'] = (df['running_X'] >= self.cutoff).astype(int)
        
        intercept = 10
        coeffs = np.random.normal(0, 0.1, size=self.n_covars) 
        
        m_below = 1.5 
        m_above = 0.8  
        
        df['running_centered'] = df['running_X'] - self.cutoff
        # Use centered version for slope
        df["Y"] = (intercept + self.true_effect * df['D'] + m_below * df['running_centered'] * (1 - df['D']) +  \
                   m_above * df['running_centered'] * df['D'] +  X @ coeffs + np.random.normal(0, 0.5, size=self.n_observations))
        
        if self.plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[df['D']==0]['running_X'], df[df['D']==0]['Y'], 
                   alpha=0.5, label='Control', color='blue')
            plt.scatter(df[df['D']==1]['running_X'], df[df['D']==1]['Y'], 
                   alpha=0.5, label='Treatment', color='red')
            plt.axvline(self.cutoff, color='black', linestyle='--', label='Cutoff')
            plt.show()

        self.data = df[[cols for cols in df.columns if cols != 'running_centered']]

        return self.data 


    def test_data(self, print_=False):

        if self.data is None:
            raise ValueError("Data not generated yet.")
        df = self.data.copy()
        df['running_adj'] = df['running_X'].astype(float) - self.cutoff
        df = df[np.abs(df['running_adj']) <= self.bandwidth].copy()
        print(df.shape)
        model = smf.ols('Y ~ D + running_adj + D:running_adj', data=df).fit()
        est = model.params['D']
        conf_int = model.conf_int().loc['D']

        result = f"TRUE LATE: {self.true_effect:.3f}, ESTIMATED LATE: {est:.3f}, \
              95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
        if print_:
            print(result)
        return result


class DiDGenerator(DataGenerator):
    """
    Generate synthetic data for Difference-in-Differences (DiD) analysis
    """

    def __init__(self, n_observations, n_continuous_covars, n_binary_covars=2, n_periods=2, 
                 mean=None, covar=None, true_effect=1.0, seed=111, heterogeneity=0):
        super().__init__(n_observations, n_continuous_covars, n_binary_covars=n_binary_covars, 
                         mean=mean, covar=covar, true_effect=true_effect, 
                         seed=seed, heterogeneity=heterogeneity)
        
        self.method = "DiD"
        self.n_periods = n_periods
    
    def canonical_did_model(self):

        frac_treated = np.random.uniform(0.35, 0.65)
        n_treated = int(frac_treated * self.n_observations)
        unit_ids = np.arange(self.n_observations)
        treatment_status = np.zeros(self.n_observations, dtype=int)
        treatment_status[:n_treated] = 1
        np.random.shuffle(treatment_status)

        X = self.generate_covariates()
        cols = [f"X{i+1}" for i in range(self.n_covars)]
        covar_df = pd.DataFrame(X, columns=cols)


        vec = np.random.normal(0, 0.1, size=self.n_covars)

        intercept = np.random.normal(50, 3)
        treat_effect = np.random.normal(0, 1)
        time_effect = np.random.normal(0, 1)

        # Outcomes
        covar_term = X @ vec
        pre_noise = np.random.normal(0, 1, self.n_observations)
        pre_outcome = intercept + covar_term + pre_noise + treat_effect * treatment_status
        pre_data = pd.DataFrame({'unit_id': unit_ids, 'post': 0, 'D': treatment_status,
                                 'Y': pre_outcome})
        post_noise = np.random.normal(0, 1, self.n_observations)
        post_outcome = (intercept + time_effect + covar_term + self.true_effect * treatment_status
                        + treat_effect * treatment_status + post_noise)

        post_data = pd.DataFrame({'unit_id': unit_ids, 'post': 1, 'D': treatment_status,
                                  'Y': post_outcome})

        df = pd.concat([pre_data, post_data], ignore_index=True)

        df = df.merge(covar_df, left_on="unit_id", right_index=True)

        return df[['unit_id', 'post', 'D', 'Y'] + cols]
    
    def twfe_model(self):
        """
        Generate panel data for Two-Way Fixed Effects DiD model.
        """
        
        # Setup units and time
        frac_treated = np.random.uniform(0.35, 0.65)
        unit_ids = np.arange(1, self.n_observations + 1)
        time_periods = np.arange(0, self.n_periods)

        df = pd.DataFrame([(i, t) for i in unit_ids for t in time_periods],
                      columns=["unit", "time"])

        # Generate covariates (constant over time per unit)
        X = self.generate_covariates()
        for j in range(self.n_covars):
            df[f"X{j+1}"] = np.repeat(X[:, j], self.n_periods)

        # Assign treatment timing
        n_treated = int(frac_treated * self.n_observations)
        treated_units = np.random.choice(unit_ids, size=n_treated, replace=False)
        treatment_start = {unit: np.random.randint(1, self.n_periods) for unit in treated_units}

        # Construct treatment indicator: 1 if treated and time >= treatment start
        df["treat_post"] = df.apply(lambda row: int(row["unit"] in treatment_start and 
                                                    row["time"] >= treatment_start[row["unit"]]),axis=1)

        # Fixed effects
        unit_effects = dict(zip(unit_ids, np.random.normal(0, 1.0, self.n_observations)))
        time_effects = dict(zip(time_periods, np.random.normal(0, 1, len(time_periods))))
        df["unit_fe"] = df["unit"].map(unit_effects)
        df["time_fe"] = df["time"].map(time_effects)

        # Covariate effects
        if not hasattr(self, "covar_effects") or self.covar_effects is None:
            covar_effects = np.random.normal(0, 0.1, self.n_covars)
        else:
            covar_effects = self.covar_effects

        X_matrix = df[[f"X{j+1}" for j in range(self.n_covars)]].values
        covar_term = X_matrix @ covar_effects

        # Outcome
        intercept = np.random.normal(50, 3)
        noise = np.random.normal(0, 1, len(df))

        df["Y"] = intercept + covar_term + df["unit_fe"] + df["time_fe"] + self.true_effect * df["treat_post"] + noise

        final_df = df[["unit", "time", "treat_post", "Y"] + [f"X{j+1}" for j in range(self.n_covars)]]
        final_df = final_df.rename(columns={"time": "year", "treat_post": "D"})
    
        return final_df


    def generate_data(self):
        if self.n_periods == 2:
            self.data = self.canonical_did_model()
        else:
            self.data = self.twfe_model()

        return self.data


    def test_data(self, print_=False):
        estimated_att = None
        if self.data is None:
            raise ValueError("Data not generated yet.")
        if self.n_periods == 2:
            print("Testing canonical DiD model")
            model = smf.ols('Y ~ D * post', data=self.data).fit()
            estimated_att = model.params['D:post']
            conf_int = model.conf_int().loc['D:post']
        else:
            print("Testing TWFE model")
            model = smf.ols('Y ~ treat + C(unit) + C(year)', data=self.data).fit()
            estimated_att = model.params['treat']
            conf_int = model.conf_int().loc['treat']

        result = "TRUE ATT: {:.3f}, EMPRICAL ATT:{:.3f}\nCONFIDENCE INTERVAL:{}".format(
                 self.true_effect, estimated_att, conf_int)
        if print_:
            print(result)
        return result
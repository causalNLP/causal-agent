## This file contains the functions that uses the classes in generator.py to generate the synthetic data

from .generator import PSMGenerator, PSWGenerator, IVGenerator, RDDGenerator, RCTGenerator, DiDGenerator, MultiTreatRCTGenerator, FrontDoorGenerator
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import logging.config
import json

from .util import export_info

Path("reproduce_results/logs").mkdir(parents=True, exist_ok=True)
logging.config.fileConfig('reproduce_results/log_config.ini')

def config_hyperparameters(base_seed, base_mean, base_cov_diag, max_cont, max_bin, n_obs,
                           max_obs, min_obs, max_treat=2, max_periods=5, cutoff_max=25):
    """
    configure the hyperparameters for the data generation process.

    Args:
        base_seed (int): Base seed for random number generation
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov_diag (np.ndarray): Base (diagonal) covariance matrix for the covariates
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        n_obs (int): Number of observations to generate
        max_obs (int): Maximum number of observations to generate
        min_obs (int): Minimum number of observations to generate
        max_treat (int): Maximum number of treatment groups (default is 2)
        max_periods (int): Maximum number of periods for DiD data (default is 5)
        cutoff_max (int): Maximum value for the cutoff in RDD data (default is 25)

    Returns:
        dict: A dictionary containing the hyperparameters for data generation.
             (str) attribute -> (int) value


    """

    base_cov_mat = np.diag(base_cov_diag)
    np.random.seed(base_seed)
    n_treat = np.random.randint(2, max_treat + 1)
    true_effect = np.random.uniform(1, 10)
    true_effect_vec = np.array([0] + [np.random.uniform(1, 10) for i in range(n_treat)])
    n_continuous = np.random.randint(2, max_cont + 1)
    n_binary = np.random.randint(2, max_bin)
    n_observations = np.random.randint(min_obs, max_obs + 1)
    if n_obs is not None:
        n_observations = n_obs
    n_periods = np.random.randint(3, max_periods + 1)
    cutoff = np.random.randint(2, cutoff_max + 1)
    mean_vec = base_mean[0:n_continuous]
    cov_mat = base_cov_mat[0:n_continuous, 0:n_continuous]


    param_dict = {'tau': true_effect, 'continuous': n_continuous, 'binary': n_binary,
                  'obs': n_observations, 'mean': mean_vec, 'covar': cov_mat,
                  'tau_vec':true_effect_vec, "treat":n_treat, "periods": n_periods,
                  'cutoff':cutoff}

    return param_dict


def generate_observational_data(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs,
                                max_obs, data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate observational data using the PSMGenerator class.

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("observational_data_logger")
    logger.info("Generating observational data")
    metadata_dict = {}
    base_seed = 31
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
                                        n_obs, max_obs, min_obs)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = PSMGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed*2)
        data = gen.generate_data()
        name = "observational_data_{}.csv".format(i)
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "observational"}
        test_result = gen.test_data()
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)
    export_info(metadata_dict, metadata_save_loc, "observational")


def generate_rct_data(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs, max_obs,
                      data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generates RCT data

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("rct_data_logger")
    logger.info("Generating RCT data")
    metadata_dict = {}
    base_seed = 197
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = RCTGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed)
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "rct"}
        name = "rct_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)
    export_info(metadata_dict, metadata_save_loc, "rct")


def generate_multi_rct_data(base_mean, base_cov, dset_size, max_n_treat, max_cont, max_bin, min_obs, max_obs,
                            data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate multi-treatment RCT data
    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_n_treat (int): Maximum number of treatment groups
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """
    logger = logging.getLogger("multi_rct_data_logger")
    logger.info("Generating multi-treatment RCT data")
    metadata_dict = {}
    base_seed = 173
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i+1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs, max_treat=max_n_treat)
        n_treat = params['treat']
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}, n_treat: {}".format(
            params['obs'], params['continuous'], params['binary'], n_treat))
        logger.info("true_effect: {}".format(params['tau_vec']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = MultiTreatRCTGenerator(params['obs'], params['continuous'], params['treat'], n_binary_covars=params['binary'],
                                     mean=mean_vec, covar=cov_mat, true_effect_vec=params['tau_vec'], seed=seed,
                                     true_effect=params['tau'])
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": list(params['tau_vec']), "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "multi_rct"}
        name = "multi_rct_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)
    export_info(metadata_dict, metadata_save_loc, "multi_rct")


def generate_frontdoor_data(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs, max_obs,
                             data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generates front-door data

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Max number of continuous covariates
        max_bin (int): Max number of binary covariates
        min_obs (int): Minimum number of observations
        max_obs (int): Maximum number of observations
        data_save_loc (str): Folder to save generated CSV files
        metadata_save_loc (str): Folder to save metadata JSON
        n_obs (int or None): Fixed number of observations (if provided)
    """

    logger = logging.getLogger("frontdoor_data_logger")
    logger.info("Generating Front-Door synthetic data")
    metadata_dict = {}
    base_seed = 311 

    for i in range(dset_size):
        logger.info(f"Iteration: {i}")
        seed = (i + 1) * base_seed

        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs)

        logger.info("n_observations: {}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))

        mean_vec = params['mean']
        cov_mat = params['covar']

        gen = FrontDoorGenerator(
            n_observations=params['obs'],
            n_continuous_covars=params['continuous'],
            n_binary_covars=params['binary'],
            mean=mean_vec,
            covar=cov_mat,
            true_effect=params['tau'],
            seed=seed
        )

        data = gen.generate_data()
        test_result = gen.test_data()
        logger.info("Test result: {}\n".format(test_result))

        # Save CSV
        filename = f"frontdoor_data_{i}.csv"
        gen.save_data(data_save_loc, filename)

        # Metadata
        data_dict = {
            "true_effect": params['tau'],
            "observation": params['obs'],
            "continuous": params['continuous'],
            "binary": params['binary'],
            "type": "frontdoor"
        }
        metadata_dict[filename] = data_dict

    # Save metadata JSON
    export_info(metadata_dict, metadata_save_loc, "frontdoor")



def generate_canonical_did_data(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs, max_obs,
                                data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate canonical DiD data
    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """
    logger = logging.getLogger("did_data_logger")
    logger.info("Generating canonical DiD data")
    metadata_dict = {}
    base_seed = 281
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = DiDGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed)
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "did_canonical"}
        name = "did_canonical_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)

    export_info(metadata_dict, metadata_save_loc, "did_canonical")

def generate_data_iv(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs, max_obs,
                    data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate IV data
    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("iv_data_logger")
    logger.info("Generating IV data")
    metadata_dict = {}
    base_seed = 343
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = IVGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                          mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed)
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "IV"}
        name = "iv_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)

    export_info(metadata_dict, metadata_save_loc, "iv")

def generate_twfe_did_data(base_mean, base_cov, dset_size, max_cont, max_bin, n_periods,
                           min_obs, max_obs, data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate TWFE DiD data

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        n_periods (int): Number of periods for the DiD data
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("did_data_logger")
    logger.info("Generating TWFE DiD data")
    metadata_dict = {}
    base_seed = 447
    print("preiods: ", n_periods)
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs, max_periods=n_periods)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}, n_periods:{}".format(
            params['obs'], params['continuous'], params['binary'], params['periods']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = DiDGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed,
                           n_periods=n_periods)
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "did_twfe", "periods": params['periods']}
        name = "did_twfe_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)

    export_info(metadata_dict, metadata_save_loc, "did_twfe")

def generate_encouragement_data(base_mean, base_cov, dset_size, max_cont, max_bin, min_obs, max_obs,
                                data_save_loc, metadata_save_loc, n_obs=None):
    """
    Generate encouragement design data

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("iv_data_logger")
    logger.info("Generating encouragement design data")
    metadata_dict = {}
    base_seed = 571
    for i in range(dset_size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}".format(
            params['obs'], params['continuous'], params['binary']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = IVGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed,
                           encouragement=True)
        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "encouragement"}
        name = "iv_encouragement_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)

    export_info(metadata_dict, metadata_save_loc, "iv_encouragement")


def generate_rdd_data(base_mean, base_cov, dset_size, max_cont, max_bin, max_cutoff,
                      min_obs, max_obs, data_save_loc, metadata_save_loc, n_obs=None):

    """
    Generates (sharp) RDD data

    Args:
        base_mean (np.ndarray): Base mean vector for the covariates
        base_cov (np.ndarray): Base covariance matrix for the covariates
        dset_size (int): Number of datasets to generate
        max_cont (int): Maximum number of continuous covariates
        max_bin (int): Maximum number of binary covariates
        max_cutoff (int): Maximum value for the cutoff in RDD data
        min_obs (int): Minimum number of observations to generate
        max_obs (int): Maximum number of observations to generate
        data_save_loc (str): Directory to save the generated data files
        metadata_save_loc (str): Directory to save the metadata information
        n_obs (int, None): number of observations. If None, it will be randomly
                           generated within the range of min_obs and max_obs.
    """

    logger = logging.getLogger("rdd_data_logger")
    logger.info("Generating RDD data")
    metadata_dict = {}
    base_seed = 683
    for i in range(dset_size):
        logger.info("Iteration:{}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, n_obs,
                                        max_obs, min_obs, cutoff_max=max_cutoff)
        logger.info("n_observations:{}, n_continuous: {}, n_binary: {}, cutoff:{}".format(
            params['obs'], params['continuous'], params['binary'], params['cutoff']))
        logger.info("true_effect: {}".format(params['tau']))
        mean_vec = params['mean']
        cov_mat = params['covar']
        gen = RDDGenerator(params['obs'], params['continuous'], n_binary_covars=params['binary'],
                           mean=mean_vec, covar=cov_mat, true_effect=params['tau'], seed=seed,
                           cutoff=params['cutoff'], plot=True)

        data = gen.generate_data()
        test_result = gen.test_data()
        data_dict = {"true_effect": params['tau'], "observation": params['obs'], "continuous": params['continuous'],
                     "binary": params['binary'], "type": "rdd", 'cutoff': params['cutoff']}
        name = "rdd_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)

    export_info(metadata_dict, metadata_save_loc, "rdd")

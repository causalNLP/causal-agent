from .generator import PSMGenerator, PSWGenerator, IVGenerator, RDDGenerator, RCTGenerator, DiDGenerator, MultiTreatRCTGenerator
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import logging.config
import json

from .util import export_info

Path("logs").mkdir(parents=True, exist_ok=True)
logging.config.fileConfig('log_config.ini')

def config_hyperparameters(base_seed, base_mean, base_cov_diag, max_cont, max_bin, 
                           max_obs, min_obs, max_treat=2, max_periods=5, cutoff_max=25):

    base_cov_mat = np.diag(base_cov_diag)
    np.random.seed(base_seed)
    n_treat = np.random.randint(2, max_treat + 1)
    true_effect = np.random.uniform(1, 10)
    true_effect_vec = np.array([0] + [np.random.uniform(1, 10) for i in range(n_treat)])
    n_continuous = np.random.randint(2, max_cont + 1)
    n_binary = np.random.randint(2, max_bin)
    n_observations = np.random.randint(min_obs, max_obs + 1)
    n_periods = np.random.randint(3, max_periods + 1)
    cutoff = np.random.randint(2, cutoff_max + 1)
    mean_vec = base_mean[0:n_continuous]
    cov_mat = base_cov_mat[0:n_continuous, 0:n_continuous]


    param_dict = {'tau': true_effect, 'continuous': n_continuous, 'binary': n_binary,
                  'obs': n_observations, 'mean': mean_vec, 'covar': cov_mat, 
                  'tau_vec':true_effect_vec, "treat":n_treat, "periods": n_periods, 
                  'cutoff':cutoff}
    
    return param_dict


def generate_observational_data(base_mean, base_cov, size, max_cont, max_bin, min_obs, 
                                max_obs, data_save_loc, metadata_save_loc):

    logger = logging.getLogger("observational_data_logger")
    logger.info("Generating observational data")
    metadata_dict = {}
    base_seed = 31
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin, 
                                        max_obs, min_obs)
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


def generate_rct_data(base_mean, base_cov, size, max_cont, max_bin, min_obs, max_obs, 
                      data_save_loc, metadata_save_loc):

    logger = logging.getLogger("rct_data_logger")
    logger.info("Generating RCT data")
    metadata_dict = {}
    base_seed = 197
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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


def generate_multi_rct_data(base_mean, base_cov, size, max_n_treat, max_cont, max_bin, min_obs, max_obs, 
                            data_save_loc, metadata_save_loc):
    """
    Generate multi-treatment RCT data
    """
    logger = logging.getLogger("multi_rct_data_logger")
    logger.info("Generating multi-treatment RCT data")
    metadata_dict = {}
    base_seed = 173
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i+1) * base_seed 
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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


def generate_canonical_did_data(base_mean, base_cov, size, max_cont, max_bin, min_obs, max_obs, 
                                data_save_loc, metadata_save_loc):
    """
    Generate canonical DiD data
    """
    logger = logging.getLogger("did_data_logger")
    logger.info("Generating canonical DiD data")
    metadata_dict = {}
    base_seed = 281
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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
        name = "did_data_{}.csv".format(i)
        logger.info("Test result: {}\n".format(test_result))
        metadata_dict[name] = data_dict
        gen.save_data(data_save_loc, name)
    export_info(metadata_dict, metadata_save_loc, "did")

def generate_data_iv(base_mean, base_cov, size, max_cont, max_bin, min_obs, max_obs,
                    data_save_loc, metadata_save_loc):
    """
    Generate IV data
    """
    logger = logging.getLogger("iv_data_logger")
    logger.info("Generating IV data")
    metadata_dict = {}
    base_seed = 343
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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

def generate_twfe_did_data(base_mean, base_cov, size, max_cont, max_bin, n_periods, 
                           min_obs, max_obs, data_save_loc, metadata_save_loc):
    """
    Generate TWFE DiD data
    """
    logger = logging.getLogger("did_data_logger")
    logger.info("Generating TWFE DiD data")
    metadata_dict = {}
    base_seed = 447
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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

def generate_encouragement_data(base_mean, base_cov, size, max_cont, max_bin, min_obs, max_obs,
                                data_save_loc, metadata_save_loc):
    """
    Generate encouragement design data
    """
    logger = logging.getLogger("iv_data_logger")
    logger.info("Generating encouragement design data")
    metadata_dict = {}
    base_seed = 571
    for i in range(size):
        logger.info("Iteration: {}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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


def generate_rdd_data(base_mean, base_cov, size, max_cont, max_bin, max_cutoff, 
                      min_obs, max_obs, data_save_loc, metadata_save_loc):
    logger = logging.getLogger("rdd_data_logger")
    logger.info("Generating RDD data")
    metadata_dict = {}
    base_seed = 683
    for i in range(size):
        logger.info("Iteration:{}".format(i))
        seed = (i + 1) * base_seed
        params = config_hyperparameters(seed, base_mean, base_cov, max_cont, max_bin,
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
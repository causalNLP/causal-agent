## This scripts generates synthetic data associated with various causal inference methods using
## functions in auto_causal.synthetic.generator. The output is a csv file containing the data
## and a json file containing the metadata.

## ToDO: change causalscientist after renaming the package 
from causalscientist.auto_causal.synthetic.io import generate_observational_data, generate_rct_data, \
    generate_multi_rct_data, generate_canonical_did_data, generate_twfe_did_data, generate_data_iv, \
    generate_encouragement_data, generate_rdd_data

from pathlib import Path
import numpy as np
import argparse

OBS_MEAN_POOL = np.array([
    [28, 22,  8, 15,  3],
    [30, 20, 10, 14,  4],
    [25, 24,  6, 16,  2],
    [32, 18, 12, 13,  5],
])
OBS_COV_POOL = np.array([
    [81, 25,  7, 16, 2],
    [64, 36,  9, 25, 3],
    [100,16,  5, 20, 1],
    [49, 30,  8, 18, 4],
])

RCT2_MEAN_POOL = np.array([
    [3, 7],
    [5, 1],
    [2, 8],
    [4, 6],
])
RCT2_COV_POOL = np.array([
    [1, 4],
    [2, 3],
    [3, 5],
    [0.5, 2],
])

DID_CAN_MEAN_POOL = np.array([
    [3, 7],
    [5, 2],
    [2, 8],
    [4, 6],
])
DID_CAN_COV_POOL = np.array([
    [1, 3],
    [2, 2],
    [0.5,4],
    [3, 1],
])

DID_TWFE_MEAN_POOL = np.array([
    [3, 10, 20],
    [5,  8, 18],
    [2, 12, 22],
    [4,  9, 25],
])
DID_TWFE_COV_POOL = np.array([
    [1,  7, 12],
    [2, 10, 10],
    [0.5,5, 15],
    [3,  6, 20],
])

IV_MEAN_POOL = np.array([
    [13, 28, 10, 12],
    [15, 25, 12, 10],
    [10, 30, 8,  14],
    [12, 26, 9,  11],
])
IV_COV_POOL = np.array([
    [7, 9, 8, 8],
    [5, 7, 6, 9],
    [8, 5,10, 7],
    [6,12, 7, 5],
])

ENC_MEAN_POOL = np.array([
    [5, 2, 0.5],
    [6, 1, 1.0],
    [4, 3, 0.2],
    [7, 2, 0.7],
])
ENC_COV_POOL = np.array([
    [4, 0.4, 0.3],
    [3, 0.5, 0.2],
    [5, 0.3, 0.4],
    [6, 0.6, 0.1],
])

RDD_MEAN_POOL = np.array([
    [10, 5],
    [12, 4],
    [8,  6],
    [14, 3],
])
RDD_COV_POOL = np.array([
    [8, 4],
    [5, 6],
    [10,2],
    [6, 8],
])



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, required=True,
                        choices=["observational", "rct", "multi_rct", "did_canonical",
                                 "did_twfe", "iv", "iv_encouragement", "rdd"],
                        help="Method to generate data for")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to the folder where the data will be saved")
    parser.add_argument("-md", "--metadata_path", type=str, required=True,
                        help="Path to the folder where the metadata will be saved")
    parser.add_argument("-o", "--observations", type=int, default=None,
                        help="Number of observations to genegerate. If not specificied, " \
                        "a random number will be chose from pre-specified range")
    parser.add_argument("-s", "--size", type=int, default=1,
                        help="Number of datasets to generate")
    parser.add_argument("-mb", "--max_binary", type=int, default=4,
                        help="Maximum number of binary variables in the dataset")
    parser.add_argument("-mc", "--max_continuous", type=int, default=4,
                        help="Maximum number of continuous variables in the dataset")
    parser.add_argument("-mo", "--max_observations", type=int, default=500,
                        help="Maximum number of observations in the dataset")
    parser.add_argument("-mno", "--min_observations", type=int, default=501,
                        help="Minimum number of observations in the dataset")
    parser.add_argument("-nt", "--n_treatments", type=int, default=2,
                        help="Number of treatments in the dataset (for multi-RCT data)")
    parser.add_argument("-np", "--n_periods", type=int, default=3,
                        help="Number of treatment periods in the dataset (for DiD data)")
    parser.add_argument("-c", "--cutoff", type=int, default=25,
                        help="Cutoff value for RDD data. ")

    return parser.parse_args()

if __name__ == "__main__":

    ### this is used for observational and RCT data generation; inspired by the famous "Lalonde" dataset
    args = parse_args()
    method = args.method
    data_path = args.data_path
    metadata_path = args.metadata_path
    print(args)

    if method == "observational":
        idx = np.random.choice(len(OBS_MEAN_POOL))
        base_mean1 = OBS_MEAN_POOL[idx]
        base_cov_diag1 = OBS_COV_POOL[idx]
        generate_observational_data(base_mean1, base_cov_diag1, args.size, args.max_continuous, args.max_binary,
                                    args.min_observations, args.max_observations,data_path, metadata_path,
                                    n_obs=args.observations)

    elif method == "rct":
        idx = np.random.choice(len(OBS_MEAN_POOL))
        base_mean1     = OBS_MEAN_POOL[idx]
        base_cov_diag1 = OBS_COV_POOL[idx]
        generate_rct_data(base_mean1, base_cov_diag1, args.size, args.max_continuous, args.max_binary,
                                    args.min_observations, args.max_observations,data_path, metadata_path,
                                    n_obs=args.observations)

    elif method == "multi_rct":
        idx = np.random.choice(len(RCT2_MEAN_POOL))
        base_mean2     = RCT2_MEAN_POOL[idx]
        base_cov_diag2 = RCT2_COV_POOL[idx]
        generate_multi_rct_data(base_mean2, base_cov_diag2, args.size, args.n_treatments, args.max_continuous,
                                args.max_binary, args.min_observations, args.max_observations,
                                data_path, metadata_path, n_obs=args.observations)

    elif method == "did_canonical":
        idx = np.random.choice(len(DID_CAN_MEAN_POOL))
        base_mean3 = DID_CAN_MEAN_POOL[idx]
        base_cov_diag3 = DID_CAN_COV_POOL[idx]
        generate_canonical_did_data(base_mean3, base_cov_diag3, args.size, args.max_continuous, args.max_binary,
                                    args.min_observations, args.max_observations ,data_path, metadata_path,
                                    n_obs=args.observations)
    elif method == "did_twfe":
        idx = np.random.choice(len(DID_TWFE_MEAN_POOL))
        base_mean4     = DID_TWFE_MEAN_POOL[idx]
        base_cov_diag4 = DID_TWFE_COV_POOL[idx]
        generate_twfe_did_data(base_mean4, base_cov_diag4, args.size, args.max_continuous, args.max_binary,
                               args.n_periods, args.min_observations, args.max_observations,
                              data_path, metadata_path, n_obs=args.observations)
    elif method == "iv":
        idx = np.random.choice(len(IV_MEAN_POOL))
        base_mean5     = IV_MEAN_POOL[idx]
        base_cov_diag5 = IV_COV_POOL[idx]
        generate_data_iv(base_mean5, base_cov_diag5, args.size, args.max_continuous, args.max_binary,
                        args.min_observations, args.max_observations,
                        data_path, metadata_path, n_obs=args.observations)
    elif method == "iv_encouragement":

        idx = np.random.choice(len(ENC_MEAN_POOL))
        base_mean6     = ENC_MEAN_POOL[idx]
        base_cov_diag6 = ENC_COV_POOL[idx]
        generate_encouragement_data(base_mean6, base_cov_diag6, args.size, args.max_continuous, args.max_binary,
                        args.min_observations, args.max_observations,
                        data_path, metadata_path, n_obs=args.observations)
    elif method == "rdd":

        idx = np.random.choice(len(RDD_MEAN_POOL))
        base_mean7     = RDD_MEAN_POOL[idx]
        base_cov_diag7 = RDD_COV_POOL[idx]

        generate_rdd_data(base_mean7, base_cov_diag7, args.size, args.max_continuous, args.max_binary,
                          args.cutoff, args.min_observations, args.max_observations,
                          data_path, metadata_path, n_obs=args.observations)
    else:
        raise ValueError("Invalid method. Choose from: observational, rct, multi_rct, " \
        "did_canonical, did_twfe, iv, iv_encouragement, rdd")

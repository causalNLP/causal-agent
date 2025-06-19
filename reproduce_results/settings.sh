## Here we set all the hyperparams for the synthetic data and define base directories

export BASE_FOLDER="reproduce_results/samples/synthetic"

## dataset sizes
export RCT_SIZE=10
export MULTI_RCT_SIZE=5
export CANONICAL_DID_SIZE=5
export TWFE_DID_SIZE=5
export OBSERVATIONAL_SIZE=5
export IV_SIZE=5
export ENCOURAGEMENT_SIZE=5
export RDD_SIZE=5
export DEFAULT_SIZE=2

## number of observations
export MIN_OBS=300
export MAX_OBS=500
export DEFAULT_OBS=1000
export DEFAULT_OBS_TWFE=100
export MIN_OBS_TWFE=50
export MAX_OBS_TWFE=100

## maximum number of treatments for multi RCT
export MAX_TREATMENTS=5

## maximum number of periods for TWFE
export MAX_PERIODS=10

## maximum number of covariates
export N_CONTINUOUS=5
export N_CONTINUOUS_MULTI=2
export N_CONTINUOUS_DID_CANONICAL=2
export N_CONTINUOUS_DID_TWFE=2
export N_CONTINUOUS_IV=4
export N_CONTINUOUS_IV_ENCOURAGEMENT=3
export N_CONTINUOUS_RDD=2

export N_BINARY=4
export N_BINARY_OTHERS=3

## cutoff for RDD
export CUTOFF=25

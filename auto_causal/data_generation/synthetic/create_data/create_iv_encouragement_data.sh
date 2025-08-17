#!/bin/sh

#  create_descriptions.sh
#  
#
#  Created by Sawal Acharya on 5/14/25.
#

source auto_causal/data_generation/synthetic/settings.sh
METHOD="iv_encouragement"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python auto_causal/data_generation/synthetic/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY_OTHERS} -mc ${N_CONTINUOUS_IV_ENCOURAGEMENT} -o ${DEFAULT_OBS}

#!/bin/sh

#  create_descriptions.sh
#  
#
#  Created by Sawal Acharya on 5/14/25.
#

source reproduce_results/settings.sh
METHOD="did_twfe"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python main/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY_OTHERS} -mc ${N_CONTINUOUS_DID_TWFE}  -np ${MAX_PERIODS} -o ${DEFAULT_OBS_TWFE}

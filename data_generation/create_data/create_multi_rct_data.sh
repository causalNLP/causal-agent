#!/bin/sh

#  create_descriptions.sh
#  
#
#  Created by Sawal Acharya on 5/14/25.
#

source data_generation/settings.sh
METHOD="multi_rct"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python data_generation/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY} -mc ${N_CONTINUOUS_MULTI}  -nt ${MAX_TREATMENTS} -o ${DEFAULT_OBS}

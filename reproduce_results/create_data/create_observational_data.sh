#!/bin/sh

#  create_descriptions.sh
#  
#
#  Created by Sawal Acharya on 5/14/25.
#

source reproduce_results/settings.sh
METHOD="observational"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python main/generate_synthetic.py -md ${METADATA_FOLDER} -d ${DATA_FOLDER} -m ${METHOD} -s ${DEFAULT_SIZE} -mb ${N_BINARY} -mc ${N_CONTINUOUS} -o ${DEFAULT_OBS}

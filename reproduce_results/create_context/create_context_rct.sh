#!/bin/sh

#  create_descriptions.sh
#  This script generates the column labels, backstory, and causal query for RCT synthetic datasets.
#
#  Created by Sawal Acharya on 5/14/25.
#

source reproduce_results/settings.sh
METHOD="rct"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata/${METHOD}.json"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"
OUTPUT_FOLDER="${BASE_FOLDER}/${METHOD}/description"

python main/generate_context.py -mp ${METADATA_FOLDER} -d ${DATA_FOLDER} -o ${OUTPUT_FOLDER} -m ${METHOD}

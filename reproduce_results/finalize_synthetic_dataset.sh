#!/bin/sh

#  finalize_synthetic_dataset.sh
# This scripts puts together the results of generate_context.py and generate_synthetic.py. It renames the columns of the synthetic data files, and saves the resulting csv file. Additionally, it creates a summary csv file including the key information needed to run and evaluate the tests on the synthetic data.
#
#  Created by Sawal Acharya on 5/14/25.
#
source reproduce_results/settings.sh

for METHOD in rct multi_rct did_canonical did_twfe iv iv_encouragement rdd observational; do
    METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata/${METHOD}.json"
    INPUT_DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"
    OUTPUT_PATH="${BASE_FOLDER}/data_info"
    DESCRIPTION_PATH="${BASE_FOLDER}/${METHOD}/description/${METHOD}.json"
    OUTPUT_DATA_FOLDER="${BASE_FOLDER}/synthetic_data"

    python main/finalize_data.py \
        -md "$METADATA_FOLDER" \
        -id "$INPUT_DATA_FOLDER" \
        -m  "$METHOD" \
        -o  "$OUTPUT_PATH" \
        -de "$DESCRIPTION_PATH" \
        -od "$OUTPUT_DATA_FOLDER"
done


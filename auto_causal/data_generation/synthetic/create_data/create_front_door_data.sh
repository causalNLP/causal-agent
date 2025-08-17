source auto_causal/data_generation/synthetic/settings.sh
METHOD="frontdoor"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"

python auto_causal/data_generation/synthetic/generate_synthetic.py \
    -md ${METADATA_FOLDER} \
    -d ${DATA_FOLDER} \
    -m ${METHOD} \
    -s ${DEFAULT_SIZE} \
    -mb ${N_BINARY} \
    -mc ${N_CONTINUOUS_FRONTDOOR} \
    -o ${DEFAULT_OBS}

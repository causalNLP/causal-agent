source data_generation/settings.sh
METHOD="frontdoor"
METADATA_FOLDER="${BASE_FOLDER}/${METHOD}/metadata/${METHOD}.json"
DATA_FOLDER="${BASE_FOLDER}/${METHOD}/data"
OUTPUT_FOLDER="${BASE_FOLDER}/${METHOD}/description"

python data_generation/generate_context.py \
    -mp ${METADATA_FOLDER} \
    -d ${DATA_FOLDER} \
    -o ${OUTPUT_FOLDER} \
    -m ${METHOD}

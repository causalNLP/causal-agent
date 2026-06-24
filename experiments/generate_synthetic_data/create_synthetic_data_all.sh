#!/bin/sh

#  create_synthetic_data_all.sh
# This scripts generates all the synthetic data
#
#
#  Created by Sawal Acharya on 5/14/25.
#


echo "Generating RCT Data"
bash data_generation/create_data/create_rct_data.sh

echo "Generating Multi-RCT Data"
bash data_generation/create_data/create_multi_rct_data.sh

echo "Generating Front_Door Data"
bash data_generation/create_data/create_front_door_data.sh

echo "Generating Observational Data"
bash data_generation/create_data/create_observational_data.sh

echo "Generating Canonical DiD Data"
bash data_generation/create_data/create_did_canonical_data.sh

echo "Generating TWFE DiD Data"
bash data_generation/create_data/create_did_twfe_data.sh

echo "Generating IV Data"
bash data_generation/create_data/create_iv_data.sh

echo "Generating IV-Encouragement Data"
bash data_generation/create_data/create_iv_encouragement_data.sh

echo "Generating RDD Data"
bash data_generation/create_data/create_rdd_data.sh

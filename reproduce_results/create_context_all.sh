#!/bin/sh

#  create_descriptions.sh
#  This script generates the column labels, backstory, and causal query for all the synthetic datasets.
#
#  Created by Sawal Acharya on 5/14/25.
#

echo "Generating context for RCT Data"
bash reproduce_results/create_context/create_context_rct.sh

echo "Generating context for Multi-RCT Data"
bash reproduce_results/create_context/create_context_multi_rct.sh

echo "Generating context for Observational Data"
bash reproduce_results/create_context/create_context_observational.sh

echo "Generating context for Canonical DiD Data"
bash reproduce_results/create_context/create_context_did_canonical.sh

echo "Generating context for TWFE DiD Data"
bash reproduce_results/create_context/create_context_did_twfe.sh

echo "Generating context for IV Data"
bash reproduce_results/create_context/create_context_iv.sh

echo "Generating context for IV-Encouragement Data"
bash reproduce_results/create_context/create_context_iv_encouragement.sh

echo "Generating context for RDD Data"
bash reproduce_results/create_context/create_context_rdd.sh


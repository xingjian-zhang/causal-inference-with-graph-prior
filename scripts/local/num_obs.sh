#!/bin/bash
# Usage: ./num_obs.sh <seed>
# seed: Specify the seed to run

RANDOM_SEED=$1
# Raise error if seed is not specified
if [ -z "$RANDOM_SEED" ]; then
  echo "Seed is not specified"
  exit 1
fi

NUM_OBS=(100 500 2000)

for i in "${NUM_OBS[@]}"; do
  echo "Running experiment with RANDOM_SEED=$RANDOM_SEED and NUM_OBS=$i"
  experiment_name="n${i}"
  # make all RANDOM_SEED=$RANDOM_SEED DATASET_CONFIG=syntheticV2_no_ctrl.yaml OVERRIDE_CFG="data.dataset.num_observation=${i}" EXPERIMENT_NAME=${experiment_name}
  make all RANDOM_SEED=$RANDOM_SEED DATASET_CONFIG=syntheticV2_ctrl.yaml OVERRIDE_CFG="data.dataset.num_observation=${i}" EXPERIMENT_NAME=${experiment_name}
done

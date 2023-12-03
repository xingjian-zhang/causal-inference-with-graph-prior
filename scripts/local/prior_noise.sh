#!/bin/bash
# Usage: ./num_obs.sh <seed>
# seed: Specify the seed to run

RANDOM_SEED=$1
# Raise error if seed is not specified
if [ -z "$RANDOM_SEED" ]; then
  echo "Seed is not specified"
  exit 1
fi

B_NOISE=(0 0.1 0.2 0.3)

for i in "${B_NOISE[@]}"; do
  echo "Running experiment with RANDOM_SEED=$RANDOM_SEED and B_NOISE=$i"
  experiment_name="b${i}"
  make gr_lr RANDOM_SEED=$RANDOM_SEED DATASET_CONFIG=syntheticV2_ctrl.yaml OVERRIDE_CFG="data.dataset.strategy_kwargs.b_noise_sigma=${i}" EXPERIMENT_NAME=${experiment_name}
done

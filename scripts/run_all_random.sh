#!/bin/bash
# Usage: ./run_all_random.sh [-m model]
# -m model: Specify the model to run. If not provided, defaults to "all".

# Default model
model="all"

# Parse optional arguments
while getopts ":m:" opt; do
  case ${opt} in
    m)
      model=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done

for i in {1..10}; do
    echo "Running experiment with RANDOM_SEED=$i"
    make $model RANDOM_SEED=$i
done

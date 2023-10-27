#!/bin/bash

# Run the 'make all' command 10 times with 10 different random seeds
for i in {1..10}; do
    echo "Running experiment with RANDOM_SEED=$i"
    make all RANDOM_SEED=$i
done

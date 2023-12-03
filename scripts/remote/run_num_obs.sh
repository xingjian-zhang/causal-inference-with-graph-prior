#!/bin/bash
#SBATCH --job-name=causal_num_obs
#SBATCH --mail-user=jimmyzxj@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=60:00
#SBATCH --account=qmei3
#SBATCH --partition=standard
#SBATCH --array=1-5
#SBATCH --output=/home/%u/Research/causal_graph_prior/logs/%x-%A-%j.log

RANDOM_SEED=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=/home/jimmyzxj/Research/causal_graph_prior

cd $PROJECT_DIR
PYTHON=/home/jimmyzxj/miniconda3/envs/causal_graph_prior/bin/python3 bash ./scripts/local/num_obs.sh ${RANDOM_SEED}

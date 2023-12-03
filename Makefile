# Always use the following configs.
DATA_CONFIG_DIR := configs/data
MODELS_CONFIG_DIR := configs/models
# Optionally change the following configs.
PYTHON ?= /Users/jimmy/miniforge3/envs/gli/bin/python
RANDOM_SEED ?= 42
DATASET_CONFIG ?= tmdb5000_no_ctrl.yaml
REMOTE_SYNC_PATH ?= jimmyzxj@greatlakes-xfer.arc-ts.umich.edu:/home/jimmyzxj/Research/causal_graph_prior
REMOTE_PROJECT_PATH ?= /home/jimmyzxj/Research/causal_graph_prior
REMOTE_SERVER ?= greatlakes
EXPERIMENT_NAME ?= ""
OVERRIDE_CFG ?= ""

.PHONY: lr gr_lr sgc collect-results clean clean-all all results tensorboard install sync-remote sync-local run-remote-synthetic clean-all-remote

clean:
	rm -rf lightning_logs
	for dir in tb_logs/*/*; do \
		if [[ -d "$$dir" ]]; then \
			ls -1 $$dir | sort -r | tail -n +2 | xargs -I {} echo rm -rf "$$dir/{}"; \
		fi \
	done

clean-all:
	rm -rf lightning_logs
	rm -rf tb_logs

clean-all-remote:
	ssh $(REMOTE_SERVER) "cd $(REMOTE_PROJECT_PATH) && make clean-all"

lr gr_lr sgc:
	@echo "\n>>>Running $@<<<\n"
	$(PYTHON) train.py $(MODELS_CONFIG_DIR)/$@.yaml $(DATA_CONFIG_DIR)/$(DATASET_CONFIG) --random_seed $(RANDOM_SEED) --override_cfg $(OVERRIDE_CFG) --experiment_name $(EXPERIMENT_NAME)

all: lr gr_lr sgc

results:
	mkdir -p results

collect-results: results
	$(PYTHON) -c 'from utils import collect_results; collect_results()'

install:
	pip install -r requirements.txt

tensorboard:
	tensorboard --logdir tb_logs

sync-remote:
	# Generate the exclude list from .gitignore
	git ls-files --others --ignored --exclude-standard -z | xargs -0 -I {} echo '{}' > .rsync-exclude
	echo '.rsync-exclude' >> .rsync-exclude
	echo '.git' >> .rsync-exclude
	# Run rsync with the exclude list
	rsync -av --exclude-from='.rsync-exclude' . $(REMOTE_SYNC_PATH)
	# Optional: Remove the exclude file
	rm .rsync-exclude

sync-local:
	# Sync results back to local
	rsync -av $(REMOTE_SYNC_PATH)/tb_logs/ tb_logs/

run-remote-synthetic: sync-remote
	ssh $(REMOTE_SERVER) "sbatch $(REMOTE_PROJECT_PATH)/scripts/remote/run_synthetic.sh"

run-remote-tmdb5000: sync-remote
	ssh $(REMOTE_SERVER) "sbatch $(REMOTE_PROJECT_PATH)/scripts/remote/run_tmdb5000.sh"

run-remote: sync-remote
	ssh $(REMOTE_SERVER) "sbatch $(REMOTE_PROJECT_PATH)/scripts/remote/run_synthetic.sh; \
		sbatch $(REMOTE_PROJECT_PATH)/scripts/remote/run_tmdb5000.sh"

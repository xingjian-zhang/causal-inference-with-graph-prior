PYTHON := /Users/jimmy/miniforge3/envs/gli/bin/python
DATA_CONFIG_DIR := configs/data
MODELS_CONFIG_DIR := configs/models
RANDOM_SEED ?= 42
DATASET_CONFIG ?= tmdb5000_no_ctrl.yaml

.PHONY: lr gr_lr sgc collect-results clean clean-all all results tensorboard install

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

lr gr_lr sgc:
	@echo "\n>>>Running $@<<<\n"
	$(PYTHON) train.py $(MODELS_CONFIG_DIR)/$@.yaml $(DATA_CONFIG_DIR)/$(DATASET_CONFIG) --random_seed $(RANDOM_SEED)

all: lr gr_lr sgc

results:
	mkdir -p results

collect-results: results
	$(PYTHON) -c 'from utils import collect_results; collect_results()'

install:
	pip install -r requirements.txt

tensorboard:
	tensorboard --logdir tb_logs

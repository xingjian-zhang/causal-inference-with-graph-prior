PYTHON := /Users/jimmy/miniforge3/envs/gli/bin/python
CONFIG_DIR := configs

.PHONY: lr mlp lr_cov mlp_cov gr_lr gr_lr_cov collect-results

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

lr mlp lr_cov mlp_cov gr_lr gr_lr_cov:
	$(PYTHON) train.py $(CONFIG_DIR)/$@.yaml --random_seed $(RANDOM_SEED)

all: lr mlp lr_cov mlp_cov gr_lr gr_lr_cov

results:
	mkdir -p results

collect-results: results
	$(PYTHON) -c 'from utils import collect_results; collect_results()'

install:
	pip install -r requirements.txt

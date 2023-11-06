import argparse
import logging
import os
import sys
from functools import partial

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from dataset import get_dataloader, get_obs_dataset_with_gprior, split_dataset
from model import PlugInEstimator, model_factory
from utils import StreamToLogger


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg", type=str)
    parser.add_argument("data_cfg", type=str)
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    def load_yaml_cfg(config_path):
        with open(config_path, "r") as f:
            config_str = f.read()
            return yaml.safe_load(config_str)

    data_cfg = load_yaml_cfg(args.data_cfg)
    model_cfg = load_yaml_cfg(args.model_cfg)
    train_cfg = data_cfg["train"]
    dataset_cfg = data_cfg["dataset"]
    return model_cfg, train_cfg, dataset_cfg


def setup_logging(model_cfg, dataset_cfg):
    cov_str = "_".join(dataset_cfg["covariates"]
                      ) if dataset_cfg["covariates"] else "no_control"
    logger = TensorBoardLogger(
        save_dir="tb_logs/",
        name=model_cfg["type"] + "/" + cov_str,
    )
    os.makedirs(logger.log_dir)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logger.log_dir + "/train.log"),
        ],
    )
    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)
    return logger


def train_and_evaluate(model_cfg, train_cfg, dataset_cfg, logger):
    obs_data = get_obs_dataset_with_gprior(
        poi=dataset_cfg["poi"],
        covariates=dataset_cfg["covariates"],
        strategy=dataset_cfg["strategy"],
        **dataset_cfg["strategy_kwargs"],
    )
    train_data, val_data, test_data = split_dataset(
        obs_data,
        **train_cfg["split_kwargs"],
    )
    dl_kwargs = dataset_cfg["dataloader_kwargs"]

    loader = partial(get_dataloader, **dl_kwargs)
    train_loader = loader(train_data, shuffle=True)
    train_loader_for_val = loader(train_data, shuffle=False)
    val_loader = loader(val_data, shuffle=False)
    val_poi_loader = loader(val_data, shuffle=False, use_poi=True)
    test_loader = loader(test_data, shuffle=False)
    test_poi_loader = loader(test_data, shuffle=False, use_poi=True)

    input_dim = obs_data.x.shape[1]
    if obs_data.z is not None:
        input_dim += obs_data.z.shape[1]

    model = model_factory(
        input_dim,
        obs_data.graph_prior,
        model_type=model_cfg["type"],
        **model_cfg["model_kwargs"],
    )

    estimator = PlugInEstimator(
        model,
        covariate_samples=obs_data.z,  # REVIEW: should we use train_data.z?
        lr=train_cfg["lr"],
        monte_carlo_size=train_cfg["monte_carlo_size"],
    )

    trainer = pl.Trainer(**train_cfg["trainer_kwargs"],
                         logger=logger,
                         enable_progress_bar=False,
                         callbacks=[
                             EarlyStopping(monitor="metric/val",
                                           mode="min",
                                           patience=5)
                         ])
    trainer.fit(
        estimator,
        train_dataloaders=train_loader,
        val_dataloaders=[train_loader_for_val, val_loader, val_poi_loader],
    )
    test_output = trainer.test(
        estimator,
        dataloaders=[test_loader, test_poi_loader],
        verbose=False,
    )
    return test_output


def save_results(model_cfg, train_cfg, dataset_cfg, test_output, logger):
    metric = {k: v for d in test_output for k, v in d.items()}
    config = {"model": model_cfg, "dataset": dataset_cfg, "train": train_cfg}
    logger.log_hyperparams(config, metric)


def main():
    model_cfg, train_cfg, dataset_cfg = load_config()
    logger = setup_logging(model_cfg, dataset_cfg)
    test_output = train_and_evaluate(model_cfg, train_cfg, dataset_cfg, logger)
    save_results(model_cfg, train_cfg, dataset_cfg, test_output, logger)


if __name__ == "__main__":
    main()

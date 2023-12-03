import argparse
import logging
import os
import sys
from functools import partial
from typing import Dict, Any
import warnings

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from dataset import \
    get_dataloader, get_obs_dataset_with_gprior, \
    get_synthetic_dataset_with_gprior, split_dataset, \
    get_synthetic_dataset_with_gpriorV2
from model import PlugInEstimator, model_factory
from utils import StreamToLogger


def override_config(model_cfg: Dict, data_cfg: Dict, override_cfg: str):
    override_cfgs = override_cfg.split(",")
    for override_cfg in override_cfgs:
        key, value = override_cfg.split("=")
        keys = key.split(".")
        which_cfg = keys[0]
        if which_cfg == "model":
            cfg = model_cfg
        elif which_cfg == "data":
            cfg = data_cfg
        else:
            raise ValueError(f"Unknown config: {which_cfg}")
        for k in keys[1:-1]:
            cfg = cfg[k]
        cfg[keys[-1]] = type(cfg[keys[-1]])(value)
    print(model_cfg, data_cfg)
    return model_cfg, data_cfg


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg", type=str)
    parser.add_argument("data_cfg", type=str)
    parser.add_argument("--override_cfg", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
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
    if args.override_cfg is not None:
        model_cfg, data_cfg = override_config(model_cfg, data_cfg,
                                              args.override_cfg)
    train_cfg = data_cfg["train"]
    dataset_cfg = data_cfg["dataset"]
    return model_cfg, train_cfg, dataset_cfg, args.random_seed, args.experiment_name


def setup_logging(
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    random_seed: int,
    experiment_name: str = None,
):
    cov_str = "ctrl" if dataset_cfg["covariates"] else "no_ctrl"
    if experiment_name is None:
        experiment_name = "base"
    logger = TensorBoardLogger(
        save_dir="tb_logs/",
        name=
        f"{experiment_name}/{dataset_cfg['name']}/{model_cfg['type']}/{cov_str}",
        version=random_seed,
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


def get_data(dataset_cfg: Dict[str, Any]):
    dataset_name = dataset_cfg.pop("name")
    if dataset_name == "tmdb5000":
        return get_obs_dataset_with_gprior(**dataset_cfg)
    elif dataset_name == "synthetic":
        return get_synthetic_dataset_with_gprior(**dataset_cfg)
    elif dataset_name == "syntheticV2":
        return get_synthetic_dataset_with_gpriorV2(**dataset_cfg)


def log_graph_prior(graph_prior: np.ndarray):
    num_nodes = graph_prior.shape[0]
    num_edges = np.count_nonzero(graph_prior)
    logging.info(f"Graph prior: {num_nodes} nodes, {num_edges} edges.")
    logging.info(f"Graph prior sparsity: {num_edges / num_nodes**2}")


def train_and_evaluate(
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    logger: TensorBoardLogger,
):
    dl_kwargs = dataset_cfg.pop("dataloader_kwargs")
    obs_data = get_data(dataset_cfg)
    train_data, val_data, test_data = split_dataset(
        obs_data,
        **train_cfg["split_kwargs"],
    )

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

    log_graph_prior(obs_data.graph_prior)
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
        true_params=obs_data.parameters,
    )

    trainer = pl.Trainer(**train_cfg["trainer_kwargs"],
                         logger=logger,
                         enable_progress_bar=False,
                         callbacks=[
                             EarlyStopping(monitor="loss/val",
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


def save_results(
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    test_output: Any,
    logger: TensorBoardLogger,
):
    metric = {k: v for d in test_output for k, v in d.items()}
    config = {"model": model_cfg, "dataset": dataset_cfg, "train": train_cfg}
    logger.log_hyperparams(config, metric)


def main():
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    model_cfg, train_cfg, dataset_cfg, seed, name = load_config()
    logger = setup_logging(model_cfg, dataset_cfg, seed, name)
    test_output = train_and_evaluate(model_cfg, train_cfg, dataset_cfg, logger)
    save_results(model_cfg, train_cfg, dataset_cfg, test_output, logger)


if __name__ == "__main__":
    main()

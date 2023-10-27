import argparse
import logging
import os
import sys

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from dataset import get_dataloader, get_obs_dataset_with_gprior, split_dataset
from model import PlugInEstimator, model_factory
from utils import StreamToLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    parser.add_argument("--random_seed",
                        type=int,
                        default=0,
                        help="Random seed.")
    args = parser.parse_args()
    config_path = args.config

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    with open(config_path, "r") as f:
        config_str = f.read()
        config = yaml.safe_load(config_str)

    data_cfg = config["dataset"]
    train_cfg = config["train"]

    cov_str = "_".join(
        data_cfg["covariates"]) if data_cfg["covariates"] else "no_control"
    logger = TensorBoardLogger(
        save_dir="tb_logs/",
        name=config["model"]["type"] + "/" + cov_str,
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
    print(f"Config:\n{config_str}")

    obs_data = get_obs_dataset_with_gprior(
        poi=data_cfg["poi"],
        covariates=data_cfg["covariates"],
        strategy=data_cfg["strategy"],
        **data_cfg["strategy_kwargs"],
    )
    train_data, val_data, test_data = split_dataset(
        obs_data,
        **train_cfg["split_kwargs"],
    )

    train_loader = get_dataloader(
        train_data,
        shuffle=True,
        **data_cfg["dataloader_kwargs"],
    )
    train_loader_for_val = get_dataloader(
        train_data,
        shuffle=False,
        **data_cfg["dataloader_kwargs"],
    )
    val_loader = get_dataloader(
        val_data,
        shuffle=False,
        **data_cfg["dataloader_kwargs"],
    )
    val_poi_loader = get_dataloader(
        val_data,
        shuffle=False,
        use_poi=True,
        **data_cfg["dataloader_kwargs"],
    )
    test_loader = get_dataloader(
        test_data,
        shuffle=False,
        **data_cfg["dataloader_kwargs"],
    )
    test_poi_loader = get_dataloader(
        test_data,
        shuffle=False,
        use_poi=True,
        **data_cfg["dataloader_kwargs"],
    )

    input_dim = obs_data.x.shape[1]
    if obs_data.z is not None:
        input_dim += obs_data.z.shape[1]

    model = model_factory(
        input_dim,
        obs_data.graph_prior,
        model_type=config["model"]["type"],
        **config["model"]["model_kwargs"],
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
    metric = {k: v for d in test_output for k, v in d.items()}
    logger.log_hyperparams(config, metric)


if __name__ == "__main__":
    main()

import os
from typing import Any
from os import path as osp

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
from config import Config
from constants import EXPERIMENTS_PATH
from datamodule import OCRDM
from lightning_module import OCRModule
from clearml import Task


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config):
    datamodule = OCRDM(config.data_config)
    model = OCRModule(config)

    experiment_save_path = osp.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())
    
    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(),
            EarlyStopping(monitor=config.monitor_metric, patience=15, mode=config.monitor_mode),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()

    torch.set_float32_matmul_precision('medium')
    seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)

import math
import os
import re
import shutil
from typing import List, Literal, Optional

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    ModelCheckpoint,
)
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

from omnixas.data import MLSplits
from omnixas.model.training import LightningXASData, PlModule
from omnixas.model.xasblock import XASBlock
from omnixas.utils.lightning import SuppressLightningLogs, TensorboardLogTestTrainLoss


def warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
    eta_min: float,
    start_factor: float,
):
    warmup_epochs = max(0, int(warmup_epochs))
    max_epochs = max(1, int(max_epochs))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=eta_min,
        )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup_epochs),
                eta_min=eta_min,
            ),
        ],
        milestones=[warmup_epochs],
    )


class XASBlockRegressorConfig(BaseModel):
    directory: str = "checkpoints"

    # Model params
    input_dim: int = 64
    output_dim: int = 200
    hidden_dims: List[int] = [100]
    initial_lr: float = 1e-2
    min_lr: float = 1e-8
    batch_size: int = 128
    use_lr_finder: bool = True
    lr_scheduler: Literal[
        "none",
        "cosine",
        "plateau",
        "warmup_cosine",
        "onecycle",
    ] = "none"
    cosine_t_max: Optional[int] = None
    cosine_eta_min: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_min_lr: float = 1e-6
    warmup_epochs: int = 10
    warmup_start_factor: float = 0.1
    onecycle_max_lr: Optional[float] = None
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    use_early_stopping: bool = True
    monitor_metric: str = "val_loss"
    shuffle: bool = False

    # Training params
    max_epochs: int = 10
    early_stopping_patience: int = 25

    # delete save_dir
    overwrite_save_dir: bool = True

    @property
    def save_dir(self):
        return f"{self.directory}/"

    def fetch_checkpoint(
        self,
        ckpt_type: Literal["best", "last"] = "best",
    ):
        pattern = re.compile(f"{ckpt_type}.*.ckpt")
        files = os.listdir(self.save_dir)
        files = [f for f in files if pattern.match(f)]
        if len(files) != 1:
            logger.error(
                f"Found {len(files)} files in {self.save_dir} matching {pattern}"
            )
            raise FileNotFoundError
        return f"{self.save_dir}{files[0]}"

    @property
    def callbacks(self) -> List:
        if self.overwrite_save_dir and os.path.exists(self.save_dir):
            msg = f"Overwriting directory {self.save_dir}. "
            msg += "Set overwrite_save_dir=False to prevent this."
            logger.warning(msg)
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        callbacks = []
        if self.use_lr_finder:
            callbacks.append(LearningRateFinder(min_lr=self.min_lr))
        callbacks.append(TensorboardLogTestTrainLoss())
        if self.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.monitor_metric,
                    patience=self.early_stopping_patience,
                    mode="min",
                )
            )
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.save_dir,
                filename=f"best-model-{{epoch:02d}}-{{{self.monitor_metric}:.4f}}",
                monitor=self.monitor_metric,
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=True,
                save_last=True,
            )
        )
        return callbacks


class XASBlockRegressor:
    def __init__(
        self,
        **kwargs,
    ):
        self.cfg = XASBlockRegressorConfig(**kwargs)
        self.model = PlModule(
            XASBlock(
                input_dim=self.cfg.input_dim,
                hidden_dims=self.cfg.hidden_dims,
                output_dim=self.cfg.output_dim,
            ),
            **self._pl_module_kwargs,
        )

    @property
    def _pl_module_kwargs(self):
        kwargs = {"lr": self.cfg.initial_lr}
        if self.cfg.lr_scheduler == "cosine":
            kwargs.update(
                lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
                lr_scheduler_kwargs={
                    "T_max": self.cfg.cosine_t_max or self.cfg.max_epochs,
                    "eta_min": self.cfg.cosine_eta_min,
                },
                lr_scheduler_interval="epoch",
            )
        elif self.cfg.lr_scheduler == "plateau":
            kwargs.update(
                lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler_kwargs={
                    "mode": "min",
                    "factor": self.cfg.plateau_factor,
                    "patience": self.cfg.plateau_patience,
                    "min_lr": self.cfg.plateau_min_lr,
                },
                lr_scheduler_interval="epoch",
                lr_scheduler_frequency=2,
                lr_scheduler_monitor=self.cfg.monitor_metric,
            )
        elif self.cfg.lr_scheduler == "warmup_cosine":
            kwargs.update(
                lr_scheduler=warmup_cosine_scheduler,
                lr_scheduler_kwargs={
                    "warmup_epochs": self.cfg.warmup_epochs,
                    "max_epochs": self.cfg.cosine_t_max or self.cfg.max_epochs,
                    "eta_min": self.cfg.cosine_eta_min,
                    "start_factor": self.cfg.warmup_start_factor,
                },
                lr_scheduler_interval="epoch",
            )
        elif self.cfg.lr_scheduler == "onecycle":
            kwargs.update(
                lr_scheduler=torch.optim.lr_scheduler.OneCycleLR,
                lr_scheduler_kwargs={
                    "max_lr": self.cfg.onecycle_max_lr or self.cfg.initial_lr,
                    "pct_start": self.cfg.onecycle_pct_start,
                    "div_factor": self.cfg.onecycle_div_factor,
                    "final_div_factor": self.cfg.onecycle_final_div_factor,
                },
                lr_scheduler_interval="step",
            )
        return kwargs

    def _configure_fit_dependent_scheduler(self, ml_split: MLSplits):
        if self.cfg.lr_scheduler != "onecycle":
            return
        steps_per_epoch = math.ceil(len(ml_split.train) / self.cfg.batch_size)
        if steps_per_epoch <= 0:
            raise ValueError("onecycle scheduler requires non-empty train data")
        self.model.lr_scheduler_kwargs["total_steps"] = (
            self.cfg.max_epochs * steps_per_epoch
        )

    @property
    def trainer(self):
        return Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator="auto",
            devices=1,
            check_val_every_n_epoch=2,
            log_every_n_steps=1,
            callbacks=self.cfg.callbacks,
            default_root_dir=self.cfg.save_dir,
        )

    def fit(self, ml_split: MLSplits):
        self._configure_fit_dependent_scheduler(ml_split)
        trainer = self.trainer
        data_module = LightningXASData(
            ml_splits=ml_split,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
        )
        trainer.fit(self.model, data_module)
        logger.info(f"Best models saved at {self.cfg.fetch_checkpoint('best')}")
        logger.info(f"Best validation loss: {trainer.callback_metrics['val_loss']}")
        return self

    def predict(self, X: np.array):
        dummy_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.empty_like(torch.tensor(X, dtype=torch.float32)),
            )
        )
        with SuppressLightningLogs():
            trainer = Trainer(enable_progress_bar=False, callbacks=[])
            self.model.eval()
            preds = trainer.predict(self.model, dataloaders=dummy_dataloader)
        return np.array([p.detach().cpu().numpy().squeeze() for p in preds])

    def load(self, ckpt_path: Literal["best", "last"] = "best"):
        ckpt_path = self.cfg.fetch_checkpoint(ckpt_path)
        logger.info(f"Loading model from {ckpt_path}")
        self.model = PlModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=XASBlock(
                input_dim=self.cfg.input_dim,
                hidden_dims=self.cfg.hidden_dims,
                output_dim=self.cfg.output_dim,
            ),
            **self._pl_module_kwargs,
        )
        return self

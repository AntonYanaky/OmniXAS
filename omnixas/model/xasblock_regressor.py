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
    lr_scheduler: Literal["none", "cosine"] = "none"
    cosine_t_max: Optional[int] = None
    cosine_eta_min: float = 1e-6
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
        return kwargs

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

# %%

import numpy as np
from typing import Callable, Optional, Dict, Any

import hydra
import lightning
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

from omnixas.data import MLData, MLSplits
from omnixas.data.scaler import ScaledMlSplit
from omnixas.utils.callbacks import TensorboardLogTestTrainLoss
from omnixas.data.scaler import IdentityScaler

# %%


class LightningXASData(lightning.LightningDataModule):
    def __init__(
        self,
        ml_splits: MLSplits,
        batch_size: int,
        x_scaler: type = IdentityScaler,
        y_scaler: type = IdentityScaler,
        **kwargs,
    ):
        super().__init__()
        self.splits = ml_splits
        self.batch_size = batch_size
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.kwargs = kwargs

    @staticmethod
    def to_tensor(arr: np.ndarray):
        # used by external classes for consistent dtype
        return tensor(arr, dtype=torch.float32)

    @staticmethod
    def to_tensor_dataset(split: MLData):
        return TensorDataset(
            LightningXASData.to_tensor(split.X),
            LightningXASData.to_tensor(split.y),
        )

    def setup(self, stage: str = None, seed: int = 42):
        scaled_splits = ScaledMlSplit(
            x_scaler=self.x_scaler(),
            y_scaler=self.y_scaler(),
            **self.splits.dict(),
        ).shuffled_view(seed)
        self.train = self.to_tensor_dataset(scaled_splits.train)
        self.val = self.to_tensor_dataset(scaled_splits.val)
        self.test = self.to_tensor_dataset(scaled_splits.test)
        return self

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, **{**self.kwargs, "shuffle": False}
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, **{**self.kwargs, "shuffle": False}
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, **{**self.kwargs, "shuffle": False}
        )


class PlModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_metric: Callable[[], torch.nn.Module] = torch.nn.MSELoss,
        optimizer: Callable = torch.optim.Adam,
        lr: Optional[float] = 0.0001,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
        lr_scheduler_monitor: Optional[str] = None,
    ):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.loss = loss_metric()
        self.model = model
        self._val_spectral_mse = []

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
        }
        if self.lr_scheduler_monitor is not None:
            scheduler_config["monitor"] = self.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def forward(self, x):
        return self.model(x)

    def logged_loss(self, name, y, y_pred):
        loss = self.loss(y, y_pred)
        self.log(name, loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.logged_loss("train_loss", y, self.model(x))

    def on_validation_epoch_start(self):
        self._val_spectral_mse = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        self._val_spectral_mse.append(torch.mean((y - y_pred) ** 2, dim=1).detach())
        return self.logged_loss("val_loss", y, y_pred)

    def on_validation_epoch_end(self):
        if self._val_spectral_mse:
            val_median_mse = torch.cat(self._val_spectral_mse).median()
            self.log("val_median_mse", val_median_mse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.logged_loss("test_loss", y, self.model(x))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.model(x)


@hydra.main(version_base=None)
def trainModel(cfg: DictConfig):
    # torch.set_float32_matmul_precision('medium') # useful for tensor cores
    data_module = instantiate(cfg.data_module)
    module = instantiate(cfg.module)
    trainer = instantiate(cfg.trainer)
    trainer.callbacks.extend([TensorboardLogTestTrainLoss()])
    trainer.fit(module, data_module)
    trainer.test(module, datamodule=data_module)


# %%


if __name__ == "__main__":
    trainModel()

from typing import Optional

import pytorch_lightning as pl
from rtpt import RTPT


class LightningRtpt(pl.Callback):
    def __init__(
        self,
        name_initials: str,
        experiment_name: str,
        max_iterations: int,
        iteration_start=0,
        moving_avg_window_size=20,
        update_interval=1,
        precision=2
    ):
        super().__init__()

        self.rtpt = RTPT(
            name_initials=name_initials,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
            iteration_start=iteration_start,
            moving_avg_window_size=moving_avg_window_size,
            update_interval=update_interval,
            precision=precision
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional = None) -> None:
        self.rtpt.step()

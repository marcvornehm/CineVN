"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


class ContinuousTensorBoardLogger(TensorBoardLogger):
    """
    TensorBoardLogger that does not create a new version for a new run
    """

    @property
    def version(self) -> int:
        return 0


class TQDMProgressBarWithoutVersion(TQDMProgressBar):
    """
    Progress bar that does not display a version number
    """

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if 'v_num' in metrics:
            del metrics['v_num']
        return metrics



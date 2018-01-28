import logging
import os
import shutil
import time

import torch

from src.models.conversational.utils import APP_NAME


class Checkpoint(object):
    """Manage model checkpoint saving and loading."""

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, optimizer, epoch, step):
        """Initialize checkpoint parameters.

        Args:
            model (Seq2Seq): Model object to be saved.
            optimizer (Optimizer): Optimizer object to be saved.
            epoch (int): Current epoch.
            step (int): Current step.
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.logger = logging.getLogger(APP_NAME + '.Checkpoint')

    def save(self, experiment_path):
        """Save current model into a subdirectory of the save directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.

        Args:
            experiment_path (str): Path to the experiment checkpoint directory.

        Returns:
            str: Path to the checkpoint subdirectory.

        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_path, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.logger.info("Checkpoint procedure started")
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                    },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))
        self.logger.info("Checkpoint saved to %s" % path)

        return path

    @classmethod
    def load(cls, path):
        """Load a Checkpoint object from disk.

        Args:
            path (str): Path to the checkpoint subdirectory.

        Returns:
            Checkpoint: Checkpoint object with fields copied from those stored on disk.

        """
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),
                                           map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=lambda storage, loc: storage)

        model.flatten_parameters()  # make RNN parameters contiguous
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'])

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """Return the most recent checkpoint subdirectory path.

        Args:
            experiment_path (str): Path to the experiment checkpoint directory.

        Returns:
             str: Path to the last saved checkpoint's subdirectory.
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])

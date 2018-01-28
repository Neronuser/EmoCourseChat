import itertools

import torch


class Optimizer(object):
    """Encapsulate torch.optim.Optimizer with max_grad_norm."""

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, max_grad_norm=0):
        """Save torch.optim.optimizer reference and required norm.

        Args:
            optim (torch.optim.Optimizer): Optimizer object, the parameters to be optimized
                should be given when instantiating the object, e.g. torch.optim.SGD(params)
            max_grad_norm (Optional[float]): Value used for gradient norm clipping, set 0 to disable. Defaults to 0.

        """
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        """Set the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler): object of learning rate scheduler,
                e.g. torch.optim.lr_scheduler.StepLR

        """
        self.scheduler = scheduler

    def step(self):
        """Performs a single optimization step, including gradient norm clipping if necessary."""
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss):
        """Update the learning rate if the criteria of the scheduler are met.

        Args:
            loss (float): The current loss.  It could be training loss or developing loss depending on the caller.
                By default the supervised trainer uses developing loss.

        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

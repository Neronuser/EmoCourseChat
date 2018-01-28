import math

import numpy as np
import torch.nn as nn


class Loss(object):
    """ Base class for loss functions."""

    def __init__(self, name, criterion):
        """Defines criterion and logging name of the object.

        Args:
            name (str): Loss name for logging.
            criterion (torch.nn._Loss): Torch loss criterion.

        """
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """Reset the accumulated loss."""
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """Loss calculation.

        Returns:
            loss (float): Value of the loss.

        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """Accumulate batch loss.

        Args:
            outputs (torch.Tensor): Outputs of a batch.
            target (torch.Tensor): Expected output of a batch.

        """
        raise NotImplementedError

    def cuda(self):
        """Transfer to GPU."""
        self.criterion.cuda()

    def backward(self):
        """Backpropagate loss."""
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()


class NLLLoss(Loss):
    """Batch averaged negative log-likelihood loss."""

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        """Check if both weight and mask are given.

        Args:
            weight (Optional[torch.Tensor]): Tensor assigning weight to each of the classes. Defaults to None.
            mask (Optional[int]): Index of masked token(padding). Defaults to None.
            size_average (Optional[bool]): Averaging across observations if True, else - sum. Defaults to True.

        """
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data[0]
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1


class Perplexity(NLLLoss):
    """ Language model perplexity loss. Perplexity is the token averaged likelihood.
    When the averaging options are the same, it is the exponential of negative log-likelihood.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None):
        """Check if both weight and mask are given.

        Args:
            weight (Optional[torch.Tensor]): Tensor assigning weight to each of the classes. Defaults to None.
            mask (Optional[int]): Index of masked token(padding). Defaults to None.

        """
        super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)

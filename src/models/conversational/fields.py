import torch
from torch.autograd import Variable
from torchtext.data import Field


class EncodedSentenceField(Field):
    """Dataset field of already encoded sequences. Pad and convert into torch.autograd.Variable."""

    def __init__(self, **args):
        """Call Field.__init__().

        Args:
            **args: See Field.__init__().

        """
        super(EncodedSentenceField, self).__init__(**args)

    def preprocess(self, x):
        """No preprocessing needed."""
        return x

    def process(self, batch, device, train):
        """Pad a batch and convert into torch.autograd.Variable.

        Args:
            batch (list(object)): A list of examples in a batch.
            device (int): None if torch.cuda.is_available() else -1.
            train (bool): Whether the batch is from a training set.

        Returns:
            torch.autograd.Variable: Input to further models.

        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device, train=train)
        return tensor

    def pad(self, batch):
        """Pad a batch with pad_token.

        Args:
            batch (list(object)): A list of examples in a batch.

        Returns:
            list(object): List of padded examples if self.include_lengths is False.
            (list(object), list(int)): List of padded examples and list of their lengths otherwise.

        """
        batch = list(batch)
        if not self.sequential:
            return batch
        max_len = max(len(x) for x in batch)
        padded, lengths = [], []
        for x in batch:
            pad_length = max(0, max_len - len(x))
            padded.append(
                list(x[:max_len]) +
                [self.pad_token] * pad_length)
            lengths.append(len(padded[-1]) - pad_length)
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None, train=True):
        """Convert a the example into a torch.autograd.Variable.

        Args:
            arr (list(object)): A list of examples in a batch.
            device (int): None if torch.cuda.is_available() else -1. Defaults to None.
            train (bool): Whether the batch is from a training set. Defaults to True.

        Returns:
            torch.autograd.Variable: examples ready for modelling if self.include_lengths is False.
            torch.autograd.Variable, torch.LongTensor: examples ready for modelling,
                example lengths if self.include_lengths is True.

        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        arr = self.tensor_type(arr)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        if self.include_lengths:
            return Variable(arr, volatile=not train), lengths
        return Variable(arr, volatile=not train)

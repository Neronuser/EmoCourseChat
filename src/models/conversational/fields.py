import torch
from torch.autograd import Variable
from torchtext.data import Field


class EncodedSentenceField(Field):
    def __init__(self, **args):
        super(EncodedSentenceField, self).__init__(**args)

    def preprocess(self, x):
        return x

    def process(self, batch, device, train):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device, train=train)
        return tensor

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        max_len = max(len(x) for x in minibatch)
        padded, lengths = [], []
        for x in minibatch:
            pad_length = max(0, max_len - len(x))
            padded.append(
                list(x[:max_len]) +
                [self.eos_token] * pad_length)
            lengths.append(len(padded[-1]) - pad_length)
            # TODO might be wrong because of EOS instead of PAD
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None, train=True):
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

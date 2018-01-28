import torch
import torchtext

from src.models.conversational.emotion_dialogue_dataset import UTTERANCE_FIELD_NAME, RESPONSE_FIELD_NAME
from src.models.conversational.loss import NLLLoss
from src.models.conversational.utils import PAD_INDEX


class Evaluator(object):
    """Evaluate models with given datasets."""

    def __init__(self, loss=NLLLoss(), batch_size=64):
        """Initialize the evaluator with loss and batch size.

        Args:
            loss (Optional[seq2seq.loss]): Loss to count. Defaults to seq2seq.loss.NLLLoss.
            batch_size (Optional[int]): Batch size. Defaults to 64.

        """
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """Evaluate a model on given dataset and return performance.
        Args:
            model (seq2seq.models): Model to evaluate.
            data (torchtext.data.Dataset): Dataset to evaluate against.
        Returns:
            loss (float): Loss of the given model on the given dataset.

        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        pad = PAD_INDEX

        for batch in batch_iterator:
            input_variables, input_lengths = getattr(batch, UTTERANCE_FIELD_NAME)
            target_variables = getattr(batch, RESPONSE_FIELD_NAME)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy

import logging
import os
import random

import torch
import torchtext
from torch import optim

from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.emotion_dialogue_dataset import UTTERANCE_FIELD_NAME, RESPONSE_FIELD_NAME
from src.models.conversational.evaluator import Evaluator
from src.models.conversational.loss import NLLLoss
from src.models.conversational.model import Seq2seq, TopKDecoder
from src.models.conversational.optimizer import Optimizer
from src.models.conversational.predictor import Predictor
from src.models.conversational.utils import APP_NAME


class Trainer(object):
    """Seq2Seq trainer class."""

    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100):
        """Set logger, random seed, dev evaluator, loss and optimizer up.

        Args:
            expt_dir (Optional[str]): Checkpoint directory path. Defaults to 'experiment'.
            loss (Optional[loss.Loss]): Loss definition to use while training. Defaults to loss.NLLLoss.
            batch_size (Optional[int]): Number of examples in a batch. Defaults to 64.
            random_seed (Optional[int]): Random seed for random and torch. Defaults to None.
            checkpoint_every (Optional[int]): Number of steps after which to checkpoint the model. Defaults to 100.
            print_every (Optional[int]): Number of steps after which to log model state. Defaults to 100.

        """
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(APP_NAME + '.Trainer')

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        """Run training for the current batch.

        Args:
            input_variable (torch.autograd.Variable): Input sequence batch.
            input_lengths (list(int)): List of input sequence lengths.
            target_variable (torch.autograd.Variable): Target sequence batch.
            model (seq2seq.models): Model to run training on.
            teacher_forcing_ratio (Optional[float]): Teaching forcing ratio.

        Returns:
            float: Batch loss.

        """
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epochs(self, data, model, n_epochs, start_epoch, start_step, dev_data, teacher_forcing_ratio, early_stopping_patience):
        """Run training for `n_epochs` epochs.

        Args:
            data (EmotionDialogueDataset): Dataset object to train on.
            model (seq2seq.models): Model to run training on.
            n_epochs (int): Number of epochs to run.
            start_epoch (int): Starting epoch number.
            start_step (int): Starting step number.
            dev_data (torchtext.Dataset): Dev dataset.
            teacher_forcing_ratio (float): Teaching forcing ratio.
            early_stopping_patience (int): Number of epochs to tolerate dev loss increase.

        """
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(data, batch_size=self.batch_size, repeat=False,
                                                       sort_key=lambda x: len(x.src),
                                                       shuffle=True, device=device, sort=False, sort_within_batch=True)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        previous_dev_loss = 10e6
        dev_loss_increased_epochs = 0
        for epoch in range(start_epoch, n_epochs + 1):
            self.logger.info("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, UTTERANCE_FIELD_NAME)
                target_variables = getattr(batch, RESPONSE_FIELD_NAME)

                loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model,
                                         teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %.2f%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    self.logger.info(log_msg)
                    beam_search = Seq2seq(model.encoder, TopKDecoder(model.decoder, 20))
                    predictor = Predictor(beam_search, data.vocabulary)
                    # seq = "how are you ?".split()
                    seq = "how are you".split()
                    self.logger.info("Beam")
                    self.logger.info(predictor.predict(seq))
                    self.logger.info("Argmax")
                    predictor = Predictor(model, data.vocabulary)
                    self.logger.info(predictor.predict(seq))

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step).save(self.expt_dir)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
                if dev_loss > previous_dev_loss:
                    dev_loss_increased_epochs += 1
                    if dev_loss_increased_epochs == early_stopping_patience:
                        self.logger.info("EARLY STOPPING")
                        break
                else:
                    previous_dev_loss = dev_loss
                    dev_loss_increased_epochs = 0
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step).save(self.expt_dir)

            else:
                self.optimizer.update(epoch_loss_avg)

            self.logger.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0, early_stopping_patience=5):
        """Run training for a given model.

        Args:
            model (seq2seq.models): Model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (EmotionDialogueDataset): Dataset object to train on.
            num_epochs (Optional[int]): Number of epochs to run. Defaults to 5.
            resume (Optional[bool]): Resume training with the latest checkpoint. Defaults to False.
            dev_data (Optional[torchtext.Dataset]): Dev dataset. Defaults to None.
            optimizer (Optional[Optimizer]): Optimizer for training.
                Defaults to Optimizer(pytorch.optim.Adam, max_grad_norm=5).
            teacher_forcing_ratio (Optional[float]): Teaching forcing ratio. Defaults to 0.
            early_stopping_patience (Optional[int]): Number of epochs to tolerate dev loss increase. Defaults to 5.

        Returns:
            model (seq2seq.models): Trained model.

        """
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epochs(data, model, num_epochs,
                           start_epoch, step, dev_data,
                           teacher_forcing_ratio, early_stopping_patience)
        return model

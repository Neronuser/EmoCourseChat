import torch
import torchtext

from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.emotion_dialogue_dataset import UTTERANCE_FIELD_NAME, RESPONSE_FIELD_NAME, \
    EMOTION_FIELD_NAME
from src.models.conversational.emotion_model import EmotionSeq2seq, EmotionTopKDecoder
from src.models.conversational.predictor import Predictor
from src.models.conversational.trainer import Trainer


class EmotionTrainer(Trainer):

    def train_batch(self, input_variable, input_lengths, target_variable, emotion_variable, model):
        """Run training for the current batch.

        Args:
            input_variable (torch.autograd.Variable): Input sequence batch.
            input_lengths (list(int)): List of input sequence lengths.
            target_variable (torch.autograd.Variable): Target sequence batch.
            emotion_variable (torch.autograd.Variable): Emotion IDs batch.
            model (seq2seq.models): Model to run training on.

        Returns:
            float: Batch loss.

        """
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable, emotion_variable)
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

    def _train_epochs(self, data, model, n_epochs, start_epoch, start_step, dev_data, teacher_forcing_ratio):
        """Run training for `n_epochs` epochs.

        Args:
            data (EmotionDialogueDataset): Dataset object to train on.
            model (seq2seq.models): Model to run training on.
            n_epochs (int): Number of epochs to run.
            start_epoch (int): Starting epoch number.
            start_step (int): Starting step number.
            dev_data (Optional[torchtext.Dataset]): Dev dataset.
            teacher_forcing_ratio (Optional[float]): Teaching forcing ratio. Not used here.

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
                emotion_variables = getattr(batch, EMOTION_FIELD_NAME)

                loss = self.train_batch(input_variables, input_lengths.tolist(), target_variables, emotion_variables,
                                        model)

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
                    beam_search = EmotionSeq2seq(model.encoder, EmotionTopKDecoder(model.decoder, 20))
                    predictor = Predictor(beam_search, data.vocabulary, data.emotion_vocabulary)
                    seq = "how are you ?".split()
                    self.logger.info("Happy: " + " ".join(predictor.predict(seq, 'happiness')))
                    self.logger.info("Angry: " + " ".join(predictor.predict(seq, 'anger')))

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
            else:
                self.optimizer.update(epoch_loss_avg)

            self.logger.info(log_msg)

import logging
import os

import torch

from src.models.conversational.EmotionDialogueDataset import EmotionDialogueDataset
from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.loss import Perplexity
from src.models.conversational.model import EncoderRNN, DecoderRNN, Seq2seq
from src.models.conversational.predictor import Predictor
from src.models.conversational.trainer import Trainer
from src.models.conversational.utils import EOS_INDEX, SOS_INDEX
from src.utils import parse_config, LOG_FORMAT
from src.models.conversational.utils import APP_NAME


def run(config):
    resume = config.getboolean('Resume')
    train_path = config["Train"]
    epochs = config.getint('Epochs')
    teacher_forcing = config.getfloat('TeacherForcing')
    print_every = config.getint('Print')
    save_every = config.getfloat('SaveEvery')
    save_dir = config['SavePath']
    learning_rate = config.getfloat('LearningRate')
    n_layers = config.getint('Layer')
    hidden_size = config.getint('Hidden')
    batch_size = config.getint('Batch_size')
    beam_size = config.getint('Beam')
    input = config.getboolean('Input')
    max_length = config.getint('MaxLength')
    max_words = config.getint('MaxWords')

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(config['LogLevel'])
    handler = logging.FileHandler(config['LogPath'])
    handler.setLevel(config['LogLevel'])
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(config)

    dataset = EmotionDialogueDataset(train_path, save_dir, max_length, max_words)

    load_checkpoint = config['LoadCheckpoint']
    if load_checkpoint:
        checkpoint_path = os.path.join(save_dir, Checkpoint.CHECKPOINT_DIR_NAME, load_checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
    else:
        weight = torch.ones(dataset.vocabulary.n_words)
        pad = EOS_INDEX
        loss = Perplexity(weight, pad)
        if torch.cuda.is_available():
            loss.cuda()

        seq2seq = None
        optimizer = None
        if not resume:
            # Initialize model
            bidirectional = True
            encoder = EncoderRNN(dataset.vocabulary.n_words, max_length, hidden_size,
                                 bidirectional=bidirectional, variable_lengths=True)
            decoder = DecoderRNN(dataset.vocabulary.n_words, max_length,
                                 hidden_size * 2 if bidirectional else hidden_size,
                                 dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                 eos_id=EOS_INDEX, sos_id=SOS_INDEX)
            seq2seq = Seq2seq(encoder, decoder)
            if torch.cuda.is_available():
                seq2seq.cuda()

            for param in seq2seq.parameters():
                param.data.uniform_(-0.08, 0.08)

            # Optimizer and learning rate scheduler can be customized by
            # explicitly constructing the objects and pass to the trainer.
            #
            # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
            # scheduler = StepLR(optimizer.optimizer, 1)
            # optimizer.set_scheduler(scheduler)

        # train
        t = Trainer(loss=loss, batch_size=batch_size,
                    checkpoint_every=save_every,
                    print_every=print_every, expt_dir=save_dir)

        seq2seq = t.train(seq2seq, dataset,
                          num_epochs=epochs, dev_data=None,
                          optimizer=optimizer,
                          teacher_forcing_ratio=teacher_forcing,
                          resume=resume)

    predictor = Predictor(seq2seq, dataset.vocabulary)

    while True:
        seq_str = input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))


if __name__ == '__main__':
    config = parse_config('dialogue')
    run(config)

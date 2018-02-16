import logging
import os

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.emotion_dialogue_dataset import EmotionDialogueDataset
from src.models.conversational.emotion_model import EmotionDecoderRNN, EmotionSeq2seq, EmotionTopKDecoder
from src.models.conversational.emotion_trainer import EmotionTrainer
from src.models.conversational.loss import NLLLoss, Perplexity
from src.models.conversational.model import EncoderRNN
from src.models.conversational.optimizer import Optimizer
from src.models.conversational.predictor import Predictor
from src.models.conversational.utils import APP_NAME
from src.models.conversational.utils import EOS_INDEX, SOS_INDEX, PAD_INDEX
from src.utils import parse_config, LOG_FORMAT


def run(config):
    resume = config.getboolean('Resume')
    train_path = config["Train"]
    epochs = config.getint('Epochs')
    print_every = config.getint('Print')
    save_every = config.getfloat('SaveEvery')
    save_dir = config['SavePath']
    learning_rate = config.getfloat('LearningRate')
    n_layers = config.getint('Layer')
    embeddings_dim = config.getint('EmbeddingsDim')
    emotion_embeddings_dim = config.getint('EmotionEmbeddingsDim')
    hidden_size = config.getint('Hidden')
    batch_size = config.getint('BatchSize')
    beam_size = config.getint('Beam')
    max_length = config.getint('MaxLength')
    max_words = config.getint('MaxWords')

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(config['LogLevel'])
    handler = logging.FileHandler(config['LogPath'])
    handler.setLevel(config['LogLevel'])
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(dict(config.items()))

    dataset = EmotionDialogueDataset(train_path, save_dir, max_length, max_words)

    load_checkpoint = config['LoadCheckpoint']
    if load_checkpoint:
        checkpoint_path = os.path.join(save_dir, Checkpoint.CHECKPOINT_DIR_NAME, load_checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
    else:
        weight = torch.ones(dataset.vocabulary.n_words)
        pad = PAD_INDEX
        loss = Perplexity(weight, pad)
        if torch.cuda.is_available():
            loss.cuda()

        seq2seq = None
        optimizer = None
        if not resume:
            # Initialize model
            bidirectional = False
            encoder = EncoderRNN(dataset.vocabulary.n_words, max_length, embeddings_dim, hidden_size,
                                 bidirectional=bidirectional, variable_lengths=True, n_layers=n_layers, rnn_cell='lstm')
            decoder = EmotionDecoderRNN(dataset.vocabulary.n_words, dataset.emotion_vocabulary.n_words, max_length,
                                        embeddings_dim,
                                        emotion_embeddings_dim, hidden_size * 2 if bidirectional else hidden_size,
                                        dropout_p=0, use_attention=True, bidirectional=bidirectional,
                                        eos_id=EOS_INDEX, sos_id=SOS_INDEX, n_layers=n_layers, rnn_cell='lstm')
            seq2seq = EmotionSeq2seq(encoder, decoder)
            if torch.cuda.is_available():
                seq2seq.cuda()

            for param in seq2seq.parameters():
                param.data.uniform_(-0.1, 0.1)

            # Optimizer and learning rate scheduler can be customized by
            # explicitly constructing the objects and pass to the trainer.
            #
            optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=learning_rate), max_grad_norm=1)
            # optimizer = Optimizer(SGD(seq2seq.parameters(), lr=0.1),
            #                       max_grad_norm=1)
            # scheduler = StepLR(optimizer.optimizer, 1)
            # optimizer.set_scheduler(scheduler)

        # train
        t = EmotionTrainer(loss=loss, batch_size=batch_size,
                           checkpoint_every=save_every,
                           print_every=print_every, expt_dir=save_dir)

        seq2seq = t.train(seq2seq, dataset,
                          num_epochs=epochs, dev_data=None,
                          optimizer=optimizer,
                          resume=resume)

    beam_search = EmotionSeq2seq(seq2seq.encoder, EmotionTopKDecoder(seq2seq.decoder, beam_size))
    predictor = Predictor(beam_search, dataset.vocabulary, emotion_vocabulary=dataset.emotion_vocabulary)

    seq = "how are you ?".split()
    logger.info("Happy: " + " ".join(predictor.predict(seq, 'happiness')))
    logger.info("Angry: " + " ".join(predictor.predict(seq, 'anger')))
    logger.info("Sad: " + " ".join(predictor.predict(seq, 'sadness')))
    logger.info("Neutral: " + " ".join(predictor.predict(seq, 'neutral')))
    logger.info("Love: " + " ".join(predictor.predict(seq, 'love')))


if __name__ == '__main__':
    config = parse_config('dialogue')
    run(config)

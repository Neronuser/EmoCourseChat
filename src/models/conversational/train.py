# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

import itertools
import random
import math
import sys
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from load import loadPrepareData

from src.models.conversational.prepare_data import load_prepare_data

from src.models.conversational.EmotionDialogueDataset import EmotionDialogueDataset
from src.models.conversational.utils import SOS_INDEX, binary_mask, pad, indexes_from_sentence
from model import EncoderRNN, LuongAttnDecoderRNN, Attn
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir

cudnn.benchmark = True
USE_CUDA = torch.cuda.is_available()


#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
    filename = ''
    if reverse:
        filename += 'reverse_'
    filename += obj
    return filename


# convert to index, add EOS
# return input pack_padded_sequence
def input_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    pad_list = pad(indexes_batch)
    pad_var = Variable(torch.LongTensor(pad_list))
    return pad_var, lengths


# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def output_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = pad(indexes_batch)
    mask = binary_mask(pad_list)
    mask = Variable(torch.ByteTensor(mask))
    pad_var = Variable(torch.LongTensor(pad_list))
    return pad_var, mask, max_target_len


# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch_data(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        output_batch.append(pair_batch[i][1])
    input, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return input, lengths, output, mask, max_target_len


#############################################
# Training
#############################################

def masked_cross_entropy(input, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.cuda() if USE_CUDA else loss
    return loss, n_total.data[0]


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = Variable(torch.LongTensor([[SOS_INDEX for _ in range(batch_size)]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1)  # Next input is current target
            mask_loss, n_total = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.data.topk(1)  # [64, 1]

            decoder_input = Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            mask_loss, n_total = masked_cross_entropy(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * n_total)
            n_totals += n_total

    loss.backward()

    clip = 50.0
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_iterations(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                     print_every, save_every, max_length, max_words, load_filename=None, attn_model='dot', decoder_learning_ratio=5.0):
    dataset = EmotionDialogueDataset(corpus, save_dir, max_length, max_words)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration,
                                                                         filename(reverse, 'training_batches'),
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch_data(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                            for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration,
                                                                        filename(reverse, 'training_batches'),
                                                                        batch_size)))
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(dataset.vocabulary.n_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)
    if load_filename:
        checkpoint = torch.load(load_filename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if load_filename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if load_filename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            # perplexity.append(print_loss_avg)
            # plotPerplexity(perplexity, iteration)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if iteration % save_every == 0:
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))

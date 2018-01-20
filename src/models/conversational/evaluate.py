import torch
from torch.autograd import Variable
import random
from train import indexesFromSentence
from load import SOS_token, EOS_token
from load import MAX_LENGTH, loadPrepareData, Voc
from model import *
import sys

from src.models.conversational.utils import SOS_INDEX

USE_CUDA = torch.cuda.is_available()


class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_INDEX, sentence_idxes=None, sentence_scores=None):
        self.sentence_idxes = [] if sentence_idxes is None else sentence_idxes
        self.sentence_scores = [] if sentence_scores is None else sentence_idxes
        if len(self.sentence_idxes) != len(self.sentence_scores):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx

    def avg_score(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def add_top_k(self, top_i, top_v, decoder_hidden, beam_size, voc):
        top_v = torch.log(top_v)
        terminates, sentences = [], []
        for i in range(beam_size):
            if top_i[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avg_score()))  # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:]  # pass by value
            scores = self.sentence_scores[:]  # pass by value
            idxes.append(top_i[0][i])
            scores.append(top_v[0][i])
            sentences.append(Sentence(decoder_hidden, top_i[0][i], idxes, scores))
        return terminates, sentences

    def to_word_score(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i]])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return words, self.avg_score()


def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.data.topk(beam_size)
            term, top = sentence.add_top_k(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avg_score(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.to_word_score(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]


def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):
    decoder_input = Variable(torch.LongTensor([[SOS_INDEX]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)  # or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        topv, topi = decoder_output.data.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]  # [1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
    input_batch = input_batch.cuda() if USE_CUDA else input_batch

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluate_randomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        if beam_size == 1:
            output_words, attentions = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))


def evaluate_input(encoder, decoder, voc, beam_size):
    pair = ''
    while 1:
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, attentions = evaluate(encoder, decoder, voc, pair, beam_size)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def run_test(n_layers, hidden_size, reverse, model_file, beam_size, input, corpus):
    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

    checkpoint = torch.load(model_file)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False)
    decoder.train(False)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    if input:
        evaluate_input(encoder, decoder, voc, beam_size)
    else:
        evaluate_randomly(encoder, decoder, voc, pairs, reverse, beam_size, 20)

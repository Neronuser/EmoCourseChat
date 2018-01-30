from src.models.conversational.model import Seq2seq, BaseRNN, Attention, DecoderRNN, inflate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EmotionSeq2seq(Seq2seq):

    def forward(self, input_variable, input_lengths=None, target_variable=None, target_emotion=None):
        """Seq2seq forward pass.

        Args:
            input_variable (list): List of token IDs.
            input_lengths (Optional[list(int)]): Lengths of sequences. Defaults to None.
            target_variable (Optional[list]): List of token IDs. Defaults to None.
            target_emotion (Optional[list]): List of emotion IDs. Defaults to None.

        Returns:
            torch.Tensor(seq_len, batch, vocab_size): Decoding outputs.
            torch.Tensor(num_layers * num_directions, batch, hidden_size): Hidden state.
            dict: {*KEY_LENGTH* : lengths of output sequences, *KEY_SEQUENCE* : predicted token IDs}.

        """
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              emotion_inputs=target_emotion,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function)
        return result


class EmotionDecoderRNN(BaseRNN):
    """Provides decoding in a seq2seq framework, with a controllable input(emotion).

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, emotion_vocab_size, max_len, embeddings_dim, emotion_embeddings_dim, hidden_size,
                 sos_id, eos_id, n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        """Build decoder layers.

        Args:
            vocab_size (int): Size of the vocabulary.
            emotion_vocab_size (int): Size of the emotion vocabulary.
            max_len (int): A maximum allowed length for the sequence to be processed.
            embeddings_dim (int): The size of embeddings vectors.
            emotion_embeddings_dim (int): The size of emotion embedding vectors.
            hidden_size (int): The number of features in the hidden state `h`.
            sos_id (int): Index of the start of sentence symbol.
            eos_id (int): Index of the end of sentence symbol.
            n_layers (Optional[int]): Number of recurrent layers. Defaults to 1.
            rnn_cell (Optional[str]): Type of RNN cell. Defaults to gru.
            bidirectional (Optional[bool]): If the encoder is bidirectional. Defaults to False.
            input_dropout_p (Optional[float]): Dropout probability for the input sequence. Defaults to 0.
            dropout_p (Optional[float]): Dropout probability for the output sequence. Defaults to 0.
            use_attention(Optional[bool]): Flag indication whether to use attention mechanism or not. Defaults to False.

        """
        super().__init__(vocab_size, max_len, embeddings_dim, hidden_size,
                         input_dropout_p, dropout_p,
                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embeddings_dim + emotion_embeddings_dim, hidden_size, n_layers, batch_first=True,
                                 dropout=dropout_p)

        self.output_size = vocab_size
        self.emotion_vocab_size = emotion_vocab_size
        self.emotion_embeddings_dim = emotion_embeddings_dim
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.embeddings_dim)
        self.emotion_embedding = nn.Embedding(self.emotion_vocab_size, self.emotion_embeddings_dim)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, input_emotion, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        embedded_emotion = self.emotion_embedding(input_emotion)
        embedded = torch.cat((embedded, embedded_emotion.view((batch_size, 1, self.emotion_embeddings_dim))), dim=2)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.view(-1, self.hidden_size)), dim=1).view(batch_size, output_size,
                                                                                              -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, emotion_inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax):
        """Decoder forward pass.

        Args:
            inputs (Optional[torch.Tensor(batch, seq_len, input_size)]): List of token IDs for teacher forcing.
                Defaults to None.
            emotion_inputs (Optional[torch.Tensor(batch, seq_len, input_size)]): Emotion category.
                Defaults to None.
            encoder_hidden (Optional[torch.Tensor(num_layers * num_directions, batch_size, hidden_size)]): Hidden
                state `h` of encoder for the initial hidden state of the decoder. Defaults to None.
            encoder_outputs (Optional[torch.Tensor(batch, seq_len, hidden_size)]): Outputs of the encoder
                for attention mechanism. Defaults to None.
            function (Optional[torch.nn.Module]): Function to generate symbols from RNN hidden state.
                Defaults to `torch.nn.functional.log_softmax`.

        Returns:
            torch.Tensor(seq_len, batch, vocab_size): Decoding outputs.
            torch.Tensor(num_layers * num_directions, batch, hidden_size): Hidden state.
            dict: {*KEY_LENGTH* : lengths of output sequences, *KEY_SEQUENCE* : predicted token IDs}.
        """
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function)
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        decoder_input = inputs[:, 0].unsqueeze(1)
        for di in range(max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, emotion_inputs, decoder_hidden,
                                                                          encoder_outputs,
                                                                          function=function)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """Initialize the encoder hidden state."""
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            inputs = Variable(torch.LongTensor([self.sos_id] * batch_size),
                              volatile=True).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


class EmotionTopKDecoder(torch.nn.Module):
    """Top-K decoding with beam search."""

    def __init__(self, decoder_rnn, k):
        """Builds the decoder layers.

        Args:
            decoder_rnn (EmotionDecoderRNN): An object of EmotionDecoderRNN used for decoding.
            k (int): Size of the beam.

        """
        super().__init__()
        self.rnn = decoder_rnn
        self.k = k
        self.hidden_size = self.rnn.hidden_size
        self.V = self.rnn.output_size
        self.SOS = self.rnn.sos_id
        self.EOS = self.rnn.eos_id

    def forward(self, inputs=None, emotion_inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax,
                teacher_forcing_ratio=0, retain_output_probs=True):
        """
        Args:
            inputs (Optional[torch.Tensor(batch, seq_len, input_size)]): List of token IDs for teacher forcing.
                Defaults to None.
            emotion_inputs (Optional[torch.Tensor(batch, 1)]): Response emotion IDs. Defaults to None.
            encoder_hidden (Optional[torch.Tensor(num_layers * num_directions, batch_size, hidden_size)]): Hidden
                state `h` of encoder for the initial hidden state of the decoder. Defaults to None.
            encoder_outputs (Optional[torch.Tensor(batch, seq_len, hidden_size)]): Outputs of the encoder
                for attention mechanism. Defaults to None.
            function (Optional[torch.nn.Module]): Function to generate symbols from RNN hidden state.
                Defaults to `torch.nn.functional.log_softmax`.
            teacher_forcing_ratio (Optional[float]): Probability of teacher forcing. Defaults to 0.
            retain_output_probs (bool): If doing local backpropagation, retain the output layer. Defaults to True.

        Returns:
            torch.Tensor(seq_len, batch, vocab_size): Decoding outputs.
            torch.Tensor(num_layers * num_directions, batch, hidden_size): Hidden state.
            {}: {*length* : lengths of output sequences, *topk_length*: lengths of beam search sequences,
                *sequence* : list of predicted token IDs, *topk_sequence* : list of token IDs from beam search,
                *inputs* : target outputs if provided for decoding}.

        """
        inputs, batch_size, max_length = self.rnn._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                 function)

        pos_index = torch.LongTensor(range(batch_size)) * self.k
        if torch.cuda.is_available():
            pos_index = pos_index.cuda()
        self.pos_index = Variable(pos_index).view(-1, 1)

        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([inflate(h, self.k, 1) for h in encoder_hidden])
            else:
                hidden = inflate(encoder_hidden, self.k, 1)

        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            inflated_encoder_outputs = inflate(encoder_outputs, self.k, 0)
        else:
            inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        if torch.cuda.is_available():
            sequence_scores = sequence_scores.cuda()
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        input_seq = torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1)
        if torch.cuda.is_available():
            input_seq = input_seq.cuda()
        input_var = Variable(input_seq)

        emotion_inputs = inflate(emotion_inputs, self.k, dim=0)

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _ in range(0, max_length):

            # Run the RNN one step forward
            log_softmax_output, hidden, _ = self.rnn.forward_step(input_var, emotion_inputs, hidden,
                                                                  inflated_encoder_outputs, function=function)

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = inflate(sequence_scores, self.V, 1)
            sequence_scores += log_softmax_output.squeeze(1)
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        if isinstance(h_n, tuple):
            decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
        else:
            decoder_hidden = h_n[:, :, 0, :]
        metadata = {}
        metadata['inputs'] = inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[0] for seq in p]
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.

        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network.
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network.
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors.
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens.
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores
                for every token t = [0, ... , seq_len - 1].
            b (int): Size of the batch.
            hidden_size (int): Size of the hidden state.

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
                from the last layer of the RNN, for every n = [0, ... , seq_len - 1].
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
                from the last layer of the RNN, for every n = [0, ... , seq_len - 1].
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences.
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates.
            p (batch, k, sequence_len): A Tensor containing predicted sequence.
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())
        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
        # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b  # the number of EOS found
        # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #

            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.data[0]] for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in
                   reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            if torch.cuda.is_available():
                h_n = h_n.index_select(1, re_sorted_idx.data.cpu()).view(-1, b, self.k, hidden_size)
            else:
                h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
        score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)

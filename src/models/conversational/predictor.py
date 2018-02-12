import torch
from torch.autograd import Variable


class Predictor(object):

    def __init__(self, model, vocabulary, emotion_vocabulary=None):
        """Inference class for a given model.

        Args:
            model (seq2seq.models): Trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`.
            vocabulary (utils.Vocabulary): Language vocabulary.
            emotion_vocabulary (Optional[utils.Vocabulary]): Emotion vocabulary. Defaults to None.

        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.vocabulary = vocabulary
        self.emotion_vocabulary = emotion_vocabulary

    def predict(self, src_seq, emotion=None):
        """Predict given `src_seq` as input.

        Args:
            src_seq (list(str)): List of tokens.
            emotion (Optional[str]): Predict with target emotion. Defaults to None.

        Returns:
            tgt_seq (list(str)): Predicted list of tokens.

        """
        src_id_seq = Variable(torch.LongTensor([self.vocabulary.word2index[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        cuda = torch.cuda.is_available()
        if cuda:
            src_id_seq = src_id_seq.cuda()

        if emotion is None:
            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        else:
            emotion_id = Variable(torch.LongTensor([self.emotion_vocabulary.word2index[emotion]]),
                                  volatile=True).view(1, -1)

            if cuda:
                emotion_id = emotion_id.cuda()

            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)], target_emotion=emotion_id)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.vocabulary.index2word[tok] for tok in tgt_id_seq]
        return tgt_seq



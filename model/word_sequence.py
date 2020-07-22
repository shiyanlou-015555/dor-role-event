import torch
import torch.nn as nn
import torchcrf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding import Glove_Bert_Embedding
from crf import CRF
from utils.ConsolLog import Print
from utils.Reader import Reader
from utils.Alphabet import Alphabet


class RoleFiller(nn.Module):
    r"""
    Intergrated Model of Document Event Role Filler.
    Model contents:
        1. word embedding, including pre-trained Glove and Bert.
        2. 2 LSTMs for both sentence level and paragraph level.
        3. Concat.
        4. CRF calculates score.
    Args:
        reader: Reader instance from Main script.
    """

    def __init__(self, reader):
        super(RoleFiller, self).__init__()
        reader = Reader('')
        self.embedding = Glove_Bert_Embedding(
            reader.word_dict.word_size,
            reader.config.parser['word_embed_dim'],
            reader.config.parser['HP_dropout'],
            reader.build_pre_embedding(),
            reader.word_dict.idx2word
        )
        self.drop_lstm_sent = reader.config.parser['HP_dropout'] - 0.1
        self.drop_lstm_para = reader.config.parser['HP_dropout']

        self.embedding_dim = reader.config.parser['word_embed_dim'] + 768
        # 768 is set to be the statical bert dimension

        # LSTM
        self.hidden_dim = reader.config.parser['HP_hidden_dim']
        if reader.config.parser['HP_bilstm']:
            self.hidden_dim //= 2
        # LSTM for paragraph level
        self.lstm_para = nn.LSTM(
            input_size=self.embedding_dim,
            self.hidden_dim,
            reader.config.parser['HP_lstm_layers_num'],
            batch_first=True,
            bidirectional=reader.config.parser['HP_bilstm']
        )
        # LSTM for sentence level
        self.lstm_sent = nn.LSTM(
            input_size=self.embedding_dim,
            self.hidden_dim,
            reader.config.parser['HP_lstm_layers_num'],
            batch_first=True,
            bidirectional=reader.config.parser['HP_bilstm']
        )

        # gate-sigmoid sum
        self.gate = nn.Linear(
            2 * reader.config.parser['HP_hidden_dim'],
            reader.config.parser['HP_hidden_dim']
        )
        self.sigmoid = nn.Sigmoid()

        self.hidden2tag = nn.Linear(
            reader.config.parser['HP_hidden_dim'],
            reader.tag_dict.word_size
        )
        self.softmax = nn.Softmax(dim=-1)

        self.crf = CRF(reader.tag_dict.word_size)

    def sentence_level_process(self, sentence):
        r"""
        LSTM Processing for sentence level.
        Args:
            sentence: Tensor of one sentence.

        Return:
            Tensor shape of [1, sentence len, hidden dim]
        """
        sentence_tensor = self.embedding(sentence.view(1, -1))
        # sentence: [1, sentence len, embedding dim]
        sentence = pack_padded_sequence(sentence_tensor,
                                        lengths=[sentence_tensor.shape[1]], batch_first=True)
        hidden = None
        sentence_tensor, hidden = self.lstm_sent(sentence_tensor, hidden)
        sentence_tensor, _ = pad_packed_sequence(
            sentence_tensor, batch_first=True)
        # sentence_tensor: [1, sentence len, hidden dim]
        return self.drop_lstm_sent(sentence_tensor)

    def forward(self, inputs_sent, inputs_para, lengths):
        r"""
        Args:
            inputs_sent: [batch size, k(which equals 3), sentence len]
            inputs_para: [batch size, k X max_len]
        """
        # process paragraph level embedding and lstm
        para_tensor = self.embedding(inputs_para)
        # para_tensor: [batch size, sentence len, embedding dim]
        para_tensor = pack_padded_sequence(para_tensor, lengths=lengths,
                                           batch_first=True)
        hidden = None
        para_tensor, hidden = self.lstm_para(para_tensor, hidden)
        para_tensor, _ = pad_packed_sequence(para_tensor, batch_first=True)
        # para_tensor: [batch_size, sentence len, hidden dim]

        para_tensor = self.drop_lstm_para(para_tensor)
        # para_tensor: [batch size, sentence len, hidden dim]

        max_len = para_tensor.shape[1]
        sent_tensor = torch.zeros(para_tensor.shape)
        for idx, sentences in enumerate(inputs_sent):
            sample_sent = [
                self.sentence_level_process(sentence).squeeze(0)
                for sentence in sentences
            ]
            sent_tensor[idx][: lengths[idx]][:] = torch.cat(sample_sent, dim=0)
        # sent_tensor: [batch size, max len, hidden dim]

        gamma = self.sigmoid(self.gate(torch.cat([sent_tensor, para_tensor], dim=2)))
        # gamma: [batch size, max len, hidden dim]
        gamma = self.hidden2tag(gamma * sent_tensor + (1 - gamma) * para_tensor)
        # gamma: [batch size, max len, tag num]

        

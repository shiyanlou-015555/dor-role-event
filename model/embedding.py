from utils.ConsolLog import Print

import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Glove_Bert_Embedding(nn.Module):
    """
    Word embedding concatened Glove pre-trained embedding and contextualized Bert Model.

    Args:
        embed_dim: Glove Embedding dimension. (Bert dimension is statically 768)
        pre_build_embedding: Glove embedding. `np.array` is acceptable.
        idx2word: word dictionary in `Reader`.
    """

    def __init__(self, input_size, embed_dim, dropout_prob, pre_build_embedding, idx2word):
        super(Glove_Bert_Embedding, self).__init__()
        self.input_size, self.embed_dim =\
            input_size, embed_dim
        self.embedding = nn.Embedding(input_size, self.embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pre_build_embedding))
        self.dropout = nn.Dropout(dropout_prob)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.idx2word = idx2word
        Print('Word embedding Model built done.', 'success')

    def bert_react(self, inputs):
        """
        React inputs data to bert tensor.

        Args:
            inputs: [batch size, sentence len].
        """
        out = []
        for sentence in inputs:
            tmp = []
            for idx in sentence:
                if idx in self.idx2word:
                    tmp.append(self.idx2word[idx])
                else:
                    tmp.append(self.idx2word[0])
                # use bert to get ids of tokens
            out.append(self.tokenizer.convert_ids_to_tokens(tmp))
        out = torch.tensor(out)
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor_batch)
        # encoded layers is a list of whole bert layer outputs
        # we need the average layer outputs
        return sum(encoded_layers) / len(encoded_layers)

    def forward(self, inputs):
        """
        forward
        """
        embed = self.embedding(inputs)
        # embed: [batch size, sentences length, embedding dim]
        embed = torch.cat([embed, self.bert_react(inputs)], dim=2)
        # embed: [batch size, sentences length, embedding dim + 768]
        embed = self.dropout(embed)
        return embed

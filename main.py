import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import warnings

# customize
from utils.ConsolLog import Print
from utils.pipeline import *
from utils.Alphabet import Alphabet
from utils.Reader import Reader

from model.word_sequence import RoleFiller
from model.crf import CRF
from model.embedding import Glove_Bert_Embedding


def train(reader, model):
    print(reader.config)
    optim_choice = reader.config.parser['optimizer']

    # switch optimizer
    if optim_choice.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            reader.config.parser['HP_learning_rate'],
            momentum=reader.config.parser['momentum'],
            weight_decay=reader.config.parser['HP_l2']
        )
    elif optim_choice.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            reader.config.parser['HP_learning_rate'],
            weight_decay=reader.config.parser['HP_l2']
        )
    else:   # default adam
        Print('optimizer is not found, using Adam by default', 'warning')
        optimizer = optim.Adam(
            model.parameters(),
            reader.config.parser['HP_learning_rate'],
            weight_decay=reader.config.parser['HP_l2']
        )

    best_dev = -10

    train_doc_idx, train_sentence_list_idx, train_tag_idx =\
        reader.read_from_file('train')

    Print('Start training model...', 'information')
    for epoch in range(reader.config.parser['iteration']):
        Print(
            f'Epoch: {epoch + 1} / {reader.config.parser["iteration"]}',
            'information'
        )

        # update learning rate if optimizer is 'SGD'
        if optim_choice is 'SGD':
            optimizer = lr_decay(
                optimizer,
                epoch,
                reader.config.parser['HP_lr_decay'],
                reader.config.parser['HP_learning_rate']
            )
        # random shuffle train data
        train_ids = np.arange(len(train_doc_idx))
        np.random.shuffle(train_ids)

        model.train()
        batch_size = reader.config.parser['batch_size']
        batch_nums = len(train_doc_idx) // batch_size
        tqdm_iter = tqdm.tqdm(range(batch_nums))
        tqdm_iter.set_description('training on batch', refresh=False)
        loss_sum = 0
        for batch_id in tqdm_iter:
            model.zero_grad()
            start = batch_id * batch_size
            end = start + batch_size
            
            # the following three are train inputs
            train_doc = [train_doc_idx[idx] for idx in train_ids[start: end]]
            train_sentence_list = [train_sentence_list_idx[idx] for idx in train_ids[start: end]]
            train_tag = [train_tag_idx[idx] for idx in train_ids[start: end]]
            
            word_seq_tensor, word_seq_len, seq_order_recovery, sentence_tensor_list, tag_seq_tensor,\
                seq_mask = batchify_generation(train_doc, train_sentence_list, train_tag, True)
            loss, tag_seq = model.forward(sentence_tensor_list, word_seq_tensor, word_seq_len, tag_seq_tensor, seq_mask)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        Print(f'average loss {loss_sum / batch_nums: .4f}')


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # if this necessary
    os.chdir('my_rep')

    reader = Reader('conf/default.json')
    for file_type in ['train', 'test', 'dev']:
        reader.build_word_dict(file_type)

    # training model
    model = RoleFiller(reader)
    train(reader, model)

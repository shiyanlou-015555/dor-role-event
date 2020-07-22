import numpy as np
import torch


def batchify_generation(batch_doc_seq, batch_sentence_list, batch_tag_seq, if_train):
    """
    For the downstream model, LSTM layer requires same-length batch data,
    which is not coordinate with current data.
    Pad `batch_doc_seq` to the max batch length.
    Pad `batch_sentence_list` to the max sentence length. In the basic version, no padding.
    Pad `batch_tag_seq` to the max batch length.

    Args:
        Refer to `Reader.read_from_file`.
        if_train: Whether in training state. 

    Return:
        Tuple of:
            packed padded doc sequence,
            length for each group,
            order list to recover to original order(useful only on prediction),
            list of (list of sentences from one group),
            packed padded tag sequence,
            mask for each group.
    """

    batch_size = len(batch_doc_seq)

    # process `batch_doc_seq` and `batch_tag_seq`
    # pad to same length
    word_seq_len = torch.LongTensor(list(map(len, batch_doc_seq)))
    max_seq_len = word_seq_len.max().item()
    word_seq_tensor = torch.zeros(
        (batch_size, max_seq_len), requires_grad=if_train).long()
    tag_seq_tensor = torch.zeros(
        (batch_size, max_seq_len), requires_grad=if_train).long()

    seq_mask = torch.zeros((batch_size, max_seq_len),
                           requires_grad=if_train).byte()
    for idx, (doc, tag, seq_len) in enumerate(zip(batch_doc_seq, batch_tag_seq, word_seq_len)):
        this_seq_len = seq_len.item()
        word_seq_tensor[idx, :this_seq_len] = torch.LongTensor(doc)
        tag_seq_tensor[idx, :this_seq_len] = torch.LongTensor(tag)
        seq_mask[idx, :this_seq_len] = torch.Tensor([1] * this_seq_len)

    # get pack order for `word_seq_tensor`, which is padded `batch_doc_seq`
    word_seq_len, seq_order_by_length = word_seq_len.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[seq_order_by_length]
    tag_seq_tensor = tag_seq_tensor[seq_order_by_length]
    seq_mask = seq_mask[seq_order_by_length]

    # process `batch_sentence_list`

    # basic version: no padding, each sentence is a dependent batch
    sentence_tensor_list = []
    for sentences in batch_sentence_list:
        k_group_list = [
            torch.LongTensor(sentence, requires_grad=if_train) for sentence in sentences
        ]
        sentence_tensor_list.append(k_group_list)
    sentence_tensor_list = [sentence_tensor_list[idx]
                            for idx in seq_order_by_length]

    # recover order
    _, seq_order_recovery = seq_order_by_length.sort(0, descending=False)

    return word_seq_tensor, word_seq_len, seq_order_recovery,\
        sentence_tensor_list, tag_seq_tensor, seq_mask


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

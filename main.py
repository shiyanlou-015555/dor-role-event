import os
import numpy as np
import torch

# customize
from utils.ConsolLog import Print
from utils.pipeline import *
from utils.Alphabet import Alphabet
from utils.Reader import Reader

from model.word_sequence import RoleFiller
from model.crf import CRF
from model.embedding import Glove_Bert_Embedding

def train(reader, model):
    Print('Training model...', 'information')
    print(reader.config)


if __name__ == "__main__":
    # if this necessary
    os.chdir('my_rep')

    reader = Reader('conf/default.json')
    # training model
    model = RoleFiller(reader)
    train(reader)
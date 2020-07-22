from ConsolLog import Print
import json


class Alphabet:
    """
    An alphabet api to word dictionary.

    Args:
        name: A name for the alphabet. Necessary.
        with_unknown_label: An unknown label for words not in dictionary.
            Not necessary when it's sure that no unknown words during usage.
            Make it None then.
    """

    def __init__(self, name, unknown_label='</unk>'):
        self.word2idx = {}
        self.idx2word = {}
        self.name = name
        self.unknown = unknown_label
        self.word_size = 0
        if self.unknown:
            self.add_word(self.unknown)

    def add_word(self, word, ignore_existed_word=True):
        """
        Add a word to the dictionary.
        """
        if word in self.word2idx:
            if not ignore_existed_word:
                Print('word ' + word + ' already exists in alphabet ' +
                      self.name, 'warning')
        else:
            self.word2idx[word] = self.word_size
            self.idx2word[self.word_size] = word
            self.word_size += 1

    def get_word_index(self, word):
        """
        Simple `word2idx[word]`.
        """
        if word not in self.word2idx:
            Print('No word ' + word + ' in alphabet ' + self.name, 'error')
        else:
            return self.word2idx[word]

    def get_index_word(self, idx):
        """
        Simple `idx2word[idx]`.
        """
        if idx >= self.word_size:
            Print('No idx ' + idx + ' in alphabet ' + self.name, 'error')
        else:
            return self.idx2word[idx]

    def save_to_json(self, alphabet_file):
        """
        Save alphabet with `self.name` wo alphabet_file.

        Args:
            alphabet_file: No need to add .json
        """
        with open(alphabet_file + '_' + self.name + '.json', 'r') as f:
            json.dump(vars(self.word2idx), f)

import numpy as np
import tqdm
from utils.Alphabet import Alphabet
from utils.ConsolLog import Print
from utils.Config import Configuration


class Reader:
    """
    K-sentences reader.

    Args:
        config_file: `Configuration` class initializing file for training settings.
    """

    def __init__(self, config_file):
        # alphabet for words and roles
        self.word_dict = Alphabet(name='word_dictionary')
        self.tag_dict = Alphabet(name='tag_dictionary', unknown_label='O')
        # base configuration
        self.config = Configuration(parser=None, config_file=config_file)

        # word embedding
        self.embedding = None

        # build role dictionary
        key_roles = ['perp_individual_id', 'perp_organization_id', 'phys_tgt_id',
                     'hum_tgt_name', 'incident_instrument_id']
        for role in key_roles:
            for position in 'BI':
                # whole role is like 'B-I-hum_tgt_name'
                whole_role = position + '-' + role
                self.tag_dict.add_word(whole_role)

    def build_word_dict(self, file_type):
        """
        Build word dictionary from file_type.

        Args:
            file_type: Valid in range ['train', 'test', 'dev'].
        """
        file_name = self.config.parser[file_type + '_dir']
        with open(file_name, 'r') as f:
            for line in f.readlines():
                contents = line.strip().split(' ')
                # an empty line, means seperator for two batch
                # doc id, means a new batch whose `docid` is doc id
                # a word and its tag sepaerated by a blank
                if len(contents) >= 2:
                    word, _ = contents[0], contents[1]
                    self.word_dict.add_word(word)

    def read_from_file(self, file_type):
        """
        Read processed data from file.
        Generate following message from target file:
            1. Indexes of texts with shape of [batch size, sequence len].
            Name as `doc_idx`.
            2. List of texts from one `k group` with shape of [batch size, k, sentences len].
            Name as `sentence_list_idx`.
            3. Indexes of labels with shape of [batch size, sequence len].
            Name as `tag_idx`.

        Args:
            file_type: Valid in range ['train', 'test', 'dev'].

        Return:
            (doc_idx, sentence_list_idx, tag_idx)
        """

        doc_idx, sentence_list_idx, tag_idx = [], [], []

        file_name = self.config.parser[file_type + '_dir']
        with open(file_name, 'r') as f:

            new_batch_doc_idx, new_batch_sentence_list_idx,\
                new_batch_tag_idx = [], [], []
            new_sentence_idx = []   # for a sentence
            # temprate variable to store current batch data

            for idx, line in enumerate(f.readlines()):
                if idx == 95:
                    xu = 1
                contents = line.strip().split(' ')
                # an empty line, means seperator for two batch
                # doc id, means a new batch whose `docid` is doc id
                # a word and its tag sepaerated by a blank
                if len(contents) >= 2:
                    word, role = contents[0], contents[1]
                    new_batch_doc_idx.append(
                        self.word_dict.get_word_index(word)
                    )
                    new_batch_tag_idx.append(
                        self.tag_dict.get_word_index(role)
                    )
                    new_sentence_idx.append(
                        self.word_dict.get_word_index(word)
                    )
                    if word is '.':
                        # default: '.' is the seperator for two sentences.
                        new_batch_sentence_list_idx.append(new_sentence_idx)
                        new_sentence_idx = []
                elif len(contents) == 1 and contents[0] != '':

                    new_batch_doc_idx, new_batch_sentence_list_idx,\
                        new_batch_tag_idx = [], [], []
                    new_sentence_idx = []   # for a sentence
                    # temprate variable to store current batch data

                elif len(contents) == 1 and contents[0] == ''\
                    and len(new_batch_doc_idx) < self.config.parser['HP_max_len']:
                    doc_idx.append(new_batch_doc_idx)
                    sentence_list_idx.append(new_batch_sentence_list_idx)
                    tag_idx.append(new_batch_tag_idx)

        return doc_idx, sentence_list_idx, tag_idx

    def build_pre_embedding(self):
        """
        Build word embedding from pre-trained Glove model by default.
        For the word not in pre-trained Glove model,
        we apply random vector to represent the embedding.

        This should be after building word dictionary.
        """
        if self.config.parser['embed_dir'] is None:
            Print('Pre-trained embedding file not available.', 'error')
            return

        embed_file = self.config.parser['embed_dir']

        # load in pre-trained Glove model, save it as a dict
        pretrain_embed = {}
        with open(embed_file, 'r', encoding='utf-8') as f:
            tqdm_iter = tqdm.tqdm(f.readlines())
            tqdm_iter.set_description('read from pre-trained file', False)
            for line in tqdm_iter:
                embed_content = line.strip().split()
                word, embed_content = embed_content[0], embed_content[1:]
                if self.config.parser['word_embed_dim'] < 0:
                    self.config.parser['word_embed_dim'] = len(embed_content)
                elif self.config.parser['word_embed_dim'] != len(embed_content):
                    # invalid embedding word
                    continue
                embed_content = np.array([float(x) for x in embed_content])
                pretrain_embed[word] = embed_content

        embed_dim = self.config.parser['word_embed_dim']

        # build embedding if find it in pre-trained model
        # else randomly generate one.
        self.embedding = np.empty([
            self.word_dict.word_size, embed_dim
        ])
        scale = np.sqrt(3 / embed_dim)
        perfect_match, case_match, not_match = 0, 0, 0
        for word, index in self.word_dict.word2idx.items():
            if word in pretrain_embed:
                self.embedding[index, :] = self.norm2one(pretrain_embed[word]) \
                    if self.config.parser['norm_word_embed'] else pretrain_embed[word]
                perfect_match += 1
            if word.lower() in pretrain_embed:
                self.embedding[index, :] = self.norm2one(pretrain_embed[word.lower()]) \
                    if self.config.parser['norm_word_embed'] else pretrain_embed[word.lower()]
                case_match += 1
            else:
                # not found
                self.embedding[index,
                               :] = np.random.uniform(-scale, scale, [embed_dim])
                not_match += 1
        Print(
            f'Pre-trained embedding loaded in from {embed_file}, '\
            'pre-train words: {len(pretrain_embed)}, perfect match {perfect_match}, '\
            'case match {case_match}, not match {not_match}, '\
            'oov {not_match / self.word_dict.word_size}', 'success'
        )
        return self.embedding

    def norm2one(self, vec):
        """
        Vector normalization.
        """
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec / root_sum_square


if __name__ == "__main__":
    import os
    os.chdir('my_rep')
    reader = Reader('conf/default.json')
    for file_type in ['train', 'test', 'dev']:
        reader.build_word_dict(file_type)
    # reader.build_pre_embedding()
    reader.read_from_file('train')
    print('terminate')

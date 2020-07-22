import argparse
import json
from ConsolLog import Print


class Configuration:
    """
    Trying to build a high API for settings of training procedure.
    """

    def __init__(self, parser, config_file):
        self.parser = None
        if parser:
            self.parser = parser
            if config_file:
                self.save_to_json_file(config_file)
        else:
            Print('A configuration file is required', 'information')
            if config_file:
                Print(config_file + ' loading', 'information')
                self.load_from_json_file(config_file)
            else:
                Print('But no files founded!', 'warning')

    def load_from_json_file(self, config_file):
        r"""
        load configuration from json.
        """
        if config_file is None:
            Print('No config file founded.', 'error')
            return
        with open(config_file, 'r') as f:
            self.parser = json.load(f)

    def save_to_json_file(self, config_file):
        r"""
        Save configuration as json.
        """
        if config_file is None:
            Print('No config file founded.', 'error')
            return
        with open(config_file, 'w') as f:
            json.dump(vars(self.parser.parse_args()), f)

    def __repr__(self):
        Print('Configuration settings:', 'information')
        Print('--------------------------------', 'information')
        ret = ''
        for key in self.parser:
            ret += key + '\t\t' + str(self.parser[key]) + '\n'
        return ret


# run `generate_conf_json` to generate configuration file for convienet usage

def generate_conf_json():
    parser = argparse.ArgumentParser(
        description='Configuration for training settings')
    parser.add_argument('--config', default='conf/default.json')
    parser.add_argument('--train_dir', default='data/formed/train_full')
    parser.add_argument('--dev_dir', default='data/formed/dev_full')
    parser.add_argument('--test_dir', default='data/formed/test')
    parser.add_argument('--bert_dir', default='sources/bert_model.pkl')
    parser.add_argument('--embed_dir', default='sources/glove.6B.100d.txt')
    parser.add_argument('--model_dir', default='sources/model.pkl')
    parser.add_argument('--norm_word_embed', default=False)
    parser.add_argument('--norm_char_embed', default=False)
    parser.add_argument('--word_embed_dim', default=100)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--iteration', default=15)
    parser.add_argument('--batch_size', default=5)
    parser.add_argument('--HP_hidden_dim', default=200)
    parser.add_argument('--HP_dropout', default=0.4)
    parser.add_argument('--HP_lstm_layers_num', default=1)
    parser.add_argument('--HP_learning_rate', default=0.015)
    parser.add_argument('--HP_lr_decay', default=0.05)
    parser.add_argument('--momentum', default=0)
    parser.add_argument('--HP_l2', default=1e-8)
    parser.add_argument('--HP_max_len', default=200)

    my_conf = Configuration(parser, 'conf/default.json')


if __name__ == '__main__':
    # generate_conf_json()

    config = Configuration(parser=None, config_file='conf/default.json')
    print(config)

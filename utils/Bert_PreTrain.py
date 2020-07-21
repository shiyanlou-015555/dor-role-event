import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from ConsolLog import Print

def BertPreTrain(bert_file_name='sources/bert_model.pkl'):
    r"""
    It takes a lot time to train Bert embedding if needed.
    
    Run this function before training, then just use the model directly.

    Bert embedding dimmension is 768.

    Args:
        bert_file_name: file to store the pre-trained bert model.
    """
    Print('Training Bert...', 'information')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    Print('Training Done...', 'success')
    torch.save(bert_model, bert_file_name)
    Print('Model Saved at ' + bert_file_name, 'success')

if __name__ == '__main__':
    BertPreTrain()
    bert_model = torch.load('sources/bert_model.pkl')
    Print('Bert Model load done', 'success')

import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.nn import utils, init
import torch.nn.functional as F
from torch.utils import data
from gensim.models import KeyedVectors
import sys
sys.path.append('.')
import metrics
from HW2.src import CNN_BLSTM


class EssayDataset(data.Dataset):
    def __init__(self, essays, labels, ids):
        self.essays = essays
        self.labels = labels
        self.ids = ids
    def __getitem__(self, index):
        essay = self.essays[index]
        label = self.labels[index]
        essay_id = self.ids[index]
        return essay, label, essay_id
    def __len__(self):
        return len(self.labels)

def loadNet(embedding, embed_size, set_id, device):
    inception_size = 100
    hidden_size = 300
    batch_size = 64
    topk = 3
    min_label, max_label = 0, 0
    if set_id == 1:
        min_label = 2
        max_label = 12
    if set_id == 2:
        min_label = 1
        max_label = 6
    if set_id == 3 or set_id == 4:
        max_label = 3
    if set_id == 5 or set_id == 6:
        max_label = 4
    if set_id == 7:
        max_label = 30
    if set_id == 8:
        max_label = 60
    net = CNN_BLSTM.EssayNet(embedding, embed_size, inception_size, hidden_size, topk, min_label, max_label).to(device)
    path = 'HW2/models/model_' + str(set_id) + '.pt'
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: len(data[0]), reverse=True)
    sentences = []
    lengths = []
    labels = []
    ids = []

    for data in batch_data:
        essay = data[0]
        label = data[1]
        essay_id = data[2]
        length = len(essay)

        essay = [list(map(int, [word for word in sen.strip().split(' ') if word.isdigit()])) for sen in essay]
        sentences.extend(essay)
        labels.append(label)
        lengths.append(length)
        ids.append(essay_id)

    sentences = [torch.LongTensor(sentence) for sentence in sentences]
    sentences = utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    lengths = torch.IntTensor(lengths)
    labels = np.array(labels)
    ids = np.array(ids)
    return sentences, lengths, labels, ids

def loadData(essays, set_id, batch_size):
    test_data = essays[(essays['class'] == 'test') & (essays['set'] == set_id)]
    test_essays = [essay.split(' ## ') for essay in test_data['sens_index'].values.tolist()]
    test_labels = test_data['score'].values
    test_ids = test_data['id'].values
    test_dataset = EssayDataset(test_essays, test_labels, test_ids)
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    return test_loader

def predict(set_id, net, test_loader, device):
    preds = []
    essays_ids = []
    test_kappa = 0.0

    net.eval()
    with torch.no_grad():    
        m = 0
        for essays, lengths, labels, ids in test_loader:
            essays = essays.to(device)
            lengths = lengths.to(device)
            pred = net(essays, lengths)
            pred = pred.cpu().detach().numpy()
            test_kappa += metrics.kappa(pred, labels, 'quadratic') * len(labels)
            preds.extend(pred.round().tolist())
            essays_ids.extend(ids.tolist())
            m += len(labels)
    test_kappa /= m
    return preds, essays_ids, test_kappa



if __name__ == "__main__":
    device = torch.device('cuda', 0)
    glove = KeyedVectors.load_word2vec_format('/home/lj/dataset/glove/w2v.6B.300d.txt')
    essays = pd.read_csv('HW2/data/essays.csv')

    embed_size = 300
    corpus = essays['words'].values
    vocab = CNN_BLSTM.createVocab(corpus, glove)
    embedding = CNN_BLSTM.getEmbedding(glove, vocab, embed_size)

    preds = []
    set_ids = []
    kappas = []

    for i in range(1, 9):
        net = loadNet(embedding, embed_size, i, device)
        test_data = loadData(essays, i, 64)
        pred, ids, kappa = predict(i, net, test_data, device)
        preds.append(pred)
        set_ids.append(ids)
        kappas.append(kappa)

    print(kappas)

    result = []
    for i in range(8):
        for j in range(len(set_ids[i])):
            result.append([int(set_ids[i][j]), i+1, int(preds[i][j])])

    result = pd.DataFrame(result)
    result.to_csv('HW2/data/MF1933059.tsv', sep='\t', index=False, header=False)
















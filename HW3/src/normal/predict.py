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
from HW3.src.normal import CNN_BLSTM


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
    net = CNN_BLSTM.EssayNet(embedding, embed_size, inception_size, hidden_size, topk).to(device)
    path = 'HW3/models_normal/model_' + str(set_id) + '.pt'
    net.load_state_dict(torch.load(path))
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
    test_essays = [essay.split(' ## ') for essay in test_data['index'].values.tolist()]
    test_labels = test_data['score'].values
    test_ids = test_data['id'].values
    test_dataset = EssayDataset(test_essays, test_labels, test_ids)
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    min_score, max_score = 0, 0
    if set_id == 1:
        min_score = 2
        max_score = 12
    if set_id == 2:
        min_score = 1
        max_score = 6
    if set_id == 3 or set_id == 4:
        max_score = 3
    if set_id == 5 or set_id == 6:
        max_score = 4
    if set_id == 7:
        max_score = 30
    if set_id == 8:
        max_score = 60
    return test_loader, min_score, max_score

def predict(set_id, min_score, max_score, net, test_loader, device):
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
            pred = pred * (max_score - min_score) + min_score
            labels = labels * (max_score - min_score) + min_score

            pred = pred.cpu().detach().numpy()
            test_kappa += metrics.kappa(pred, labels, 'quadratic') * len(labels)

            preds.extend(pred.round().tolist())
            essays_ids.extend(ids.tolist())
            m += len(labels)
    test_kappa /= m
    return preds, essays_ids, test_kappa


def main():
    device = torch.device('cuda', 1)
    glove = KeyedVectors.load_word2vec_format('/home/lj/dataset/glove/w2v.6B.300d.txt')
    essays = pd.read_csv('HW3/data/essays.csv')

    embed_size = 300
    corpus = essays['sentences'].values
    vocab = CNN_BLSTM.createVocab(corpus, glove)
    embedding = CNN_BLSTM.getEmbedding(glove, vocab, embed_size)

    preds = []
    set_ids = []
    kappas = []

    for i in range(1, 9):
        net = loadNet(embedding, embed_size, i, device)
        test_data, min_score, max_score = loadData(essays, i, 64)
        pred, ids, kappa = predict(i, min_score, max_score, net, test_data, device)
        preds.append(pred)
        set_ids.append(ids)
        kappas.append(kappa)

    print(kappas)

    result = []
    for i in range(8):
        for j in range(len(set_ids[i])):
            result.append([int(set_ids[i][j]), i+1, int(preds[i][j])])

    result = pd.DataFrame(result)
    result.to_csv('HW3/data/normal.tsv', sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
    
















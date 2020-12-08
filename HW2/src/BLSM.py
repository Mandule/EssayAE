import time
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.nn import utils, init
from torch.utils import data
from gensim.models import KeyedVectors

import sys
sys.path.append('.')
import metrics


device = torch.device('cuda', 0)
loss_func = nn.MSELoss()
num_epochs = 100
batch_size = 256
embed_size = 300
hidden_size = 300
num_layers = 2
topk = 3
bi = True
num_label = 4
lr = 1e-3


class EssayDataset(data.Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]
        return seq, label
    def __len__(self):
        return len(self.seqs)

class EssayNet(nn.Module):
    def __init__(self, embedding, vocab_size, embed_size, hidden_size, num_layers, topk, num_label):
        super(EssayNet, self).__init__()
        self.num_label = num_label
        self.topk = topk
        self.num_layers = num_layers
        
        self.embedLayer = nn.Embedding.from_pretrained(embedding)
        self.LSTMLayer = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size * 2 * topk, hidden_size * topk),
            # nn.BatchNorm1d(hidden_size * topk),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size * topk, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

        self.init_LSTM()

    def forward(self, seqs, lens):
        # x.shpae = (batch_size, seq_len)
        x = self.embedLayer(seqs)
        # x.shape = (batch_size, seq_len, embed_size)
        x = torch.transpose(x, 0, 1)
        # x.shape = (seq_len, batch_size, embed_size)
        x = utils.rnn.pack_padded_sequence(x, lens)
        # x -> PackedSequence
        y, (h, c) = self.LSTMLayer(x)
        # y -> PackedSequence
        y, lens = utils.rnn.pad_packed_sequence(y)
        # y.shape = (seq_len, batch, 2 * hidden_size)
        y = torch.transpose(y, 0, 1)
        y = torch.transpose(y, 1, 2)
        # y.shape = (batch, 2 * hidden_sizen, seq_len)
        y = self.kmax_pooling(y)
        # y.shape = (batch, 2 * hidden_sizen, topk)
        y = y.view(y.shape[0], -1)
        # y.shape = (batch, 2 * hidden_size * topk)
        y = self.fc(y)
        y = y.squeeze(1) * (self.num_label-1)
        return y

    def kmax_pooling(self, x):
        index = x.topk(self.topk, dim = 2)[1].sort(dim = 2)[0]
        return x.gather(2, index)

    def init_LSTM(self):
        nn.init.orthogonal_(self.LSTMLayer.weight_hh_l0)
        nn.init.orthogonal_(self.LSTMLayer.weight_ih_l0)
        nn.init.orthogonal_(self.LSTMLayer.weight_hh_l1)
        nn.init.orthogonal_(self.LSTMLayer.weight_ih_l1)

def createVocab(glove, corpus):
    vocab = {'<pad>': 0, '<unk>': 1}
    index = 2
    for essay in corpus:
        words = essay.split(' ')
        for word in words:
            if glove.vocab.get(word):
                if vocab.get(word) is None:
                    vocab[word] = index
                    index += 1
    return vocab, index

def getEmbedding(glove, vocab):
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for word, index in vocab.items():
        if glove.vocab.get(word):
            embedding[index] = torch.from_numpy(glove.get_vector(word))
    embedding[1] = torch.randn(embed_size)
    return embedding

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: len(data[0]), reverse=True)
    seqs = [torch.LongTensor(data[0]) for data in batch_data]
    labels =  [data[1] for data in batch_data]
    seqs_len = [len(seq) for seq in seqs]
    
    seqs = utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.FloatTensor(labels)
    seqs_len = torch.IntTensor(seqs_len)
    return seqs, seqs_len, labels

def getDataLoader(essay, set_id):
    set_data = essay[essay['set'] == set_id]
    train_data = set_data[set_data['class'] == 'train']
    dev_data = set_data[set_data['class'] == 'dev']
    train_seq = [list(map(int, seq.split(' '))) for seq in train_data['words_index'].values.tolist()]
    train_label = train_data['score'].values
    dev_seq = [list(map(int, seq.split(' '))) for seq in dev_data['words_index'].values.tolist()]
    dev_label = dev_data['score'].values

    train_dataset = EssayDataset(train_seq, train_label)
    dev_dataset = EssayDataset(dev_seq, dev_label)

    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = data.DataLoader(dev_dataset, batch_size, collate_fn=collate_fn)
    
    return train_loader, dev_loader

def train(net, loss_func, optimizer, train_loader, dev_loader):
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, dev_loss = 0.0, 0.0
        train_kappa, dev_kappa = 0.0, 0.0
        n, m = 0, 0
        for seqs, seqs_len, labels in train_loader:
            seqs = seqs.to(device)
            seqs_len = seqs_len.to(device)
            labels = labels.to(device)
            y = net(seqs, seqs_len)
            loss = loss_func(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seqs.shape[0]
            train_kappa += metrics.kappa(y.cpu().detach().numpy(), labels.cpu().detach().numpy(), 'quadratic') * seqs.shape[0]
            n += seqs.shape[0]
        train_loss /= n
        train_kappa /= n
        
        net.eval()
        with torch.no_grad():
            for seqs, seqs_len, labels in dev_loader:
                seqs = seqs.to(device)
                seqs_len = seqs_len.to(device)
                labels = labels.to(device)
                y = net(seqs, seqs_len)
                loss = loss_func(y, labels)
                
                dev_loss += loss.item() * labels.shape[0]
                dev_kappa += metrics.kappa(y.cpu().detach().numpy(), labels.cpu().detach().numpy(), 'quadratic') * seqs.shape[0]
                m += labels.shape[0]
        net.train()
        dev_loss /= m
        dev_kappa /= m
        
        end = time.time()
        runtime = end - start
        print('epoch %d, train loss: %.4f, train kappa: %.4f, dev loss: %.4f, dev kappa: %.4f, time: %.2f'
            % (epoch, train_loss, train_kappa, dev_loss, dev_kappa, runtime))

def main():
    glove = KeyedVectors.load_word2vec_format('/home/lj/dataset/glove/w2v.6B.300d.txt')
    essay = pd.read_csv('HW2/data/essay.csv')
    vocab, vocab_size = createVocab(glove, essay['words'].values)
    embedding = getEmbedding(glove, vocab)

    train_loader, dev_loader = getDataLoader(essay, 3)
    net = EssayNet(embedding, vocab_size, embed_size, hidden_size, num_layers, topk, num_label).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train(net, loss_func, optimizer, train_loader, dev_loader)

if __name__ == "__main__":
    main()


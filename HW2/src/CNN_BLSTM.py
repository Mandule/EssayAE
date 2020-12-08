import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.nn import utils, init
import torch.nn.functional as F
from torch.utils import data
import sys
sys.path.append('.')
import metrics
from gensim.models import KeyedVectors


max_test_kappa = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

class EssayDataset(data.Dataset):
    def __init__(self, essays, labels):
        self.essays = essays
        self.labels = labels
    def __getitem__(self, index):
        essay = self.essays[index]
        label = self.labels[index]
        return essay, label
    def __len__(self):
        return len(self.labels)

class Inception(nn.Module):
    def __init__(self, cin, co, relu=True):
        super(Inception, self).__init__()
        assert(co%4 == 0)
        cos = co // 4
        
        self.branch1 = nn.Conv1d(cin, cos, 1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(cin, cos, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(cos, cos, 3, padding=1))
        self.branch3 = nn.Sequential(
            nn.Conv1d(cin, cos, 3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(cos, cos, 5,stride=1,padding=2))
        self.branch4 = nn.Conv1d(cin,cos, 3, stride=1, padding=1)
        self.activa= nn.LeakyReLU(inplace=True)
        self.init_cnn()
    
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=self.activa(torch.cat((branch1,branch2,branch3,branch4),1))
        return result
    
    def init_cnn(self):
        init.xavier_normal_(self.branch1.weight)
        for name, param in self.branch2[0].named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0)
        for name, param in self.branch3[0].named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0)
        init.xavier_normal_(self.branch4.weight)

class EssayNet(nn.Module):
    def __init__(self, embedding, embedding_size, inception_size, hidden_size, topk, min_label, max_label):
        super(EssayNet, self).__init__()
        self.topk = topk
        self.min_label = min_label
        self.max_label = max_label
        self.embed = nn.Embedding.from_pretrained(embedding)
        self.conv = nn.Sequential(
            Inception(embedding_size, inception_size),
            Inception(inception_size, inception_size))
        self.rnn = nn.LSTM(input_size=inception_size, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=True, dropout=0.5, num_layers=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2 * topk, hidden_size * topk),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size * topk, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.init_LSTM()

    def forward(self, sens, lengths):
        # sens.shape = (batch_sens_num, seq_ln)
        sens = self.embed(sens)
        # sens.shape = (batch_sens_num, seq_ln, embedding_size)
        sens = self.conv(sens.permute(0, 2, 1))
        # sens.shape = (batch_sens_num, Inception_size, seq_ln)
        sens = F.max_pool1d(sens, sens.shape[-1]).squeeze(-1)
        # sens.shape = (batch_sens_num, Inception_size)
        essays = []
        index = 0
        for l in lengths:
            essays.append(sens[index:index+l])
            index += l
        # essays = [Tensor]
        essays = utils.rnn.pad_sequence(essays, batch_first=True)
        # essays.shape = (batch_size, essay_ln, Inception_size)
        essays = utils.rnn.pack_padded_sequence(essays, lengths, batch_first=True)
        y, (h, c) = self.rnn(essays)
        y, lens = utils.rnn.pad_packed_sequence(y, batch_first=True)
        # y.shape = (batch_size, essay_ln, 2 * hidden_size)
        y = torch.transpose(y, 1, 2)
        # y.shape = (batch_size, 2 * hidden_size, essay_ln)
        y = self.kmax_pooling(y).view(y.shape[0], -1)
        # y.shape = (batch_size, 2 * hidden_size * topk)
        y = self.fc(y).squeeze(-1) * (self.max_label - self.min_label) + self.min_label
        # y.shape = batch_size
        return y
    
    def kmax_pooling(self, x):
        index = x.topk(self.topk, dim = 2)[1].sort(dim = 2)[0]
        return x.gather(2, index)
    
    def init_LSTM(self):
        init.orthogonal_(self.rnn.weight_hh_l0)
        init.orthogonal_(self.rnn.weight_ih_l0)
        init.orthogonal_(self.rnn.weight_hh_l1)
        init.orthogonal_(self.rnn.weight_ih_l1)
        
    def init_fc(self):
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0)

def createVocab(corpus, glove):
    vocab = {'<pad>': 0, '<unk>': 1}
    index = 2
    for essay in corpus:
        words = essay.split(' ')
        for word in words:
            if glove.vocab.get(word):
                if vocab.get(word) is None:
                    vocab[word] = index
                    index += 1
    return vocab

def getEmbedding(glove, vocab, embed_size):
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for word, index in vocab.items():
        if glove.vocab.get(word):
            embedding[index] = torch.from_numpy(glove.get_vector(word))
    embedding[1] = torch.randn(embed_size)
    return embedding

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: len(data[0]), reverse=True)
    sentences = []
    labels = []
    lengths = []
    for data in batch_data:
        essay = data[0]
        label = data[1]
        length = len(essay)
        essay = [list(map(int, [word for word in sen.strip().split(' ') if word.isdigit()])) for sen in essay]
        sentences.extend(essay)
        labels.append(label)
        lengths.append(length)
    
    sentences = [torch.LongTensor(sentence) for sentence in sentences]
    sentences = utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.FloatTensor(labels)
    lengths = torch.IntTensor(lengths)
    return sentences, labels, lengths

def loadData(essays, set_id, batch_size):
    train_data = essays[(essays['class'] != 'test') & (essays['set'] == set_id)]
    test_data = essays[(essays['class'] == 'test') & (essays['set'] == set_id)]
    train_essays = [essay.split(' ## ') for essay in train_data['sens_index'].values.tolist()]
    train_labels = train_data['score'].values

    test_essays = [essay.split(' ## ') for essay in test_data['sens_index'].values.tolist()]
    test_labels = test_data['score'].values
    train_dataset = EssayDataset(train_essays, train_labels)
    test_dataset = EssayDataset(test_essays, test_labels)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
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
    return train_loader, test_loader, min_label, max_label

def train(net, set_id, train_loader, test_loader, loss_func, optimizer, num_epochs, lr, device):
    import time
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, test_loss = 0.0, 0.0
        train_kappa, test_kappa = 0.0, 0.0
        n, m = 0, 0
        for essays, labels, lengths in train_loader:
            essays = essays.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            y = net(essays, lengths)
            loss = loss_func(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.shape[0]
            train_kappa += metrics.kappa(y.cpu().detach().numpy(), labels.cpu().detach().numpy(), 'quadratic') * labels.shape[0]
            n += labels.shape[0]
        train_loss /= n
        train_kappa /= n

        net.eval()
        with torch.no_grad():
            for essays, labels, lengths in test_loader:
                essays = essays.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                y = net(essays, lengths)
                loss = loss_func(y, labels)

                test_loss += loss.item() * labels.shape[0]
                test_kappa += metrics.kappa(y.cpu().detach().numpy(), labels.cpu().detach().numpy(), 'quadratic') * labels.shape[0]
                m += labels.shape[0]
        net.train()
        test_loss /= m
        test_kappa /= m

        if test_kappa > max_test_kappa[set_id]:
            max_test_kappa[set_id] = test_kappa
            torch.save(net.state_dict(), 'HW2/models/model_' + str(set_id) + '.pt')
            print(max_test_kappa)

        end = time.time()
        runtime = end - start
        print('set %d, epoch %d, train loss: %.4f, train kappa: %.4f, dev loss: %.4f, dev kappa: %.4f, time: %.2f'
                % (set_id, epoch, train_loss, train_kappa, test_loss, test_kappa, runtime))

def experiment(essays, embedding, embed_size, device):
    num_epochs = 60
    inception_size = 100
    hidden_size = 300
    batch_size = 64
    topk = 3
    lr = 1e-4
    for set_id in range(1,2):
        for i in range(3):
            print('set: %d' % set_id)
            train_loader, test_loader, min_label, max_label = loadData(essays, set_id, batch_size)
            net = EssayNet(embedding, embed_size, inception_size, hidden_size, topk, min_label, max_label).to(device)
            loss_func = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            train(net, set_id, train_loader, test_loader, loss_func, optimizer, num_epochs, lr, device)

def main():
    glove = KeyedVectors.load_word2vec_format('/home/lj/dataset/glove/w2v.6B.300d.txt')
    essays = pd.read_csv('HW2/data/essays.csv')
    corpus = essays['words'].values
    device = torch.device('cuda', 0)

    embed_size = 300
    vocab = createVocab(corpus, glove)
    embedding = getEmbedding(glove, vocab, embed_size)
    experiment(essays, embedding, embed_size, device)
    print('===========================================================')
    print(max_test_kappa)

if __name__ == "__main__":
    main()
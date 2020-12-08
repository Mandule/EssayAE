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
# [0.7589694231338838, 0.5874508068488263, 0.5879492892363011, 0.598078340152131, 0.630621149100407, 0.6251465263171478, 0.7145893347245704, 0.4456838283263835]
max_test_kappa = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0}

class EssayDataset(data.Dataset):
    def __init__(self, essays, labels, prompts):
        self.essays = essays
        self.labels = labels
        self.prompts = prompts
    def __getitem__(self, index):
        essay = self.essays[index]
        label = self.labels[index]
        prompt = self.prompts[index]
        return essay, label, prompt
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
    def __init__(self, embedding, embedding_size, inception_size, hidden_size, topk):
        super(EssayNet, self).__init__()
        self.topk = topk
        self.embed = nn.Embedding.from_pretrained(embedding)
        self.conv = nn.Sequential(
            Inception(embedding_size, inception_size),
            Inception(inception_size, inception_size))
        self.rnn = nn.LSTM(input_size=inception_size, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=True, dropout=0.5, num_layers=2)
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

class EssayPromptNet(nn.Module):
    def __init__(self, embedding, embedding_size, inception_size1, hidden_size1, inception_size, hidden_size, topk):
        super(EssayPromptNet, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding)
        self.promptNet = nn.Sequential(
            Inception(embedding_size, inception_size1),
            Inception(inception_size1, inception_size1),
            Inception(inception_size1, hidden_size1))
        self.essaysNet = EssayNet(embedding, embedding_size, inception_size, hidden_size, topk)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size1 + hidden_size * 2 * topk, hidden_size * topk),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * topk, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.init_fc()
    
    def forward(self, sens, prompts, lengths):
        # prompts.shape = (batch_size, max_ln)
        prompts = self.embed(prompts)
        # prompts.shape = (batch_size, max_ln, embedding_size)
        prompts = self.promptNet(prompts.permute(0, 2, 1))
        # prompts.shape = (batch_size, Inception_size, max_ln)
        prompts = F.max_pool1d(prompts, prompts.shape[-1]).squeeze(-1)
        # prompts.shape = (batch_size, Inception_size)
        essays = self.essaysNet(sens, lengths)
        # essays.shape = (batch_size, 2 * hidden_size * topk)
        essay_prompt = torch.cat((essays, prompts), dim=1)
        # essay_prompt.shape = (batch_size, 2 * hidden_size * topk + Inception_size)
        score = self.fc(essay_prompt).squeeze(-1)
        # socre.shape = batch_size
        return score
    
    def kmax_pooling(self, x):
        index = x.topk(self.topk, dim = 2)[1].sort(dim = 2)[0]
        return x.gather(2, index)
    
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
            if word.strip() == '##':
                continue
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
    prompts = []
    lengths = []
    
    for data in batch_data:
        essay = data[0]
        label = data[1]
        prompt = data[2]
        
        length = len(essay)
        essay = [list(map(int, [word for word in sen.strip().split(' ') if word.isdigit()])) for sen in essay]
        prompt = list(map(int, [word for word in prompt.strip().split(' ') if word.isdigit()]))
        
        sentences.extend(essay)
        labels.append(label)
        prompts.append(prompt)
        lengths.append(length)
        
    sentences = [torch.LongTensor(sentence) for sentence in sentences]
    prompts = [torch.LongTensor(prompt) for prompt in prompts]
    
    prompts = utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0)
    sentences = utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.FloatTensor(labels)
    lengths = torch.IntTensor(lengths)
    
    return sentences, prompts, labels, lengths

def loadData(essays, set_id, batch_size):
    train_data = essays[(essays['class'] != 'test') & (essays['set'] != set_id)]
    test_data = essays[(essays['class'] == 'test') & (essays['set'] == set_id)]
    
    train_essays = [essay.split(' ## ') for essay in train_data['index'].values.tolist()]
    train_prompts = train_data['prompt_index'].values
    train_labels = train_data['score'].values

    test_essays = [essay.split(' ## ') for essay in test_data['index'].values.tolist()]
    test_prompts = test_data['prompt_index'].values
    test_labels = test_data['score'].values
    
    train_dataset = EssayDataset(train_essays, train_labels, train_prompts)
    test_dataset = EssayDataset(test_essays, test_labels, test_prompts)
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

def adjust_lr(optimizer, epoch, lr_origin):
    lr = lr_origin * (0.9 ** (epoch // 2))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr

def train(net, topk, set_id, min_score, max_score, train_loader, test_loader, loss_func, optimizer, num_epochs, lr_origin, device):
    import time
    for epoch in range(num_epochs):
        adjust_lr(optimizer, epoch, lr_origin)
        start = time.time()
        train_loss, test_loss = 0.0, 0.0
        train_kappa, test_kappa = 0.0, 0.0
        n, m = 0, 0

        for essays, prompts, labels, lengths in train_loader:
            essays = essays.to(device)
            prompts = prompts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            y = net(essays, prompts, lengths)
            loss = loss_func(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.shape[0]
            n += labels.shape[0]
        train_loss /= n
        train_kappa /= n

        net.eval()
        with torch.no_grad():
            for essays, prompts, labels, lengths in test_loader:
                essays = essays.to(device)
                prompts = prompts.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                y = net(essays, prompts, lengths)
                y = y * (max_score - min_score) + min_score
                labels = labels * (max_score - min_score) + min_score
                loss = loss_func(y, labels)

                test_loss += loss.item() * labels.shape[0]
                test_kappa += metrics.kappa(y.cpu().detach().numpy(), labels.cpu().detach().numpy(), 'quadratic') * labels.shape[0]
                m += labels.shape[0]
        net.train()
        test_loss /= m
        test_kappa /= m

        end = time.time()
        runtime = end - start
        print('set %d, epoch %d, train loss: %.4f, test loss: %.4f, test kappa: %.4f, time: %.2f'
                % (set_id, epoch, train_loss, test_loss, test_kappa, runtime))

        if test_kappa > max_test_kappa[set_id]:
            max_test_kappa[set_id] = test_kappa
            torch.save(net.state_dict(), 'HW3/models_prompt/model_' + str(set_id) + '.pt')
            print(max_test_kappa)

def experiment(essays, embedding, embed_size, device):
    num_epochs = 10
    inception_size1 = 100
    hidden_size1 = 200
    inception_size = 100
    hidden_size = 300
    batch_size = 64
    topk = 3
    lr_origin = 1e-3
    for set_id in range(1,9):
        for _ in range(2):
            train_loader, test_loader, min_score, max_score = loadData(essays, set_id, batch_size)
            net = EssayPromptNet(embedding, embed_size, inception_size1, hidden_size1, inception_size, hidden_size, topk).to(device)
            loss_func = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lr_origin)
            train(net, topk, set_id, min_score, max_score, train_loader, test_loader, loss_func, optimizer, num_epochs, lr_origin, device)

def loadPrompt():
    prompt = []
    prompt.append("More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.")
    prompt.append("Censorship in the Libraries. \"All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.\" --Katherine Paterson, Author. Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and, or reading.")
    prompt.append("Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.")
    prompt.append("Read the last paragraph of the story. \"When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again.\" Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.")
    prompt.append("Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.")
    prompt.append("Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.")
    prompt.append("Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.")
    prompt.append("We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.")
    prompt = np.array(prompt)
    return prompt

def main():
    device = torch.device('cuda', 1)
    print('loading the glove')
    glove = KeyedVectors.load_word2vec_format('/home/lj/dataset/glove/w2v.6B.300d.txt')
    essays = pd.read_csv('HW3/data/essays_prompt.csv')
    prompts = loadPrompt()
    corpus = np.concatenate((essays['sentences'].values, prompts))
    embed_size = 300
    vocab = createVocab(corpus, glove)
    embedding = getEmbedding(glove, vocab, embed_size)
    print('begin experiment')
    experiment(essays, embedding, embed_size, device)
    print(max_test_kappa)

if __name__ == "__main__":
    main()
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
from HW3.src.prompt import CNN_BLSTM


class EssayDataset(data.Dataset):
    def __init__(self, essays, labels, ids, prompts):
        self.essays = essays
        self.labels = labels
        self.ids = ids
        self.prompts = prompts
    def __getitem__(self, index):
        essay = self.essays[index]
        label = self.labels[index]
        essay_id = self.ids[index]
        prompt = self.prompts[index]
        return essay, label, essay_id, prompt
    def __len__(self):
        return len(self.labels)

def loadNet(embedding, embed_size, set_id, device):
    inception_size1 = 100
    hidden_size1 = 200
    inception_size = 100
    hidden_size = 300
    batch_size = 64
    topk = 3
    net = CNN_BLSTM.EssayPromptNet(embedding, embed_size, inception_size1, hidden_size1, inception_size, hidden_size, topk).to(device)
    path = 'HW3/models_prompt/model_' + str(set_id) + '.pt'
    net.load_state_dict(torch.load(path))
    return net

def collate_fn(batch_data):
    batch_data.sort(key=lambda data: len(data[0]), reverse=True)
    sentences = []
    lengths = []
    labels = []
    prompts = []
    ids = []

    for data in batch_data:
        essay = data[0]
        label = data[1]
        essay_id = data[2]
        prompt = data[3]
        length = len(essay)

        essay = [list(map(int, [word for word in sen.strip().split(' ') if word.isdigit()])) for sen in essay]
        prompt = list(map(int, [word for word in prompt.strip().split(' ') if word.isdigit()]))

        sentences.extend(essay)
        labels.append(label)
        prompts.append(prompt)
        lengths.append(length)
        ids.append(essay_id)

    sentences = [torch.LongTensor(sentence) for sentence in sentences]
    prompts = [torch.LongTensor(prompt) for prompt in prompts]

    prompts = utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=0)
    sentences = utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    lengths = torch.IntTensor(lengths)
    labels = np.array(labels)
    ids = np.array(ids)
    return sentences, lengths, labels, ids, prompts

def loadData(essays, set_id, batch_size):
    test_data = essays[(essays['class'] == 'test') & (essays['set'] == set_id)]
    test_essays = [essay.split(' ## ') for essay in test_data['index'].values.tolist()]
    test_labels = test_data['score'].values
    test_prompts = test_data['prompt_index'].values
    test_ids = test_data['id'].values
    test_dataset = EssayDataset(test_essays, test_labels, test_ids, test_prompts)
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
        for essays, lengths, labels, ids, prompts in test_loader:
            essays = essays.to(device)
            prompts = prompts.to(device)
            lengths = lengths.to(device)
            pred = net(essays, prompts, lengths)
            
            pred = pred * (max_score - min_score) + min_score
            labels = labels * (max_score - min_score) + min_score

            pred = pred.cpu().detach().numpy()
            test_kappa += metrics.kappa(pred, labels, 'quadratic') * len(labels)

            preds.extend(pred.round().tolist())
            essays_ids.extend(ids.tolist())
            m += len(labels)
    test_kappa /= m
    return preds, essays_ids, test_kappa

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

    embed_size = 300
    prompts = loadPrompt()
    corpus = np.concatenate((essays['sentences'].values, prompts))
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
    result.to_csv('HW3/data/prompt.tsv', sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
    
















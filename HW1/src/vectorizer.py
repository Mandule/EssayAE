import pandas as pd
import numpy as np
import sklearn as skl
import enchant
import nltk
from nltk.tree import Tree
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('/home/lj/dataset/stanfordCoreNLP')

import sys
sys.path.append('.')
import metrics

def segment(essay):
    import re
    words = [word.strip().lower() for word in nltk.word_tokenize(re.sub(r'@[a-zA-Z0-9]+', '', essay).replace('/', ' or '))]
    return ' '.join(words)

def splitSen(essay):
    sentences = [segment(sen) for sen in nltk.sent_tokenize(essay)]
    return ' ## '.join(sentences)

# 1 2
def getMeanVarOfWords(words):
    words = words.split(' ')
    lens = [len(item) for item in words]
    mean = np.mean(lens)
    var = np.var(lens)
    return mean, var

#3 4
def getMeanVarOfSens(sens):
    sens = sens.split(' ## ')
    lens = [len(item) for item in sens]
    mean = np.mean(lens)
    var = np.var(lens)
    return mean, var
    
# 5 6 7
def getLen(essay, words, sentences):
    return len(essay), len(words.split(' ')), len(sentences.split(' ## '))

# 8 9 10
def getTag(words):
    tags = np.array(nltk.pos_tag(words.split(' ')))
    num_IN = 0
    num_RB = 0
    num_comma = 0
    for tag in tags:
        if tag[1] == 'IN':
            num_IN += 1
        if tag[1] == 'RB':
            num_RB += 1
        if  tag[1] == ',':
            num_comma += 1
    return num_IN, num_RB, num_comma

# 11
def getUniqueWord(words):
    wordDic = {}
    for word in words.split(' '):
        if word in wordDic.keys():
            wordDic[word] += 1
        else:
            wordDic[word] = 1
    num = 0
    for word, count in wordDic.items():
        if count == 1:
            num += 1
    return num

# 12 13 14 15
def getClause(sentences):
    sentences = sentences.split(' ## ')
    num_clauses = [len(list(sen.split(','))) for sen in sentences]
    len_clauses = [len(clause) for sen in sentences for clause in sen.split()]
    mean1 = np.mean(num_clauses)
    mean2 = np.mean(len_clauses)
    max1 = np.max(num_clauses)
    max2 = np.max(len_clauses)
    return mean1, mean2, max1, max2

# 16
def getSpellError(words):
    dic = enchant.Dict('en_US')
    num = 0
    for word in words.split(' '):
        if not word.isalpha():
            continue
        if dic.check(word):
            continue
        num += 1
    return num

# 17 18
def getDepth(sentences):
    sentences = sentences.split(' ## ')
    trees = [Tree.fromstring(nlp.parse(sen)) for sen in sentences]
    leafDepth = [getleafDepth(tree) for tree in trees]
    depths = [tree.height() for tree in trees]
    avg1 = np.mean(depths)
    avg2 = np.mean(leafDepth)
    return avg1, avg2

def getleafDepth(tree):
    leafDepth = []
    def helper(root, depth, leafDepth):
        if(type(root) == str):
            leafDepth.append(depth)
            return
        for i in range(len(root)):
            helper(root[i], depth+1, leafDepth)
    helper(tree, 0, leafDepth)
    avg = np.mean(leafDepth)
    return avg

def loadData():
    train_data = pd.read_csv('EssayAS/essay_data/train.tsv', sep='\t', usecols=['essay_id' , 'essay_set', 'essay', 'domain1_score'])
    dev_data = pd.read_csv('EssayAS/essay_data/dev.tsv', sep='\t', usecols=['essay_id' , 'essay_set', 'essay', 'domain1_score'])
    test_data = pd.read_csv('EssayAS/essay_data/test.tsv', sep='\t', usecols=['essay_id', 'essay_set', 'essay'])
    train_data = train_data.rename(columns={'essay_id':'id','essay_set':'set', 'domain1_score':'score'})
    dev_data = dev_data.rename(columns={'essay_id':'id','essay_set':'set', 'domain1_score':'score'})
    test_data = test_data.rename(columns={'essay_id':'id','essay_set':'set'})
    test_data['score'] = None
    test_data['class'] = 'test'
    dev_data['class'] = 'dev'
    train_data['class'] = 'train'
    essay =  pd.concat([train_data, test_data, dev_data])
    return essay

def feature(essay):
    essay['words'] = essay['essay'].apply(segment)
    essay['sentences'] = essay['essay'].apply(splitSen)
    essay['x1'] = essay['words'].apply(getMeanVarOfWords)
    essay['x2'] = essay['x1'].apply(lambda x: x[1])
    essay['x1'] = essay['x1'].apply(lambda x: x[0])
    essay['x3'] = essay['sentences'].apply(getMeanVarOfSens)
    essay['x4'] = essay['x3'].apply(lambda x: x[1])
    essay['x3'] = essay['x3'].apply(lambda x: x[0])
    essay['x5'] = essay['essay'].apply(len)
    essay['x6'] = essay['words'].apply(lambda x: len(x.split(' ')))
    essay['x7'] = essay['sentences'].apply(lambda x: len(x.split(' ## ')))
    essay['x8'] = essay['words'].apply(getTag)
    essay['x9'] = essay['x8'].apply(lambda x: x[1])
    essay['x10'] = essay['x8'].apply(lambda x: x[2])
    essay['x8'] = essay['x8'].apply(lambda x: x[0])
    essay['x11'] = essay['words'].apply(getUniqueWord)
    essay['x12'] = essay['sentences'].apply(getClause)
    essay['x13'] = essay['x12'].apply(lambda x: x[1])
    essay['x14'] = essay['x12'].apply(lambda x: x[2])
    essay['x15'] = essay['x12'].apply(lambda x: x[3])
    essay['x12'] = essay['x12'].apply(lambda x: x[0])
    essay['x16'] = essay['words'].apply(getSpellError)
    # essay['x17'] = essay['sentences'].apply(getDepth)
    # essay['x18'] = essay['x17'].apply(lambda x: x[1])
    # essay['x17'] = essay['x17'].apply(lambda x: x[0])

    def standar(data):
        for i in range(1, 17):
            col = 'x' + str(i)
            data[col] = data[col].apply(float)
            mean = data[col].mean()
            var = data[col].std()
            data[col] = data[col].apply(lambda x: (x - mean) / var)
    standar(essay)

def main():
    essay = pd.read_csv('data/essays.csv')
    feature(essay)
    essay.to_csv('HW1/data/essays_16.csv', index=False)

if __name__ == "__main__":
    main()
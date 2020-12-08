import sys
sys.path.append('.')
import metrics

import pandas as pd
import numpy as np


essays = pd.read_csv('HW1/data/essays_16.csv')
essays = essays[['id', 'set', 'class', 'score', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']]

svm_kappa = []
lr_kappa = []
svm_label = []
lr_label = []
essay_id = []
for i in range(1, 9):
    set_data = essays[essays['set'] == i]
    train_data = set_data[set_data['class'] != 'test']
    test_data = set_data[set_data['class'] == 'test']
    
    train_x = train_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']]
    train_y = train_data['score']
    
    test_id = test_data['id'].values.tolist()
    test_x = test_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']]
    test_y = test_data['score']
    
    from sklearn.svm import SVC
    svm = SVC(gamma='scale')
    svm.fit(train_x.values, train_y.values)
    pred_label = svm.predict(test_x.values)
    acc = metrics.kappa(test_y.values, pred_label, 'quadratic')
    svm_label.append(pred_label.tolist())
    svm_kappa.append(acc)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', penalty='l2')
    lr.fit(train_x.values, train_y.values)
    pred_label = lr.predict(test_x.values).round()
    acc = metrics.kappa(test_y.values, pred_label, 'quadratic')
    lr_label.append(pred_label.tolist())
    lr_kappa.append(acc)
    
    
    essay_id.append(test_id)

print(svm_kappa)
print(np.mean(svm_kappa))

print(lr_kappa)
print(np.mean(lr_kappa))


test_label = []
for i in range(8):
    if i == 0:
        test_label.append(svm_label[i])
    else:
        test_label.append(lr_label[i])

result = []
for i in range(8):
    for j in range(len(essay_id[i])):
        result.append([int(essay_id[i][j]), i+1, int(svm_label[i][j])])

result = pd.DataFrame(result)

result.to_csv('HW1/data/MF1933059.tsv', sep='\t', index=False, header=False)
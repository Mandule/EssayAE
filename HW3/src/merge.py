import pandas as pd

result1 = pd.read_csv('HW3/data/normal.tsv', sep='\t', header=None, names=['id', 'set', 'score'])
result2 = pd.read_csv('HW3/data/prompt.tsv', sep='\t', header=None, names=['id', 'set', 'score'])

set4 = result1[result1['set'] == 4]
set5 = result1[result1['set'] == 5]
set6 = result1[result1['set'] == 6]
set8 = result1[result1['set'] == 8]

set1 = result2[result2['set'] == 1]
set2 = result2[result2['set'] == 2]
set3 = result2[result2['set'] == 3]
set7 = result2[result2['set'] == 7]

result = pd.concat([set1, set2, set3, set4, set5, set6, set7, set8])
result.to_csv('HW3/data/MF1933059.tsv', sep='\t', index=False, header=False)
print(result.head())

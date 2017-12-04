import json
import numpy as np
import pandas as pd
from functools import reduce 
import os

pipeline = os.environ['PIPELINE']

with open('pipeline/data/tokens_{}.json'.format(pipeline), 'r') as f:
     tokens = json.load(f)
"""
tokens = {
    0: {
        "word": 4,
        "me": 1
        }
    ...
}
"""
columns = list(set(reduce(lambda x, y: x + list(y.keys()), tokens.values(), [])))
token_matrix = np.zeros((len(tokens.keys()), len(columns)))

for img_num, sentence in tokens.items():
    for token in sentence.keys():
        token_matrix[int(img_num)][columns.index(token)] = sentence[token]

df = pd.DataFrame(token_matrix, columns=columns)
df.to_csv('pipeline/data/tokens_{}.csv'.format(pipeline))

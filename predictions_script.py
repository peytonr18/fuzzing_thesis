import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json


df = pd.read_json('test.json', lines=True)
df2 = pd.DataFrame(df)

#with open('test.json', 'r') as f:
		#json_str = f.read()

#json_str = json_str.replace('\n', ' ')

#data = json.loads(json_str)

#idx_target_results = [item['idx'] + '\t' + item['target'] for item in data]

df = df[['idx', 'target']]
df.to_csv('predictions.txt', sep='\t', index=False)

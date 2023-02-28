import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

df = pd.read_json('function_new.json')
df2 = pd.DataFrame(df)

df = df.rename({'func' : 'function'}, axis=1)
df = df['function'].apply(lambda x: x.replace('\n', ''))

training_data = df.sample(frac=0.7, random_state=25)
testing_data = df.drop(training_data.index)
validation_data = df.drop(training_data.index)


validation_data = testing_data.sample(frac = .1, random_state=25)
testing_data = testing_data.drop(validation_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")
print(f"No. of testing examples: {validation_data.shape[0]}")

print(training_data.index)
print(testing_data.index)
print(validation_data.index)


with open('train.txt', 'w') as f:
	for i, index in enumerate(training_data.index):
		f.write(str(index) + '\n')

with open('test.txt', 'w') as f:
	for i, index in enumerate(testing_data.index):
		f.write(str(index) + '\n')

with open('validate.txt', 'w') as f:
	for i, index in enumerate(validation_data.index):
		f.write(str(index) + '\n')
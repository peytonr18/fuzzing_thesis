import pandas as pd
import numpy as np

import json

if __name__ == '__main__':
	j = pd.read_json('function.json')
	# print(type(j))
	j = j.rename({'func' : 'function'}, axis=1)
	j['function'] = j['function'].apply(lambda x: x.replace('\n', ''))
	j.to_json('function_new.json', orient='records')
	# j = pd.read_json('function_new.json')
	# print(type(j))


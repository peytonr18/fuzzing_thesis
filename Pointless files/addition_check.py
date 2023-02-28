import pandas as pd
df = pd.read_json('function.json')
#df = df.drop(['commit id'], axis=1)
print(df.iloc[-1])


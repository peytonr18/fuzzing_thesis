# Python script to process the dataset as necessary for training, testing, and validation purposes. 
# 25% for testing, 10% for validation, and 70% for training. 


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
vulnDB=json.load(open('function_new.json', 'rb'))

train_index=set()
validate_index=set()
test_index=set()

with open('train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('validate.txt') as f:
    for line in f:
        line=line.strip()
        validate_index.add(int(line))
        
with open('test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))
    
with open('train.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('validate.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in validate_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('test.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')

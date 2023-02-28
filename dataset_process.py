# Python script to process the dataset as necessary for training, testing, and validation purposes. 
# 25% for testing, 10% for validation, and 70% for training. 


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
vulnDB=json.load(open('function_new.json', 'rb'))

train_index=set()
validate_index=set()
test_index=set()

with open('/Users/peytonrobertson/fuzzing_thesis/train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('/Users/peytonrobertson/fuzzing_thesis/validate.txt') as f:
    for line in f:
        line=line.strip()
        validate_index.add(int(line))
        
with open('/Users/peytonrobertson/fuzzing_thesis/test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))
    
with open('/Users/peytonrobertson/fuzzing_thesis/train.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('/Users/peytonrobertson/fuzzing_thesis/validate.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in validate_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('/Users/peytonrobertson/fuzzing_thesis/test.json','w') as f:
    for idx,js in enumerate(vulnDB):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
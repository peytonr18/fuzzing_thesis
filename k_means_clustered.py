import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
import torch.nn.utils.rnn as rnn_utils
import torch
import matplotlib.patches as mpatches
import itertools
import random
from sklearn.cluster import KMeans

# Load the files
tokenizer_config = "trained_model/roberta_model/tokenizer_config.json"
special_tokens_map = "trained_model/roberta_model/special_tokens_map.json"
vocab = "trained_model/roberta_model/vocab.json"

with open('function_new.json', 'r') as f:
    file_contents = f.read()
    data = json.loads(file_contents)

random.shuffle(data)
new_function = data[:500]

# Set the path to the directory containing the pre-trained model files
pretrained_model_name_or_path = 'trained_model/roberta_model'

# Create the tokenizer object
tokenizer = RobertaTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    tokenizer_config=tokenizer_config,
    special_tokens_map=special_tokens_map,
    vocab_file=vocab
)

tokens=[]
labels = []
colors=[]
max_length=512
for item in new_function:
    token = tokenizer(
        item["function"],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        pad_to_max_length=True
    )["input_ids"]
    label = item["target"]
    tokens.append(token)
    labels.append(label)

# Pad the input sequences
padded_tokens = rnn_utils.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

# Convert the tensors to numpy arrays
padded_tokens = padded_tokens.numpy()
labels = np.array(labels)

# Concatenate the padded tokens along the second dimension
tokens = np.concatenate(padded_tokens, axis=0)

# Fit the KMeans algorithm to the data and predict the cluster labels
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(tokens)

# Assign a unique color to each cluster label
for label in cluster_labels:
    if label == 1:
        colors.append("red")
    else:
        colors.append("blue")

# Perform TSNE
perplexity = min(30, len(tokens) - 1)
perplexity = max(perplexity, 1e-5)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=294967295)
embeddings = tsne.fit_transform(tokens)

# Plot the embeddings with the cluster labels
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Define custom labels for the color legend
legend_labels = ['Non-Vulnerable', 'Vulnerable']
# Create the color legend with the custom labels
plt.legend(handles=[mpatches.Patch(color='blue', label=legend_labels[0]),
                    mpatches.Patch(color='red', label=legend_labels[1])])
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)


plt.savefig('KMeans_TSNE.png')

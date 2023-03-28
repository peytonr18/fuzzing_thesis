import json
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# Load the JSON file into a Python object

with open('function_new.json', 'r') as f:
    file_contents = f.read()
    data = json.loads(file_contents)

smaller_data = data[:25]

# Extract the values from the desired column and store them in an array
columns = ['function', 'target']
#values_array = [item[column_name] for item in data]
values_array=[]

# Iterate over each row of data and extract the key-value pairs
dict_data = {}
for row in smaller_data:
    key = row['function']
    value = row['target']
    dict_data[key] = value

# Use a for loop to iterate through the dictionary keys
for key in dict_data:
    # Convert the value to a string using the str() function
    str_value = str(key)
    # Append the string representation of the value to the array
    values_array.append(str_value)

# Print the resulting array
print(values_array)

plt.clf()

colors = []

# Iterate over the dictionary keys
for key in dict_data:
    # Check if the key matches the value in the given array and has a value of 1
    if dict_data[key] == 1:
        # If so, append 'blue' to the list of colors
        colors.append('blue')
    else:
        # If not, append 'orange' to the list of colors
        colors.append('orange')

# Extract the token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state[:, 0, :].numpy()

# Perform dimensionality reduction using T-SNE
#tsne = TSNE(n_components=2, perplexity=10, random_state=0)
#embeddings_tsne = tsne.fit_transform(embeddings)
pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(embeddings.reshape(-1, embeddings.shape[-1]))
tsne = TSNE(n_components=2, perplexity=10, random_state=0)
tsne_vectors = tsne.fit_transform(pca_vectors)


# Create a scatter plot of the embeddings
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Define custom labels for the color legend
legend_labels = ['Vulnerable', 'Not Vulnerable']

# Create the color legend with the custom labels
plt.legend(handles=[mpatches.Patch(color='blue', label=legend_labels[0]), 
                    mpatches.Patch(color='orange', label=legend_labels[1])])

plt.scatter(tsne_vectors[:,0], tsne_vectors[:,1], c=colors)

plt.savefig('TSNE_plot.png')

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import json
import argparse
import pickle
parser = argparse.ArgumentParser(description='PCA on mean embeddings from pickle dictionary')

# Add arguments
parser.add_argument('--model', type=str, default='esm2_t33_650M', help='Model name')
parser.add_argument('--mean', action='store_true', help='Use mean embedding')
parser.add_argument('--layer', type=int, default=36, help='Layer number')
parser.add_argument('--max_len', type=int, default=None, help='Maximum length')
parser.add_argument('--components', type=int, default=200, help='Number of components')

args = parser.parse_args()

# Access the arguments
model = args.model
mean = args.mean
layer = args.layer
max_len = args.max_len
components = args.components

# nur auf training fitten

embedding_name = f'{model}/mean' if mean else f'{model}/per_tok'
out = "D:/PPI_prediction_study-master/data/pca/"
directory ="D:/PPI_prediction_study-master/data/Embeddings/embeddins_mean.pkl"


def get_embedding_per_tok(dirpath, protein_id, layer):
    embedding = torch.load(os.path.join(dirpath, protein_id + ".pt"))
    return embedding['representations'][layer]

def get_embedding_mean(dirpath, protein_id, layer):
    embedding = torch.load(os.path.join(dirpath, protein_id + ".pt"))
    return embedding['mean_representations'][layer] 

def padd_embedding(embedding, maximum):
    '''
    Padds the data with 0s to have them all be the same size
    following the better functions
    '''
    padding = torch.zeros((maximum - embedding.shape[0], embedding.shape[1]))
    return torch.cat((embedding, padding), dim=0)


# Initialize PCA
pca = PCA(n_components=components)


emb_list = []
protein_ids = []
i = 0
# Load and flatten the embeddings

for filename in os.listdir(directory):
    if filename.endswith(".pt"):
        protein_id = filename[:-3]
        protein_ids.append(protein_id)
        # Load the embedding
        if mean:
            embedding = get_embedding_mean(directory, protein_id, layer)
        else:
            embedding = get_embedding_per_tok(directory, protein_id, layer)
            embedding = padd_embedding(embedding, max_len)
            embedding = embedding.reshape(-1)
        
        emb_list.append(embedding)

# Convert the list of flattened embeddings to a 2D array
emb_list = np.stack(emb_list)

# Fit the PCA on the reshaped embeddings
pca.fit(emb_list)

# Transform the reshaped embeddings using the fitted PCA
pca_embeddings = pca.transform(emb_list)


protein_dict = {k: v.tolist() for k, v in zip(protein_ids, pca_embeddings)}

print(len(protein_dict))

os.makedirs(out, exist_ok=True)

with open(out + "pca_" + str(components) + ".json", "w") as outfile:
    json.dump(protein_dict, outfile)

#df = pd.DataFrame(protein_ids, pca_embeddings)
#df.to_csv(out + "pca.csv", index=False, header=False)


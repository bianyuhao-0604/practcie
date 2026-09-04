import os
import torch
import pickle


embedding_directory = "D:/PPI_prediction_study-master/data/output_embeddings"

def get_embeddings(dirpath):

    # Create an empty dictionary to store embeddings
    embeddings_dict = {}

    # Iterate over files in the directory
    for filename in os.listdir(dirpath):
        if filename.endswith(".pt"):
            # Extract protein ID from the filename (assuming the filename format is "uniprot_id.pt")
            protein_id = os.path.splitext(filename)[0]
            
            # Load the embedding tensor from the .pt file
            embedding = torch.load(os.path.join(dirpath, filename))
            print(embedding['representations'][33].shape)
            # Store the embedding in the dictionary with the protein ID as the key
            embeddings_dict[protein_id] = embedding
    return embeddings_dict        

'''embdic = get_embeddings(embedding_directory)   

with open('/nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_embed_dict.pkl', 'wb') as f:
    embedding = pickle.dump(embdic, f)
embedding['representations'][33]
'''

#emb = torch.load('D:/PPI_prediction_study-master/data/output_embeddings/Q9Y261.pt')
#print(emb['representations'][33].shape)                 
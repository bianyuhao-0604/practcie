import numpy as np
import pandas as pd
import json
import torch
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import argparse

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN

start_time = time.time()
if False:
    parser = argparse.ArgumentParser(description='Argument Parser for baselineRFC.py')

    # Add arguments
    parser.add_argument('--model', type=str, default='esm2_t48_15B', help='Model name')
    parser.add_argument('--mean', action='store_true', help='Use mean representation')
    parser.add_argument('--layer', type=int, default=48, help='Layer number')
    parser.add_argument('--components', type=int, default=200, help='Number of components for PCA')
    parser.add_argument('--pca', action='store_true', help='Use PCA')
    parser.add_argument('--data_name', type=str, default='gold_stand', help='Data name')

    args = parser.parse_args()

    # Access the arguments
    model = args.model
    mean = args.mean
    layer = args.layer
    components = args.components
    pca = args.pca
    data_name = args.data_name
else:
    model = "esm2_t33_650"
    mean = True
    layer = 33
    components = 20
    pca = True
    data_name = "gold_stand"

embedding_name = f'{model}/mean' if mean else f'{model}/per_tok'
train = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_train_all_seq.csv"
test = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_test_all_seq.csv"
pca_dic = "/nfs/scratch/t.reim/embeddings/" + embedding_name + "/pca/"
embedding_dir = "/nfs/scratch/t.reim/embeddings/" + embedding_name + "/"
max_len = 10000

# params

def get_embedding_mean(dirpath, protein_id, layer):
    embedding = torch.load(os.path.join(dirpath, protein_id + ".pt"))
    return embedding['mean_representations'][layer] 

def create_dataset(mean, pca, embedding_dir, layer, df, max_len):
    XY = pd.DataFrame(columns=['inputs', 'labels'])
    if max_len is None:
        max_len = max(max(df['sequence_a'].apply(len)), max(df['sequence_b'].apply(len)))
    else:
        max_len = max_len
    df = df[(df['sequence_a'].apply(len) <= max_len) & (df['sequence_b'].apply(len) <= max_len)]
    df = df.reset_index(drop=True)
    if mean:
        if not pca:
            for index, row in df.iterrows():
                data = df.iloc[index]
                prot1 = get_embedding_mean(embedding_dir, data['Id1'], layer)
                prot2 = get_embedding_mean(embedding_dir, data['Id2'], layer)
                inputs = np.concatenate((prot1.numpy(), prot2.numpy()))
                labels = data['Interact']
                new_row = pd.DataFrame({'inputs': [inputs], 'labels': [labels]})
                XY = pd.concat([XY, new_row], ignore_index=True)
        else:
            with open(pca_dic + 'pca_' + str(components) + '.json', 'r') as f:
                dic = json.load(f)
            for index, row in df.iterrows():
                data = df.iloc[index]
                prot1 = dic[data['Id1']]
                prot2 = dic[data['Id2']]
                labels = data['Interact']
                inputs = prot1 + prot2
                new_row = pd.DataFrame({'inputs': [inputs], 'labels': [labels]})
                XY = pd.concat([XY, new_row], ignore_index=True)
    return XY        

df_train = pd.read_csv(train) 
df_test = pd.read_csv(test)


XY_train = create_dataset(mean=mean, pca=pca, embedding_dir=embedding_dir, layer=layer, df=df_train, max_len=max_len)
XY_test = create_dataset(mean=mean, pca=pca, embedding_dir=embedding_dir, layer=layer, df=df_test, max_len=max_len)

print(1)

X_train = XY_train['inputs']
y_train = XY_train['labels']

X_test = XY_test['inputs']
y_test = XY_test['labels']

X_train = np.stack(X_train)
X_test = np.stack(X_test)

y_train = np.stack(y_train)
y_test = np.stack(y_test)

print(X_train.shape)
print(X_test.shape)

intermediate_start_time = time.time()

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

TP, FP, TN, FN = perf_measure(y_test, y_pred)

print(f'TP: {TP}')
print(f'FP: {FP}')
print(f'TN: {TN}')
print(f'FN: {FN}')

print(f'Intermediate time: {time.time() - intermediate_start_time} seconds')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')

print(f'Total execution time: {time.time() - start_time} seconds')
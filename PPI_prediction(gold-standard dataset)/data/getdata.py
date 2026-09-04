from Bio import SeqIO
from Bio import SwissProt
from Bio import ExPASy
import numpy as np
import pandas as pd
import requests as r
from io import StringIO


def file2df(file_path, interact):

    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['Id1', 'Id2'])
    df['Interact'] = interact
    return df

def get_seq(cID):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq, 'fasta'))
    return pSeq[0].seq

def addseqtodf(df):
    seq_a = []
    seq_b = []
    for index, row in df.iterrows():
        a = get_seq(row['Id1'])
        b = get_seq(row['Id2'])
        if len(a) <= 1166 and len(b) <=1166:
            seq_a.append(a)
            seq_b.append(b)
    df['sequence_a'] = seq_a
    df['sequence_b'] = seq_b
    return df

def addseqtodf_ff(df, fasta_df, max_len):
    seq_a = []
    seq_b = []
    drop = []
    for index, row in df.iterrows():
        id1 = row['Id1']
        id2 = row['Id2']
        if id1 in fasta_df["ID"].values:
            a = str(fasta_df[fasta_df["ID"] == id1]["Sequence"].values[0])
        else:
            drop.append(index)    
            continue
            a = get_seq(id1)  
            print(id1)
        if id2 in fasta_df["ID"].values:
            b = str(fasta_df[fasta_df["ID"] == id2]["Sequence"].values[0])
        else: 
            drop.append(index)   
            continue
            b = get_seq(id2) 
            print(id2)
        if len(a) <= max_len and len(b) <= max_len:
            seq_a.append(a)
            seq_b.append(b)
        else:
            drop.append(index)          
    df = df.drop(drop)          
    df['sequence_a'] = seq_a
    df['sequence_b'] = seq_b
    return df

def read_fasta_as_df(fasta):
    with open(fasta) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    seqs = [x for x in content if x[0] != '>']
    ids = [x.split('>')[1] for x in content if x[0] == '>']
    df = pd.DataFrame()
    df["ID"] = ids
    df["Sequence"] = seqs
    return df


fasta_df = read_fasta_as_df('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta')


print(fasta_df)

train_pos = file2df("bachelor/pytorchtest/data/richoux_regular/richoux_regular_train_pos.txt", 1)
train_neg = file2df("bachelor/pytorchtest/data/richoux_regular/richoux_regular_train_neg.txt", 0)
test_pos = file2df("bachelor/pytorchtest/data/richoux_regular/richoux_regular_test_pos.txt", 1)
test_neg = file2df("bachelor/pytorchtest/data/richoux_regular/richoux_regular_test_neg.txt", 0)

print(1)
print(train_pos)

train_all = pd.concat([train_pos, train_neg]).reset_index()
test_all = pd.concat([test_pos, test_neg]).reset_index()

print(2)
print(train_all)




train_all_seq = addseqtodf_ff(train_all, fasta_df, 10000)
test_all_seq = addseqtodf_ff(test_all, fasta_df, 10000)

train_all_seq.to_csv('bachelor/pytorchtest/data/richoux_regular/richoux_regular_train_all_seq.csv')
test_all_seq.to_csv('bachelor/pytorchtest/data/richoux_regular/richoux_regular_test_all_seq.csv')


'''
train = pd.read_csv('bachelor/pytorchtest/data/train_all_seq_1166.csv')
ones = train['Interact'].value_counts().get(1, 0)
zeros = train['Interact'].value_counts().get(0, 0)
print("ones: "+str(ones)+", zeros: "+str(zeros))

test = pd.read_csv('bachelor/pytorchtest/data/test_all_seq_1166.csv')
ones = test['Interact'].value_counts().get(1, 0)
zeros = test['Interact'].value_counts().get(0, 0)
print("ones: "+str(ones)+", zeros: "+str(zeros))
'''
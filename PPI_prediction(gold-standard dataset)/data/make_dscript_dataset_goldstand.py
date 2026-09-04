from Bio import SeqIO
import pandas as pd
from io import StringIO

def file2df(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['Id1', 'Id2', 'Interact'])
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
        if id2 in fasta_df["ID"].values:
            b = str(fasta_df[fasta_df["ID"] == id2]["Sequence"].values[0])
        else: 
            drop.append(index)   
            continue
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

train = file2df("bachelor/pytorchtest/data/gold_stand_dscript/Intra1.txt")
test = file2df("bachelor/pytorchtest/data/gold_stand_dscript/Intra2.txt")
val = file2df("bachelor/pytorchtest/data/gold_stand_dscript/Intra0.txt")

print(1)

train_all = train.reset_index()
test_all = test.reset_index()
val_all = val.reset_index()

print(2)

train_all_seq = addseqtodf_ff(train_all, fasta_df, 10000)
test_all_seq = addseqtodf_ff(test_all, fasta_df, 10000)
val_all_seq = addseqtodf_ff(val_all, fasta_df, 10000)

train_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand_dscript/gold_stand_dscript_all_seq.csv', index=False)
test_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand_dscript/gold_stand_dscript_test_all_seq.csv', index=False)
val_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand_dscript/gold_stand_dscript_val_all.csv', index=False)
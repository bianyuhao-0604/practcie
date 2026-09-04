import pandas as pd
import ast
import pickle

def main(model, dataset):
    mapp = {
        'Intra0': 'val',
        'Intra1': 'train',
        'Intra2': 'test',
    }
    def sort_ids(row):
        return tuple(sorted((row[0], row[1])))

    sett = mapp[dataset]
    # Read the CSV files
    df1 = pd.read_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confident_{sett}_pred_{model}.csv', sep='\t')
    df2 = pd.read_csv(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/ppis_with_structures_{dataset}_pos_rr.csv', sep=',',skiprows=1)


    pairs1 = [sort_ids(row) for row in df1.values]
    pairs2 = {sort_ids(row): row[2] for row in df2.values}

    # Find the intersection of the two sets
    common_pairs = {pair: pairs2[pair] for pair in pairs1 if pair in pairs2}

    print(f'Found {len(common_pairs)} common pairs')

    confpred_dict = {pair: ast.literal_eval(common_pairs[pair])[0] if ast.literal_eval(common_pairs[pair]) else None for pair in common_pairs}
    with open(f'/nfs/home/students/t.reim/bachelor/pytorchtest/data/gold_stand/confpred_dict_{dataset}_{model}.pkl', 'wb') as f:
        pickle.dump(confpred_dict, f)

if __name__ == "__main__":
    main(dataset='Intra2', model='baseline2d')        
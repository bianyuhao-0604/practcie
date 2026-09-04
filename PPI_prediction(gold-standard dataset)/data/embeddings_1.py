import os
import torch
import pickle
import h5py
import numpy as np

embedding_directory = "D:/PPI_prediction_study-master/data/output_embeddings"
output_h5 = "D:/PPI_prediction_study-master/data/embeddings_per_tok.h5"
output_mean_pkl = "D:/PPI_prediction_study-master/data/embeddings_mean.pkl"

def extract_and_store(dirpath):
    mean_dict = {}
    with h5py.File(output_h5, 'w') as hf:
        for filename in os.listdir(dirpath):
            if not filename.endswith(".pt"):
                continue
            protein_id = os.path.splitext(filename)[0]
            path = os.path.join(dirpath, filename)

            # 加载 .pt 文件
            emb = torch.load(path, map_location="cpu")
            token_repr = emb['representations'][33]  # [L, 1280]

            # 保存 per-token 到 HDF5（按需访问）
            hf.create_dataset(protein_id, data=token_repr.numpy())

            # 计算 mean（per-protein），这里取所有 token 的平均，也可去掉特殊 token
            # 如果要忽略首尾特殊 token，用 token_repr[1:-1].mean(dim=0)
            mean_repr = token_repr.mean(dim=0).numpy()  # [1280]
            mean_dict[protein_id] = mean_repr

            print(f"已处理 {protein_id}: per-token {tuple(token_repr.shape)}, mean {mean_repr.shape}")

    # 保存 mean 字典
    with open(output_mean_pkl, 'wb') as f:
        pickle.dump(mean_dict, f)

    print(f"完成！per-token 已存入 {output_h5}，mean 已存入 {output_mean_pkl}")
    return mean_dict

if __name__ == "__main__":
    mean_dict = extract_and_store(embedding_directory)
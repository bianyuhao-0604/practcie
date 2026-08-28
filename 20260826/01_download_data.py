"""
01_download_data.py
下载所有需要的原始数据文件
"""

import os
import urllib.request
import zipfile
import io

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url: str, save_path: str, desc: str = ""):
    """下载单个文件，支持断点续传"""
    if os.path.exists(save_path):
        print(f"[跳过] {desc} 已存在: {save_path}")
        return
    print(f"[下载] {desc}: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[完成] {desc} -> {save_path}")


def download_huri_data():
    """
    下载 HuRI 人类参考互作组数据（HI-II-14 / HI-III / Lit-BM）
    来源: http://interactome-atlas.org
    """
    print("\n" + "=" * 60)
    print("下载 HuRI 人类参考互作组数据")
    print("=" * 60)

    # HI-II-14: ~14,000 个二元 PPI（Rolland et al., 2014）
    download_file(
        url = "http://interactome-atlas.org/data/HI-II-14.tsv",
        save_path=os.path.join(DATA_DIR, "HI-II-14.tsv"),
        desc="HI-II-14 (14k binary PPIs)"
    )

    # HI-III (HuRI): ~53,000 个二元 PPI（Luck et al., 2020）
    download_file(
        url = "http://interactome-atlas.org/data/HuRI.tsv",
        save_path=os.path.join(DATA_DIR, "HI-III.tsv"),
        desc="HI-III / HuRI (53k binary PPIs)"
    )

    # Lit-BM: 文献策展的二元多证据 PPI
    download_file(
        url = "http://interactome-atlas.org/data/Lit-BM.tsv",
        save_path=os.path.join(DATA_DIR, "Lit-BM.tsv"),
        desc="Lit-BM (literature-curated binary PPIs)"
    )


def download_shs_data():
    """
    下载 SHS27k 和 SHS148k 评测基准数据
    来源: Zenodo (https://doi.org/10.5281/zenodo.7213401)
    """
    print("\n" + "=" * 60)
    print("下载 SHS27k / SHS148k 评测基准数据")
    print("=" * 60)

    base_url = "https://zenodo.org/records/7213401/files"

    files = [
        ("protein.actions.SHS27k.STRING.pro2.txt", "SHS27k PPI data"),
        ("protein.actions.SHS148k.STRING.txt", "SHS148k PPI data"),
        ("protein.SHS27k.sequences.dictionary.pro3.tsv", "SHS27k protein sequences"),
        ("protein.SHS148k.sequences.dictionary.tsv", "SHS148k protein sequences"),
    ]

    for filename, desc in files:
        download_file(
            url=f"{base_url}/{filename}",
            save_path=os.path.join(DATA_DIR, filename),
            desc=desc
        )


def download_string_data(score_threshold: int = 500):
    """
    从 STRING 数据库下载高置信度 PPI 数据（String-50）
    来源: https://string-db.org

    Args:
        score_threshold: 置信度阈值（乘以1000），500 = 0.5
    """
    print("\n" + "=" * 60)
    print(f"下载 STRING 高置信度 PPI 数据 (score >= {score_threshold/1000})")
    print("=" * 60)

    # STRING 批量下载链接（人类，combined score >= 阈值）
    url = (
        f"https://stringdb-downloads.org/download/"
        f"protein.links.detailed.v12.0/"
        f"9606.protein.links.detailed.v12.0.txt.gz"
    )
    save_path = os.path.join(DATA_DIR, "9606.protein.links.detailed.txt.gz")
    download_file(url=url, save_path=save_path, desc="STRING human PPI (full)")

    # 解压并过滤
    import gzip
    import shutil

    txt_path = os.path.join(DATA_DIR, f"STRING_score{score_threshold}.tsv")
    if not os.path.exists(txt_path):
        print(f"[过滤] 提取 combined_score >= {score_threshold} 的交互...")
        count = 0
        with gzip.open(save_path, 'rt') as fin, open(txt_path, 'w') as fout:
            header = fin.readline()
            fout.write(header)
            for line in fin:
                cols = line.strip().split()
                # 最后一列是 combined_score
                combined_score = int(cols[-1])
                if combined_score >= score_threshold:
                    fout.write(line)
                    count += 1
        print(f"[完成] 过滤后保留 {count} 条交互")


if __name__ == "__main__":
    print("L3-PPI 数据下载脚本")
    print("请确保网络连接正常...\n")

    # 1. 下载 HuRI 数据（预训练用）
    download_huri_data()

    # 2. 下载 SHS 数据（评测基准用）
    download_shs_data()

    # 3. 下载 STRING 数据（String-50 评测基准用）
    download_string_data(score_threshold=500)

    print("\n" + "=" * 60)
    print("所有数据下载完成！")
    print(f"数据目录: {DATA_DIR}")
    print("=" * 60)
import gzip
import re
import os

# 配置
FASTA_FILE = "data/raw/Homo_sapiens.GRCh38.pep.all.fa.gz"   # 下载的文件
GENE_IDS_FILE = "protein_ids.txt"                  # 你的基因 ID 列表
OUTPUT_TSV = "data/raw/HI-II-14_sequences.tsv"

# 读取目标基因 ID 集合
with open(GENE_IDS_FILE) as f:
    target_ids = set(line.strip() for line in f if line.strip())

print(f"目标基因数: {len(target_ids)}")

# 正则提取 gene:ENSG...
gene_pattern = re.compile(r'gene:(ENSG\d+)')

# 解析 FASTA
current_gene = None
current_seq = []
found = 0
os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)

with gzip.open(FASTA_FILE, 'rt') as fin, open(OUTPUT_TSV, 'w') as fout:
    for line in fin:
        line = line.strip()
        if line.startswith('>'):
            # 处理上一条记录
            if current_gene in target_ids:
                fout.write(f"{current_gene}\t{''.join(current_seq)}\n")
                found += 1
            # 提取新记录的基因 ID
            match = gene_pattern.search(line)
            if match:
                current_gene = match.group(1)
            else:
                current_gene = None
            current_seq = []
        else:
            if current_gene is not None:   # 仅当基因属于目标集合时才保存序列，但此处无法提前判断，可以先全部保存后过滤？
                # 优化：只保留目标基因的序列，避免内存占用
                if current_gene in target_ids:
                    current_seq.append(line)
                else:
                    current_seq = []   # 如果不是目标基因，跳过该条记录的所有序列行
            else:
                current_seq = []
    # 处理最后一条
    if current_gene in target_ids:
        fout.write(f"{current_gene}\t{''.join(current_seq)}\n")
        found += 1

print(f"成功提取 {found}/{len(target_ids)} 条序列")
print(f"输出文件: {OUTPUT_TSV}")
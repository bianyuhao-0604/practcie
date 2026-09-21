import sys, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, FEATURES, MAX_LEN

MODEL = "facebook/esm2_t12_35M_UR50D"   # 35M 参数 / 480 维<span data-allow-html class='source-item source-aggregated' data-group-key='source-group-4' data-url='https://ai&#46;azure&#46;com' data-id='turn0search6'><span data-allow-html class='source-item-num' data-group-key='source-group-4' data-id='turn0search6' data-url='https://ai&#46;azure&#46;com'><span class='source-item-num-name' data-allow-html>azure.com</span><span data-allow-html class='source-item-num-count'></span></span></span>
OUT = FEATURES / "esm2"; OUT.mkdir(parents=True, exist_ok=True)

def main():
    ids = (DATA / "proteins.txt").read_text().split()
    seqs = {}
    for line in open(DATA / "sequences.fasta"):
        if line.startswith(">"): pid = line[1:].strip(); seqs[pid] = ""
        else: seqs[pid] += line.strip()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda().eval()
    if torch.cuda.is_bf16_supported(): model = model.to(torch.bfloat16)
    else: model = model.half()
    print(f"{MODEL} 加载完成, VRAM 峰值: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    order = sorted(range(len(ids)), key=lambda i: len(seqs[ids[i]]))
    B = 16
    for s in range(0, len(order), B):
        idxs = order[s:s+B]
        batch_ids = [ids[i] for i in idxs]
        if all((OUT / f"{p}.npy").exists() for p in batch_ids): continue
        batch_seq = [seqs[p][:MAX_LEN] for p in batch_ids]
        enc = tok(batch_seq, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = model(**enc).last_hidden_state          # [B, L+2, 480]
        out = out.float().cpu().numpy()
        for j, p in enumerate(batch_ids):
            emb = out[j, 1:len(batch_seq[j]) + 1]          # ★ 去 <cls>/<eos>
            np.save(OUT / f"{p}.npy", emb.astype(np.float16))
    print("ESM-2 特征完成:", len(list(OUT.glob('*.npy'))))

if __name__ == "__main__": main()

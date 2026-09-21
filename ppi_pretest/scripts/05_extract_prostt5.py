import sys, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, T5EncoderModel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, FEATURES, MAX_LEN

# 直接指向本地 snapshot：safetensors 加载绕过 torch.load/CVE 检查，不联网
MODEL = r"C:\Users\yuhao.bian\.cache\huggingface\hub\models--Rostlab--ProstT5\snapshots\af1c8468b4d20bd4e5ea99babdaab08dc941b791"
OUT = FEATURES / "prostt5"; OUT.mkdir(parents=True, exist_ok=True)

def main():
    ids = (DATA / "proteins.txt").read_text().split()
    seqs, cur = {}, None
    for line in open(DATA / "sequences.fasta", encoding="utf-8"):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]; seqs[cur] = ""
        elif cur is not None:
            seqs[cur] += line

    print(f"从本地加载 {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)
    # 前缀必须是单个 token，否则切片错位
    n_prefix = len(tok("<AA2fold>", add_special_tokens=False).input_ids)
    assert n_prefix == 1, f"<AA2fold> 被拆成 {n_prefix} 个 token，请检查 added_tokens.json 是否拷全"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = T5EncoderModel.from_pretrained(MODEL, torch_dtype=dtype)
    except TypeError:                     # 兼容参数名变更
        model = T5EncoderModel.from_pretrained(MODEL).to(dtype)
    model = model.cuda().eval()
    print(f"  精度: {dtype}, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    order = sorted(ids, key=lambda p: len(seqs[p]))   # 长度排序，batch 内长度相近
    B = 2
    for s in range(0, len(order), B):
        batch_ids = order[s:s+B]
        if all((OUT / f"{p}.npy").exists() for p in batch_ids):
            continue
        batch_seq = [seqs[p][:MAX_LEN] for p in batch_ids]
        texts = ["<AA2fold> " + " ".join(sq) for sq in batch_seq]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN + 8)
        enc = {k: v.cuda() for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        out = out.float().cpu().numpy()
        for j, p in enumerate(batch_ids):
            L = len(batch_seq[j])
            emb = out[j, 1:1+L]           # 跳过 <AA2fold> 前缀，止于 </s>
            np.save(OUT / f"{p}.npy", emb.astype(np.float16))
        if s % 200 == 0:
            print(f"  进度 {s}/{len(ids)} ({100*s/len(ids):.0f}%)  嵌入维度: {out.shape[-1]}")

    n = len(list(OUT.glob("*.npy")))
    print(f"\nProstT5 特征完成: {n}")
    if n:
        d = np.load(next(OUT.glob("*.npy"))).shape[-1]
        print(f"!! 实际嵌入维度 = {d}。若与 configs.PROSTT5_DIM 不一致，"
              f"请把 configs.py 中的 PROSTT5_DIM 改为 {d}")

if __name__ == "__main__":
    main()

"""ESM-2 t33 / ProstT5 提取。N7：--bs 按模型分级默认（esm2=3, prostt5=2, 8GB 显存）。"""
import argparse, os, sys
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LEN, FeatureModels, Paths

def read_seqs(pids, fasta):
    want = set(pids)
    out, name, buf = {}, None, []
    for line in open(fasta):
        line = line.strip()
        if line.startswith(">"):
            if name in want:
                out[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name in want:
        out[name] = "".join(buf)
    missing = want - set(out)
    assert not missing, f"fasta 缺 {len(missing)} 条, e.g. {sorted(missing)[:3]}"
    return out

def extract_esm2(pids, fasta, feat_dir, fm, bs):
    import esm
    seqs = read_seqs(pids, fasta)
    model, alphabet = esm.pretrained.__dict__[fm.esm2]()
    model = model.eval().cuda()
    bc = alphabet.get_batch_converter()
    order = sorted(pids, key=lambda p: len(seqs[p]))
    for k in range(0, len(order), bs):
        chunk = [(p, seqs[p][:MAX_LEN]) for p in order[k:k + bs]]
        _, _, toks = bc(chunk)
        toks = toks.cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = model(toks, repr_layers=[model.num_layers])
        h = out["representations"][model.num_layers].float()          # [B, L+2, 1280]
        for j, (p, s) in enumerate(chunk):
            L = len(s)
            emb = h[j, 1:1 + L].cpu().numpy()                          # off-by-one 修复点
            assert emb.shape == (L, 1280)
            np.save(os.path.join(feat_dir, f"seq_{p}.npy"), emb.astype(np.float16))
        print(f"esm2 {min(k + bs, len(order))}/{len(order)}")

def extract_prostt5(pids, fasta, feat_dir, fm, bs):
    from transformers import T5EncoderModel, T5Tokenizer
    seqs = read_seqs(pids, fasta)
    tok = T5Tokenizer.from_pretrained(fm.prostt5)
    enc = T5EncoderModel.from_pretrained(fm.prostt5).eval().cuda()
    off = len(tok("<aa2fold>", add_special_tokens=False)["input_ids"])
    order = sorted(pids, key=lambda p: len(seqs[p]))
    for k in range(0, len(order), bs):
        chunk = order[k:k + bs]
        enc_in = tok([f"<aa2fold> {' '.join(seqs[p][:MAX_LEN])}" for p in chunk],
                     return_tensors="pt", padding=True)
        with torch.no_grad():
            h = enc(input_ids=enc_in["input_ids"].cuda(),
                    attention_mask=enc_in["attention_mask"].cuda()).last_hidden_state.float()
        for j, p in enumerate(chunk):
            L = min(len(seqs[p]), MAX_LEN)
            total = int(enc_in["attention_mask"][j].sum())
            assert total == off + L + 1, f"ProstT5 对齐失败: {p} {total} vs {off + L + 1}"
            emb = h[j, off:off + L].cpu().numpy()
            np.save(os.path.join(feat_dir, f"str_{p}.npy"), emb.astype(np.float16))
        print(f"prostt5 {min(k + bs, len(order))}/{len(order)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["seq", "str"], required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--bs", type=int, default=None,
                    help="批大小；默认 esm2=3 / prostt5=2（8GB）。24GB+ 可设 8/4")
    a = ap.parse_args()
    feat_dir = a.feat_dir or Paths().feat_dir
    os.makedirs(feat_dir, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    bs = a.bs or (3 if a.kind == "seq" else 2)
    (extract_esm2 if a.kind == "seq" else extract_prostt5)(pids, a.fasta, feat_dir, FeatureModels(), bs)

"""text (128d) 与 genome (256d)。P8：text PCA 仅 train pids fit（--train-pids 必填）。"""
import argparse, os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Paths, FeatureModels

def uniprot_text(pid, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, pid + ".txt")
    if not os.path.exists(p):
        import urllib.request
        urllib.request.urlretrieve(f"https://rest.uniprot.org/uniprotkb/{pid}.txt", p)
    keep, sec = [], None
    for line in open(p):
        if line.startswith("CC   -!- "):
            sec = line[8:].strip()
        elif line.startswith("CC       ") and sec:
            keep.append(line[9:].strip())
        elif not line.startswith("CC   ") and sec is not None:
            sec = None
        if line.startswith("DR   GO;") or line.startswith("KW   "):
            keep.append(line.strip())
    return " ".join(keep)[:6000]

def extract_text(pids, feat_dir, fm, provider, cache_dir, train_pids=None):
    assert train_pids, "P8: 必须提供 --train-pids（PCA 仅在 train 蛋白上 fit，避免 transductive 泄漏）"
    texts = {p: uniprot_text(p, cache_dir) for p in pids}
    if provider == "proteinclip":
        assert fm.proteinclip_ckpt, "ProteinCLIP ckpt 未配置"
        raise NotImplementedError("ckpt 可得时按其仓库接口编码到 joint 空间 128d（接入点在此）")
    if provider == "openai":
        from openai import OpenAI
        cli = OpenAI()
        emb = {}
        items = list(texts.items())
        for k in range(0, len(items), 64):
            batch = [t[:8000] for _, t in items[k:k + 64]]
            r = cli.embeddings.create(input=batch, model=fm.openai_model)
            for (p, _), e in zip(items[k:k + 64], r.data):
                emb[p] = np.array(e.embedding, np.float32)
            print(f"openai {min(k + 64, len(items))}/{len(items)}")
    elif provider == "st_local":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(fm.st_local)
        emb = {p: m.encode(t, convert_to_numpy=True) for p, t in texts.items()}
    else:
        raise ValueError(provider)
    from sklearn.decomposition import PCA
    import joblib
    tr_set = set(l.strip() for l in open(train_pids) if l.strip())
    fit_pids = [p for p in pids if p in tr_set]
    assert fit_pids, "--train-pids 与 --pids 无交集"
    pca = PCA(n_components=128, random_state=0).fit(np.stack([emb[p] for p in fit_pids]))
    joblib.dump(pca, os.path.join(feat_dir, "text_pca.joblib"))
    for p in pids:
        np.save(os.path.join(feat_dir, f"text_{p}.npy"),
                pca.transform(emb[p][None])[0].astype(np.float16))
    print(f"P8 ok: PCA fit on {len(fit_pids)} train pids, transformed {len(pids)}")

def extract_genome(pids, feat_dir, provider, precomp_dir):
    if provider == "precomputed":
        assert precomp_dir, "--precomp-dir 必填（含 {pid}.genome.npy, 256d）"
        for p in pids:
            v = np.load(os.path.join(precomp_dir, p + ".genome.npy")).astype(np.float16)
            assert v.shape == (256,), f"genome 维度需 256: {p} got {v.shape}"
            np.save(os.path.join(feat_dir, f"genome_{p}.npy"), v)
    elif provider == "glm2":
        raise NotImplementedError("gLM2 接入点：按 HF model card 构造 interleave 基因组窗口 "
                                  "(GFF+fasta, ±邻基因上下文) → last_hidden mean-pool → 256d")
    elif provider == "zero":
        for p in pids:
            np.save(os.path.join(feat_dir, f"genome_{p}.npy"), np.zeros(256, np.float16))
        print("WARNING: genome=zero 占位。run_all 的 P10 非平凡性检查会拦截 switch_genome=1。")
    else:
        raise ValueError(provider)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["text", "genome"], required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--text-provider", default="openai")
    ap.add_argument("--genome-provider", default="precomputed")
    ap.add_argument("--precomp-dir", default="")
    ap.add_argument("--train-pids", default=None)
    ap.add_argument("--uniprot-cache", default="./workdir/uniprot")
    a = ap.parse_args()
    feat_dir = a.feat_dir or Paths().feat_dir
    os.makedirs(feat_dir, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    if a.kind == "text":
        extract_text(pids, feat_dir, FeatureModels(), a.text_provider, a.uniprot_cache, a.train_pids)
    else:
        extract_genome(pids, feat_dir, a.genome_provider, a.precomp_dir)

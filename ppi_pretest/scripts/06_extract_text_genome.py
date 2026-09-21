import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import DATA, FEATURES, GENOME_DIM

OUT_T = FEATURES / "text"; OUT_G = FEATURES / "genome"
OUT_T.mkdir(parents=True, exist_ok=True); OUT_G.mkdir(parents=True, exist_ok=True)

def main():
    cache = np.load(DATA / "uniprot_cache.npz", allow_pickle=True)
    ids = json.loads(str(cache["ids"]))
    text_map = json.loads(str(cache["text"]))
    genome_map = json.loads(str(cache["genome"]))
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2")           # CPU 即可, 384 维
    todo = [p for p in ids if not (OUT_T / f"{p}.npy").exists()]
    print(f"text 待编码 {len(todo)}")
    for p in todo:
        t = text_map.get(p, "") or "no annotation available"
        v = st.encode(t, normalize_embeddings=True).astype(np.float16)
        np.save(OUT_T / f"{p}.npy", v)
        g = np.asarray(genome_map.get(p, [0]*GENOME_DIM), dtype=np.float16)
        np.save(OUT_G / f"{p}.npy", g)
    print("text/genome 特征完成")

if __name__ == "__main__": main()

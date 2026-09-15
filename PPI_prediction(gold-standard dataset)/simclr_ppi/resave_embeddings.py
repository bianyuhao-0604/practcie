"""Stage 2: 全部蛋白过最优 Adapter → SIMCLR_NPZ（下游零改动换嵌入目录）"""
import numpy as np, torch
from config import PREP_NPZ, SIMCLR_NPZ, ADAPTER_CKPT
from simclr_model import Adapter

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = np.load(PREP_NPZ)
    ids, X = prep["ids"], prep["X"]
    adapter = Adapter(spec_norm=True).to(dev)
    adapter.load_state_dict(torch.load(ADAPTER_CKPT, map_location=dev))
    adapter.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            outs.append(adapter(torch.from_numpy(X[i:i+8192]).to(dev)).cpu().numpy())
    H = np.concatenate(outs).astype(np.float32)
    np.savez(SIMCLR_NPZ, ids=ids, X=H)
    print(f"重塑嵌入 {H.shape} → {SIMCLR_NPZ}")

if __name__ == "__main__":
    main()

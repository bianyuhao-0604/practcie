"""Global configuration for L3-PPI.

All tunables live here so that data / pre-training / fine-tuning / test
entry points share exactly the same settings.
"""
from pathlib import Path
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(os.environ.get("L3PPI_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("L3PPI_DATA", ROOT / "data" / "raw"))
CACHE_ROOT = ROOT / "data" / "processed"
CKPT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "logs"
for p in (DATA_ROOT, CACHE_ROOT, CKPT_ROOT, LOG_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class DataCfg:
    # one of: yeast | shs27k | string
    name = "yeast"
    # negative sampling: "random" = pair random non-interacting proteins
    neg_sampling = "random"
    train_ratio = 0.7
    val_ratio = 0.1
    # graph split strategy: random | bfs | dfs
    split = "random"
    # cap number of negative pairs (= pos for class balance)
    balance_neg = True


# ----------------------------------------------------------------------
# Protein encoder (ESM-2 by default; falls back to a learned CNN stub)
# ----------------------------------------------------------------------
class EncoderCfg:
    # if True and fair-esm is available, use esm2_t6_8M_UR50D (= 8M) or
    # esm2_t12_35M_UR50D (= 35M). Set to False to use the CNN baseline.
    use_esm = True
    esm_model = "esm2_t6_8M_UR50D"      # tiny model, friendly to CPU
    esm_pool = "mean"                     # mean | cls
    freeze = True                         # freeze PLM, train only prompt head
    # fallback CNN (amino-acid one-hot -> 1D conv -> global max)
    aa_vocab = 20
    cnn_channels = 128
    cnn_kernels = (3, 5, 7)
    node_dim = 64


# ----------------------------------------------------------------------
# L3-PPI prompt / graph
# ----------------------------------------------------------------------
class PromptCfg:
    K = 8                     # number of candidate L3 paths
    num_prompt_nodes = None   # auto = K + 1 (central node + K branch nodes)

    @property
    def n_nodes(self):
        return (self.K + 1)

    # initial (learnable) node feature dimension
    node_dim = 64
    edge_dim = 1
    # gating
    tau = 1.0                 # Gumbel temperature (annealed during tuning)
    gamma = 3.0               # target #active paths: + -> ~K/gamma? see ℒ_PN
    hard_gumbel_eval = True   # hard sampling at inference
    # GIN
    gin_hidden = 128
    gin_num_layers = 3
    gin_dropout = 0.1


# ----------------------------------------------------------------------
# Pre-training (L3 pattern recognition surrogate)
# ----------------------------------------------------------------------
class PreTrainCfg:
    epochs = 20
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-5
    # surrogate sees synthetic L3 pattern graphs
    n_pos_per_protein = 4     # positive patterns per graph
    n_neg_per_protein = 4     # negative (random) patterns
    gnn = "gin"               # gin | gcn
    save_name = "surrogate.pt"


# ----------------------------------------------------------------------
# Fine-tuning (prompt tuning on top of frozen surrogate + encoder)
# ----------------------------------------------------------------------
class FineTuneCfg:
    epochs = 30
    batch_size = 32
    lr = 5e-4                 # only prompt params (+ gate) are trained
    gate_lr = 1e-3
    weight_decay = 1e-5
    cls_weight = 1.0          # BCE weight for PPI classification
    reg_weight = 0.5           # ℒ_PN weight
    tau_start = 1.0
    tau_end = 0.2
    tau_anneal_epochs = 20
    early_stop = 5
    save_name = "l3ppi.pt"


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------
class RunCfg:
    seed = 42
    device = "auto"           # auto | cpu | cuda
    num_workers = 0
    log_interval = 20
    use_amp = False


# expose a bundle for convenience
def get_cfg():
    return dict(data=DataCfg, encoder=EncoderCfg, prompt=PromptCfg,
                pretrain=PreTrainCfg, finetune=FineTuneCfg, run=RunCfg)

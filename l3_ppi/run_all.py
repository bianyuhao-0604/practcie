"""End-to-end smoke test: data -> pre-train -> fine-tune -> test -> predict.

Runs entirely on CPU with the synthetic fallback dataset and a tiny CNN
encoder, so it can be executed in an offline / CI environment.  Set
``L3PPI_ALLOW_SYNTH=1`` (default) so no downloaded data is required.

    python run_all.py
"""
import os, sys
os.environ["L3PPI_ALLOW_SYNTH"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import config
from config import (DataCfg, EncoderCfg, PromptCfg, PreTrainCfg, FineTuneCfg,RunCfg,
                    CKPT_ROOT, LOG_ROOT, CACHE_ROOT)
import utils

# ---- shrink config for fast CPU demo ----
DataCfg.name = "yeast_demo"
DataCfg.split = "random"
EncoderCfg.use_esm = False
EncoderCfg.node_dim = 32
PromptCfg.K = 4
PromptCfg.node_dim = 32
PromptCfg.gin_hidden = 32
PromptCfg.gin_num_layers = 2
PreTrainCfg.epochs = 3
PreTrainCfg.batch_size = 32
FineTuneCfg.epochs = 4
FineTuneCfg.batch_size = 32
FineTuneCfg.tau_anneal_epochs = 2
FineTuneCfg.early_stop = 10
FineTuneCfg.reg_weight = 0.5
RunCfg.num_workers = 0
utils.set_seed(RunCfg.seed)

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)

log = utils.Logger(LOG_ROOT / "run_all.log")
log.header("L3-PPI end-to-end demo")


def sh(cmd):
    log.log(f"\n$ {cmd}")
    import subprocess
    r = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
    if r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")


sh("python -c \"import config, utils, dataset, encoder, model, gnn; print('imports OK')\"")

# 1. L3 rule analysis (synthetic)
sh("python -c \"from dataset import l3_rule_analysis; l3_rule_analysis()\"")

# 2. pre-train surrogate
sh("python pretrain.py --epochs 3 --batch-size 32")

# 3. fine-tune (no ESM, CNN baseline)
sh("python finetune.py --encoder cnn --epochs 4 --batch-size 32 --no-pretrain")

# 4. re-run with surrogate pre-training (full method, short)
sh("python finetune.py --encoder cnn --epochs 4 --batch-size 32 --eval-only")

# 5. test + ablation + L3 figure
sh("python test.py --encoder cnn --run-ablation --l3-analysis")

# 6. external prediction: generate a tiny query FASTA on the fly
import textwrap
fa = CACHE_ROOT / "queries.fasta"
fa.write_text(textwrap.dedent("""\
    >Q1
    MKTAYIAKQRQISFVKSHFSRYLGLASRLFGQSLQGQAKA
    >Q2
    MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTY
    >Q3
    MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRS
    """))
sh(f"python predict.py --fasta {fa} --out {CACHE_ROOT / 'preds.tsv'} --encoder cnn")

log.log("\nALL STAGES COMPLETED SUCCESSFULLY")

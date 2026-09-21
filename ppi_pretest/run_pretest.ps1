# ============ ppi_pretest 一键运行 ============
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
# 国内网络如 HuggingFace 下载失败，取消下行注释：
# $env:HF_ENDPOINT = "https://hf-mirror.com"

# --- 自动探测 torch_final venv（不依赖 Activate，规避执行策略）---
$candidates = @(".\torch_final", ".\venv\torch_final",
                "$PWD\torch_final", "$env:USERPROFILE\venv\torch_final")
$PY = $null
foreach ($c in $candidates) {
    if (Test-Path (Join-Path $c "Scripts\python.exe")) { $PY = Join-Path $c "Scripts\python.exe"; break }
}
if (-not $PY) { Write-Host "!! 未找到 torch_final\Scripts\python.exe，请手动改 \$PY"; exit 1 }
Write-Host "使用解释器: $PY"
& $PY --version

# --- Step 0 依赖自检与安装 ---
& $PY -m pip install -q transformers sentencepiece scikit-learn pandas tqdm sentence-transformers
& $PY scripts\00_check_env.py; if ($LASTEXITCODE -ne 0) { exit 1 }

# --- Step 1-2 数据 ---
& $PY scripts\01_download_data.py
& $PY scripts\02_parse_and_sample.py; if ($LASTEXITCODE -ne 0) { exit 1 }
& $PY scripts\03_fetch_uniprot.py

# --- Step 3 特征提取（合计约 2h）---
& $PY scripts\04_extract_esm2.py
& $PY scripts\05_extract_prostt5.py
& $PY scripts\06_extract_text_genome.py
& $PY scripts\07_verify_features.py; if ($LASTEXITCODE -ne 0) { exit 1 }

# --- Step 4 训练矩阵（6 runs ≈ 4-5h）---
foreach ($s in 42, 1234) {
    & $PY scripts\08_train.py --r-stage R1 --ablation none --seed $s
    & $PY scripts\08_train.py --r-stage R7 --ablation none --seed $s
    & $PY scripts\08_train.py --r-stage R7 --ablation no_global --seed $s
}

# --- Step 5 汇总 ---
& $PY scripts\09_evaluate_all.py
Write-Host "`n预实验完成，查看 results\all_metrics.csv"

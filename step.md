# GRPO_TES 環境構築手順（成功した手順のメモ）

2025-10-18 時点でクラウド GPU (NVIDIA H100) 上で確認済みの Python 仮想環境セットアップ手順です。公式ノートブック（`gpt_oss_(20B)_GRPO.ipynb`）の依存と整合するように構成しています。

---

## 1. 仮想環境の作成と基本ツール更新

```bash
mkdir -p ~/workspace && cd ~/workspace
git --version || (sudo apt-get update && sudo apt-get install -y git)
```

```bash
git clone https://github.com/takumi0211/GRPO_TES.git
cd GRPO_TES
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

## 2. PyTorch / TorchVision のインストール

Unsloth が内部で利用する `torchao` は `torch.int1` を前提にしており、`torch>=2.9.0` が必要です。  
CUDA ランタイムは依存として自動取得されるため PyPI 本家からで問題ありません。

```bash
python -m pip install --no-cache-dir \
    "torch==2.9.0" \
    "torchvision==0.24.0"
```

## 3. Transformers / TRL / Tokenizers の固定

公式デモに合わせて、以下 3 つを明示的に再インストールします。  
（`trl==0.22.2` は `transformers==4.56.2` と組み合わせる前提で動作確認済み）

```bash
python -m pip install --no-cache-dir --force-reinstall \
    "transformers==4.56.2" \
    "trl==0.22.2" \
    "tokenizers==0.22.0"
```

## 4. 残りの依存をまとめて導入

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` には Unsloth/Unsloth-Zoo の Git 依存を含むため、ここでビルドが入ります。  
（この時点で `bitsandbytes==0.48.1`, `matplotlib==3.10.7` などが入ることを確認済み）

## 5. 学習実行

```bash
python train_grpo.py
```

進捗は別ターミナルで `python watch_metrics.py --metrics-csv metrics.csv` を回すとリアルタイム表示できます。

## 6. 新規 CUDA 仮想環境フルセットアップ（2025-10-20）

仮想環境を完全に作り直したケース向けの再構築手順です。  
（既存の `.venv` を消す場合は `rm -rf .venv` を実行してから以下に進んでください。）

```bash
cd ~/workspace/GRPO_TES
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

CUDA 12.x 対応の PyTorch / TorchVision / Torchaudio をインストールします。ホイールはドライバ 570.148.08 (CUDA 12.8) で動作確認済みです。

```bash
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
```

Unsloth デモと整合するバージョンを改めて導入します。

```bash
python -m pip install --no-cache-dir --force-reinstall \
    "transformers==4.56.2" \
    "trl==0.22.2" \
    "tokenizers==0.22.0"
```

残りの依存関係をまとめて入れ直します。

```bash
python -m pip install -r requirements.txt
```

必要に応じて Unsloth 本体をアップデートします。

```bash
python -m pip install --upgrade unsloth
```

CUDA を利用できるか確認します。

```bash
python - <<'PYCODE'
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
PYCODE
```

テストスクリプトで GPU を明示的に指定して動作確認します。

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

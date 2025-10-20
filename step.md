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

## 2. PyTorch / TorchVision / TorchAudio のインストール

Unsloth 2025.10 系は `torch>=2.2,<2.6` を前提とし、2.6 以降では拡張モジュールをロードできません。  
CUDA 12.1 (H100) 環境で成功した組み合わせは以下の通りです。別バージョンの CUDA イメージを使う場合は [`docs.unsloth.ai`](https://docs.unsloth.ai/get-started/install-update/pip-install?utm_source=openai) の表に合わせて `cu118`/`cu124` などへ置き換えてください。

```bash
python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    "torchao==0.13.0"
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
インストールログで `Skipping import of cpp extensions due to incompatible torch version` が出た場合は Step 2 のバージョン指定を再確認してください。  
不安な場合は Unsloth の自動診断をその場で実行すると、環境に合わせた推奨コマンドが表示されます。

```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

## 5. 学習実行

```bash
python train_grpo.py
```

進捗は別ターミナルで `python watch_metrics.py --metrics-csv metrics.csv` を回すとリアルタイム表示できます。

# GRPO_TES 環境構築手順（成功した手順のメモ）

2025-10-18 時点でクラウド GPU (NVIDIA H100) 上で確認済みの Python 仮想環境セットアップ手順です。公式ノートブック（`gpt_oss_(20B)_GRPO.ipynb`）の依存と整合するように構成しています。

---

## 1. 仮想環境の作成と基本ツール更新

```bash
mkdir -p ~/workspace && cd ~/workspace
git --version || (sudo apt-get update && sudo apt-get install -y git)
'''

'''bash
git clone https://github.com/takumi0211/GRPO_TES.git
cd GRPO_TES
'''

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

## 5. インポート確認

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "has int1?", hasattr(torch, "int1"))
from transformers import PreTrainedModel, TrainingArguments
from trl import GRPOConfig, GRPOTrainer
print("transformers / trl OK")
PY
```

問題がなければ準備完了です。

## 6. 学習実行

```bash
python train_grpo.py \
  --metrics-csv metrics.csv \
  --output-dir outputs
```

進捗は別ターミナルで `python watch_metrics.py --metrics-csv metrics.csv` を回すとリアルタイム表示できます。

---

### メモ
- Unsloth からの警告に合わせ、`train_grpo.py` 冒頭で `import unsloth` を最初に呼ぶよう修正済み。
- `generation_batch_size` / `steps_per_generation` はスクリプト内で整合を取るようにしてあるため、CLI 側で特別な指定は不要。
- `requirements.txt` には `torch`/`torchvision` を含めていない（または入れる場合は `==2.9.0` 等で固定）ことで、再インストール時に 2.5 系へ戻されないようにしている。

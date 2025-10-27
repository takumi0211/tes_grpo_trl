# train_grpo.py セットアップ手順（クラウドGPU・ターミナル専用）

更新日: 2025-10-27（この日時点での互換構成）

本手順は、このリポジトリの `train_grpo.py` を単一GPUマシン上で実行するための最小セットアップです。GUIは不要で、ターミナルのみを前提にしています。

---

## 対応バージョンまとめ（検証済み）

- Python: 3.11 または 3.12
- NVIDIA Driver: 550+ を推奨（CUDA 12.8 系ホイールに対応）
- PyTorch: 2.8.0（CUDA 12.8ビルド）
- Transformers: 4.57.1 以上（`Mxfp4Config` 利用のため）
- TRL (GRPO): 0.23.1（vLLMコロケート対応の安定版）
- vLLM: 0.10.2（colocate利用; GPT‑OSS用の 0.10.1+gptoss でも可）
- PEFT: 0.17.1 以上（`target_modules="all-linear"`, `target_parameters` を使用）
- 追加: `kernels`（TransformersのMXFP4カーネル依存）, `datasets`, `pandas`, `accelerate`

注:
- 本スクリプトは `Mxfp4Config(dequantize=True)` を用いて MXFP4 から BF16 にデクオンした上で LoRA 学習します。MXFP4 での後方伝播カーネルは現状不要です。
- `openai/gpt-oss-20b` の推論は vLLM をコロケート起動で使用します（TRLが内部で起動・停止）。

---

## 0. システム前提の確認（5分）

以下は Ubuntu 22.04/24.04 相当を想定しています。

```
# GPUとドライバの確認
nvidia-smi

# 便利ツール（入っていなければ）
sudo apt-get update -y
sudo apt-get install -y git git-lfs curl build-essential python3-distutils
git lfs install
```

表示例: ドライバ 550.x 以上、CUDA 12.x ランタイムが読み取れればOKです。

---

## 1. Python 環境の用意（uv 推奨, 3分）

uv は高速なパッケージ管理ツールです。未導入ならインストールします。

```
# uv インストール（ユーザ領域）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 仮想環境（Python 3.12推奨）
uv venv --python 3.12
source .venv/bin/activate

# pip/uv 自体を更新
uv pip install -U pip wheel setuptools
```

（標準の venv/pip を使いたい場合は `python3 -m venv .venv && source .venv/bin/activate` でもOK）

---

## 2. 依存関係のインストール（10分）

vLLM 0.10.2 の公式ホイールは、対応する PyTorch/cuDNN を自動で解決できます。まず vLLM を入れてから、NLP関連を追加で入れると依存競合が起きにくいです。

```
# vLLM 0.10.2（CUDA 12.8向け公式ホイールレジストリ）
uv pip install "vllm==0.10.2" \
  --extra-index-url https://wheels.vllm.ai/0.10.2/ \
  --config-settings vllm:torch-backend=auto

# NLP/学習系ライブラリ
uv pip install \
  "transformers>=4.57.1" \
  "trl==0.23.1" \
  "peft>=0.17.1" \
  "accelerate>=1.10.0" \
  datasets pandas \
  kernels

# もし torch が未導入の場合（vLLM 同梱解決が失敗した環境向け）
# CUDA 12.8 の公式ホイールからインストール
# uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchvision==0.19.0" "torchaudio==2.8.0"
```

補足:
- `kernels` は Transformers の MXFP4 実装が参照する Triton カーネル（`kernels-community/triton_kernels`）のパッケージ名です。
- 既存の CUDA Toolkit のインストールは不要です（ホイールにバイナリ同梱）。必要なのは十分に新しい NVIDIA Driver です。

---

## 3. リポジトリの取得（1分）

```
git clone https://github.com/takumi0211/tes_grpo_trl.git -b main
cd tes_grpo_trl
```

（既存ワークツリーがある場合はこの手順は不要）

---

## 4. データの前処理（Harmony 変換, 1–3分）

`data/*.csv` がある前提です。プロンプト列を Harmony 形式に変換します。

```
python convert_to_harmony.py data --overwrite
```

出力は `*_harmony.csv` として `data/` 内に生成されます。`data_reward.py` は Harmony 形式を優先して自動検出します。

---

## 5. 実行（学習の開始）

```
# 省メモリ・ノイズ抑制の環境変数（任意）
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

# 学習を開始
python train_grpo.py
```

学習は以下の構成で行われます:
- `openai/gpt-oss-20b` を MXFP4 ロード → BF16 にデクオンして LoRA 学習
- TRL の GRPOTrainer が vLLM をコロケート起動（`vllm_enable_sleep_mode=True` で VRAM 回収）
- 1 step あたり 12 プロンプト × 各 8 生成（設定値は `train_grpo.py` を参照）

成果物は `runs/grpo_gptoss20b_lora4_tes/` に保存されます（LoRA アダプタとトークナイザ）。

---

## 6. うまくいかない時のチェック

- vRAM 不足: `GRPOConfig.vllm_gpu_memory_utilization` を下げる、`MAX_COMPLETION_LEN` を短くする、`NUM_GENERATIONS` を減らす。
- CUDA/ドライバ不整合: `nvidia-smi` のドライバが古い場合は 550+ へ更新。PyTorch を `cu128` ホイールで再インストール。
- vLLM が起動できない: `pip freeze | grep vllm` で 0.10.2 であること、`torch` が 2.8.0 であることを確認。
- ImportError: `kernels` が未インストールだと MXFP4 のロードで失敗することがあります。`uv pip install kernels` を追加実行。

---

## 7. 再現性メモ

環境差分を最小化したい場合は、導入したバージョンをそのまま `requirements.txt` に固定し、`uv pip compile`/`uv pip sync` 等でロックすることを推奨します。

例:

```
uv pip freeze > requirements.txt
```

---

## 8. 参考（要点）

- GPT‑OSS 20B は MXFP4 での軽量ロードを公式にサポートし、学習時は BF16 へデクオンして LoRA を当てる構成が推奨です。
- TRL の GRPO は vLLM の生成エンジンをコロケート起動でき、`sleep mode` により生成⇄学習の切替で VRAM を解放できます。


# train_grpo.py セットアップ手順（GH200 Grace Hopper向け）

更新日: 2025-11-02

Grace Hopper Superchip (GH200 480GB、CUDA 12.8 + ドライバ 570.148.08) 上で `train_grpo.py` を動かすための構築メモです。GH200 は 64-bit ARM (aarch64) CPU と Hopper GPU を統合しているため、ホイール選択やビルドオプションが x86_64 環境と異なります。

---

## 0. 事前チェック

```
nvidia-smi
uname -m     # aarch64 が表示されるはず
```

* ドライバは 570.148.08 以上 (CUDA 12.8 対応) が必須。
* OS には CUDA 12.8 のランタイムおよび dev パッケージを導入しておく（GH200 bare-metal イメージでは同梱済みのケースが多い）。

必要パッケージ:
```
sudo apt update
sudo apt install -y build-essential git ninja-build cmake python3-dev pkg-config
```

---

## 1. Python 仮想環境（uv 使用例）

```
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/tes_grpo_trl
uv venv --python 3.12
source .venv/bin/activate

uv pip install -U pip wheel setuptools
```

ARM 向け Python を自前でビルドしている場合は `uv venv --system-site-packages` でも構いません。

---

## 2. PyTorch + CUDA (GH200 対応ビルド)

2025-11-02 時点では、`torch==2.8.0+cu128` の安定版ホイールは x86_64 向けのみ提供されており、aarch64 (GH200) では入手できません。そのため、以下いずれかの方法で CUDA 対応の PyTorch を確保します。

### Option A: Nightly/cu128 ホイールを利用（推奨）

PyTorch nightly では aarch64 + CUDA 12.8 ホイールが配布されています。

```
uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
  torch torchvision torchaudio
```

> nightly 版なのでバージョンは `2.9.0.dev*+cu128` のような表記になります。安定版が必要な場合は Option B を検討してください。

### Option B: ソースからビルド

時間はかかりますが、安定版 2.8 系を使いたい場合はソースビルドします。

```
git clone --recursive https://github.com/pytorch/pytorch.git --branch v2.8.0
cd pytorch
pip install -r requirements.txt
USE_CUDA=1 USE_MKLDNN=0 python setup.py bdist_wheel
pip install dist/torch-2.8.0*.whl
```

> Hopper 対応のため `CUDA_VERSION=128`、`TORCH_CUDA_ARCH_LIST="90"` 等の環境変数を設定すると最適化されます。必要に応じて `USE_NCCL=1` も追加。

### Option C: NVIDIA NGC コンテナを使う

コンテナ運用が可能なら、`nvcr.io/nvidia/pytorch:24.10-py3` (CUDA 12.8 + Hopper 対応) をベースに環境を用意するのが最速です。コンテナ内でこのリポジトリをマウントし、本手順の残りを実行します。

---

インストール確認:
```
python - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
PY
```

> `torch.cuda.get_device_name(0)` で `NVIDIA GH200 480GB` が表示されること。

システムの `LD_LIBRARY_PATH` に仮想環境の torch ライブラリを追加する必要がある場合があります:
```
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

---

## 3. vLLM 0.10.2 (CUDA 12.8 GH200 対応ビルド)

GH200 (aarch64) 対応の vLLM ホイールをインストールします。公式レジストリには aarch64 パッケージが含まれています。

```
uv pip install "vllm==0.10.2" \
  --extra-index-url https://wheels.vllm.ai/0.10.2/ \
  --config-settings vllm:torch-backend=auto
```

> `vllm:torch-backend=auto` により、既存の CUDA 版 PyTorch が再利用されます。

---

## 4. NLP / 学習系ライブラリ

```
uv pip install --no-build-isolation \
  "transformers>=4.57.1" \
  "trl==0.23.1" \
  "peft>=0.17.1" \
  "accelerate>=1.10.0" \
  datasets pandas \
  "huggingface_hub>=0.25" \
  packaging ninja \
  "kernels>=0.10" \
  "triton>=3.4"
```

ビルド時の CPU 使用率が高くなる場合は、`MAX_JOBS=8 uv pip install ...` のようにスレッド数を明示すると安定します。

---

## 5. FlashAttention 3 (Hopper / GH200 用)

FA3 は PyPI に正式公開されていないため、Hopper 対応 wheel をダウンロードしてインストールします。以下は例です（PyTorch 2.8.0 + CUDA 12.8 用 aarch64 wheel）:

```
wget https://huggingface.co/datasets/mesolitica/Flash-Attention3-wheel/resolve/main/flash_attn_3-3.0.0b1-cp312-abi3-linux_aarch64-2.8.0-12.8.whl \
  -O flash_attn_3.whl
uv pip install flash_attn_3.whl
```

インポートテスト:
```
python - <<'PY'
import flash_attn_3
print("flash_attn_3 OK")
PY
```

※ FA3 wheel の組み合わせは PyTorch / CUDA バージョンに依存します。環境に合わない場合は wheel を入れ替えるか、`ATTN_IMPL=flash_attention_2` で FA2 へフォールバックしてください。

---

## 6. 確認コマンド

```
python - <<'PY'
import torch, vllm, transformers, trl
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("CUDA OK:", torch.cuda.is_available())
PY
```

```
nvidia-smi dmon -s pucvmt -d 5
```

---

## 7. 学習の実行

```
export ATTN_IMPL=flash_attention_3  # FA3 を使用（未導入ならコメントアウト）
python train_grpo.py
```

ログは `runs/grpo_gptoss20b_lora4_tes/training.log` に出力されます。VRAM 使用量は 97871 MiB を超えないよう、`GRPOConfig` の `num_generations` や `max_completion_length` を調整してください。

---

## 8. トラブルシューティング

* `ImportError: libtorch_cuda.so not found`: CUDA 版 PyTorch が入っていない。手順 2 の再実行で GPU 対応版を入れる。
* `flash_attn_3 seems to be not installed`: FA3 wheel を導入するか、環境変数で FA2/SDPA にフォールバック。
* `vllm` が GH200 対応 wheel を見つけられない: `pip uninstall vllm` → 上記コマンドで再導入。必要なら `--extra-index-url https://pypi.nvidia.com` を併用。
* `Could not build wheels for flash-attn`: `pip install packaging ninja cmake` を済ませ、`--no-build-isolation` を付けてから再実行。CPU リソースを `MAX_JOBS` で制御。

---

## 9. 参考リンク

* NVIDIA GH200 Grace Hopper Superchip Platform Brief
* PyTorch CUDA 12.8 Release Notes
* vLLM Installation Guide (CUDA 12.8)
* FlashAttention 3 Hopper Wheel 配布 (Hugging Face Datasets)

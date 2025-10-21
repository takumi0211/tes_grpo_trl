# Step2: Hugging Face スタックでの GRPO セットアップ手順（Unsloth なし）

2025-10-21 時点で H100 (CUDA 12.4 相当) を想定した手順です。`train_grpo_hf.py` を使って Unsloth 依存なしで GRPO を回すためのセットアップ手順と推奨オプションをまとめています。

---

## 1. 仮想環境の作成

```bash
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

> 既存の Unsloth 用 `.venv` と混ざると依存が競合するので、新しい環境を作るのがおすすめです。

## 2. PyTorch (CUDA 12.4 ビルド) の導入

```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1"
```

- H100 世代では bfloat16 が高速なため、`train_grpo_hf.py` も既定で `--compute-dtype bf16` にしています。
- `pip install flash-attn` の前に PyTorch を入れておくとビルドに必要な CUDA header を拾ってくれます。

## 3. FlashAttention と Triton の導入

```bash
pip install --no-cache-dir --upgrade \
    "triton>=3.0.0" \
    "flash-attn>=3.0.0" --no-build-isolation
```

- H100 では FlashAttention 2/3 が kernel 最適化されているので、`--attn-implementation flash_attention_2` が最も安定です（スクリプトのデフォルト）。
- `--no-build-isolation` を付けると既存の `triton` を共有でき、ビルド時間が短縮されます。

## 4. Hugging Face / TRL スタックのインストール

```bash
pip install --no-cache-dir \
    "accelerate>=1.9.0" \
    "transformers==4.56.2" \
    "trl==0.22.2" \
    "peft>=0.16.0,<0.18" \
    "datasets>=2.19.0" \
    "pandas>=2.2.0" \
    "sentencepiece>=0.2.0" \
    "bitsandbytes>=0.45.0" \
    "matplotlib>=3.9.0"
```

- Unsloth / Unsloth-Zoo は **インストール不要** です。
- 4bit 量子化で学習する場合は `bitsandbytes` を GPU 用 wheel から取得できるように `LD_LIBRARY_PATH` に CUDA 12.4 の lib ディレクトリが入っているか確認してください。

## 5. Accelerate の設定（任意）

単 GPU のみであればこのステップは不要ですが、マルチ GPU や DeepSpeed Zero を使う場合は以下を実行しておくと楽です。

```bash
accelerate config
```

- `mixed_precision` は `bf16` を選択
- `dispatch` は `multi_gpu` または `fsdp` を選択

## 6. 学習データの配置

- `./data` 以下に `*_q_dataset.csv` が存在することを確認します（`train_grpo.py` と同じ形式です）。
- 報酬計算ロジックは `train_grpo.py` と同一なので、既存の CSV をそのまま流用できます。

## 7. `train_grpo_hf.py` の実行例

```bash
python train_grpo_hf.py \
    --output-dir outputs_hf \
    --metrics-csv metrics_hf.csv \
    --generations-csv generations_hf.csv \
    --max-seq-length 5000 \
    --num-generations 4 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --quantization 4bit \
    --attn-implementation flash_attention_2 \
    --device-map auto
    # MoE MLP にも LoRA を適用したい場合は公式レシピ通り target_parameters を指定
    # 例: --peft-target-parameters "7.mlp.experts.gate_up_proj,7.mlp.experts.down_proj"
```

- デフォルトで `--quantization 4bit`, `--compute-dtype bf16`, `--attn-implementation flash_attention_2` を使います。必要に応じて `--quantization none` や `--torch-compile` を指定してください。
- 生成サンプルやメトリクスは `metrics_hf.csv`, `generations_hf.csv` にストリーミングされ、`watch_metrics.py` も併用できます。

### LoRA ターゲット指定について

- GPT-OSS 20B の MLP は Mixture-of-Experts 構造なので、LoRA を当てるには `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` のように **完全なパスを target_parameters で指定**するのが安全です（スクリプトのデフォルトでも `7/15/23` 層の experts を指定済み、公式 Cookbook 推奨に準拠）。
- 実際にヒットしているか確認したいときは、以下のワンライナーを実行すると対象ごとのカウントを表示できます。

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("unsloth/gpt-oss-20b", trust_remote_code=True)
targets = ["q_proj","k_proj","v_proj","o_proj"]
experts = ["mlp.experts.gate_up_proj","mlp.experts.down_proj"]
hits = {t:0 for t in targets}
for name, _ in model.named_modules():
    for t in targets:
        if name.endswith(t):
            hits[t] += 1
print("module hits:", hits)
exp_hits = {e:0 for e in experts}
for name, _ in model.named_parameters():
    for e in experts:
        if e in name:
            exp_hits[e] += 1
print("expert param hits:", exp_hits)
PY
```

## 8. 追加の最適化 Tips

- H100 では `TORCHINDUCTOR_FUSE=1` や `CUDA_DEVICE_MAX_CONNECTIONS=1` を環境変数で設定すると FlashAttention と両立しやすいケースがあります。
- `python -m bitsandbytes` で GPU kernel がリンクできるか簡易診断できます。失敗する場合は CUDA ランタイムバージョンと `bitsandbytes` の対応表を確認してください。
- チェックポイント再開時は `--resume-from-checkpoint latest` が利用できます（`checkpoint-*` ディレクトリから自動検出）。

以上で Unsloth に依存しない GRPO 学習環境の準備は完了です。

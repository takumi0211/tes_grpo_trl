# vLLM Server Mode 学習セットアップ（H100 ×2）

以下は GPT-OSS 20B を vLLM の server mode で高速生成しつつ、既存の逐次バックプロップ構成を維持したまま学習するための手順です。GPU は Hopper 世代 (H100) を 2 台使用し、tensor parallel=2 でモデルを分割します。

## 0. 事前準備
- OS: Ubuntu 22.04 以上（CUDA 12.2+ 推奨）
- GPU: NVIDIA H100 ×2（NVLink 推奨）
- ドライバ: 535 以上
- Python: 3.10–3.12

```bash
python -m venv ~/.venv
source ~/.venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu122
pip install transformers accelerate peft trl datasets vllm==0.5.4.post1
pip install flash-attn --no-build-isolation  # FA3 カーネル
```

## 1. vLLM サーバー起動
学習ノードとは別プロセスでサーバーを立ち上げます（同一マシンで構いません）。Flash Attention 3 対応のコミュニティカーネルを明示します。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --enforce-eager \
  --max-num-seqs 16 \
  --trust-remote-code \
  --attn-impl kernels-community/vllm-flash-attn3 \
  --kv-cache-dtype bf16
```

> **メモ**:  
> - `--enforce-eager` は MXFP4 + FlashAttention3 の初期ウォームアップを安定させます。  
> - `--kv-cache-dtype bf16` は学習側と dtype を揃える設定です。追加で FP8 を使いたい場合は `fp8` に変更しても構いません。  
> - 生成負荷に応じて `--max-num-seqs` や `--max-model-len` を調整してください。

## 2. 学習ノード設定
`train_grpo_vllm.py` は既存の逐次バックプロップ版 GRPO スクリプトをベースに、vLLM server mode を有効化したものです。以下の環境変数でホスト名/ポートを上書きできます。

```bash
export VLLM_SERVER_HOST=localhost
export VLLM_SERVER_PORT=8000
# Hopper GPU の場合は FlashAttention3 を強制する
export ATTN_IMPLEMENTATION=kernels-community/vllm-flash-attn3
```

学習を開始:
```bash
python train_grpo_vllm.py
```

## 3. 動作確認
1. `runs/grpo_gptoss20b_lora4_tes/training.log` を確認し、`Generation config` の `use_vllm=True, vllm_mode=server` が出力されているかチェック。
2. ログに出る `completions/clipped_ratio` が 1.0 で張り付いていないか、`reward` に分散が出ているかをウォッチ。  
3. `torch.cuda.max_memory_allocated()` をマイクロステップ毎に出力すると、逐次バックプロップでピークメモリが抑えられていることが確認できます。

## 4. よくあるトラブル
- **HTTP 503 / Connection refused**: vLLM サーバー側のウォームアップ中です。`--enforce-eager` を付ける／学習側で再試行リトライを追加。
- **推論が非常に遅い**: `--max-num-seqs` や `--tensor-parallel-size` の設定を確認。TP=2 に対して GPU が 1 台しか見えていない場合、vLLM サーバーが起動に失敗します。
- **FlashAttention3 がロードできない**: `flash-attn` の再インストール (`pip install flash-attn --no-build-isolation`) と、環境変数 `TORCH_CUDA_ARCH_LIST="9.0"` を設定してから PyTorch を再ビルド。

## 5. 後片付け
学習終了後はサーバープロセスを明示的に停止してください。
```bash
pkill -f "vllm serve"
```

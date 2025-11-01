# vLLM Server Mode 学習セットアップ（H100 ×2）

以下は GPT-OSS 20B を vLLM の server mode で高速生成しつつ、既存の逐次バックプロップ構成を維持したまま学習するための手順です。GPU は Hopper 世代 (H100) を 2 台使用します。**同一ノードで vLLM サーバーと学習プロセスを走らせる場合は、GPU を明示的に分けないとロード段階で OOM になるため注意してください。**

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
pip install transformers accelerate peft 'trl[vllm]>=0.23.1' datasets \
  --extra-index-url https://wheels.vllm.ai/0.10.2/ \
  'vllm==0.10.2'
pip install flash-attn --no-build-isolation  # FA3 カーネル
```

## 1. vLLM サーバー起動
学習ノードとは別プロセスでサーバーを立ち上げます（同一マシンで構いません）。Flash Attention 3 対応のコミュニティカーネルを明示します。

```bash
# 単一ノード (H100×2) で学習プロセスと GPU を分離する例
export CUDA_VISIBLE_DEVICES=0          # vLLM サーバーは GPU0 専用にする
export VLLM_USE_V1=1
export VLLM_TENSOR_PARALLEL_SIZE=1     # TP=1 に固定（GPU0 のみを使用）
trl vllm-serve \
  --model openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --trust-remote-code \
  --kv-cache-dtype auto
```

> **メモ**:  
> - `VLLM_TENSOR_PARALLEL_SIZE` は学習スクリプトと揃えます。サーバーと学習を同じノードで動かす場合は、TP=1 に落としてサーバーを 1 GPU に固定し、残りの GPU を学習に回してください。別ノードを用意できる場合や 3 枚以上の GPU がある場合のみ TP=2 以上での運用を推奨します。  
> - `--enforce-eager` は MXFP4 + FlashAttention3 の初期ウォームアップを安定させます。  
> - `--kv-cache-dtype auto` でモデル dtype に追従します（bf16 モデルなら KV も bf16）。FP8 を使いたい場合は `fp8` 系に変更してください。  
> - H100 環境では FlashAttention-3 が自動的に選ばれます。明示したい場合は `VLLM_ATTENTION_BACKEND=FLASH_ATTN` と `VLLM_FLASH_ATTN_VERSION=3` を追加でエクスポートしてください。
> - 生成負荷に応じて `--max-num-seqs` や `--max-model-len` を調整してください。

## 2. 学習ノード設定
`train_grpo_vllm.py` は既存の逐次バックプロップ版 GRPO スクリプトをベースに、vLLM server mode を有効化したものです。以下の環境変数でホスト名/ポートを上書きできます。

```bash
export VLLM_SERVER_HOST=localhost
export VLLM_SERVER_PORT=8000
# TP サイズはサーバーと合わせる（単一ノード例では 1）
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
# 学習プロセスを GPU1 にピン留め（サーバー側で GPU0 を使用している想定）
export CUDA_VISIBLE_DEVICES=1
# FlashAttention3 を明示したい場合（任意）
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_FLASH_ATTN_VERSION=3
```

学習を開始:
```bash
CUDA_VISIBLE_DEVICES=1 python train_grpo_vllm.py
```

## 3. 動作確認
1. `runs/grpo_gptoss20b_lora4_tes/training.log` を確認し、`Generation config` の `use_vllm=True, vllm_mode=server` が出力されているかチェック。
2. ログに出る `completions/clipped_ratio` が 1.0 で張り付いていないか、`reward` に分散が出ているかをウォッチ。  
3. `torch.cuda.max_memory_allocated()` をマイクロステップ毎に出力すると、逐次バックプロップでピークメモリが抑えられていることが確認できます。

## 4. よくあるトラブル
- **HTTP 503 / Connection refused**: vLLM サーバー側のウォームアップ中です。`--enforce-eager` を付ける／学習側で再試行リトライを追加。
- **推論が非常に遅い**: `--max-num-seqs` や `--tensor-parallel-size` の設定を確認。TP=2 に対して GPU が 1 台しか見えていない場合、vLLM サーバーが起動に失敗します。
- **CUDA OOM（ロード直後に発生）**: サーバーが GPU すべてを占有したまま学習プロセスが `AutoModelForCausalLM.from_pretrained` を呼ぶと MXFP4→BF16 のデクオン時に 2 GiB 程度確保できず失敗します。`pkill -f "vllm serve"` で既存プロセスを落とし、`CUDA_VISIBLE_DEVICES` を使ってサーバーと学習の GPU を分けてください。必要に応じて `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を追加すると断片化を抑えられます。
- **404 Not Found が vLLM から返る**: TRL 0.23+ の server mode は `trl vllm-serve` が立てる拡張 API を利用します。素の `vllm serve` では `/get_world_size/` などのエンドポイントが無く 404 になります。サーバーを停止 (`pkill -f "vllm serve"`) し、`pip install 'trl[vllm]>=0.23.1'` で CLI を導入した上で `trl vllm-serve ...` を使って再起動してください。
- **ValueError: Some specified arguments are not used**: `trl vllm-serve` は vLLM 本家の CLI とオプションが完全一致していません。未対応のオプションを渡すとこのエラーが出ます。`--max-num-seqs` など該当フラグを削除するか、`trl/scripts/vllm_serve.py --help` でサポートされている引数だけ指定してください。
- **TypeError: default_weight_loader() got an unexpected keyword argument 'weight_name'**: vLLM 側が 0.5 系など古いバージョンのままだと、TRL が送る LoRA 更新 RPC のシグネチャに未対応でこの例外が発生します。同じ venv に `pip install --upgrade --extra-index-url https://wheels.vllm.ai/0.10.2/ 'vllm==0.10.2'` を適用し、サーバーを再起動してください。
- **FlashAttention3 がロードできない**: `flash-attn` の再インストール (`pip install flash-attn --no-build-isolation`) と、環境変数 `TORCH_CUDA_ARCH_LIST="9.0"` を設定してから PyTorch を再ビルド。

## 5. 後片付け
学習終了後はサーバープロセスを明示的に停止してください。
```bash
pkill -f "trl vllm-serve"
```

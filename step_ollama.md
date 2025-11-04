## LoRA Adapter with Ollama (GPT-OSS)

### 1. 前提
- Ollama がインストール済みであること。
- 学習した LoRA アダプタ（例: `runs_10step分/grpo_gptoss20b_lora4_tes/`）をローカルに保持していること。
- ベースモデルは LoRA 学習時と同じ GPT-OSS (`gpt-oss:20b` または `gpt-oss:120b`) を使用すること。

### 2. ベースモデルの取得
```bash
ollama pull gpt-oss:20b   # 120b を使う場合は gpt-oss:120b
```
Ollama は GPT-OSS を公式対応しているため、そのまま Pull / Run が可能。

### 3. Modelfile の準備
プロジェクトルートに `Modelfile` を作成し、LoRA ディレクトリを `ADAPTER` で指定する。
```
FROM gpt-oss:20b
ADAPTER ./runs_10step分/grpo_gptoss20b_lora4_tes
```
> `./runs_10step分/grpo_gptoss20b_lora4_tes` には `adapter_model.safetensors`, `adapter_config.json` などが含まれている必要があります。

### 4. カスタムモデルの作成
```bash
ollama create gptoss-20b-myft -f Modelfile
```

### 5. 推論実行
```bash
ollama run gptoss-20b-myft
```
GPT-OSS は Harmony フォーマットを前提としているが、Ollama 側で自動的に Harmony テンプレートが適用されるため追加設定は不要。

### 6. Tips
- LoRA アダプタは量子化しない（QLoRA 等は非推奨）。非量子化 LoRA をそのまま使うこと。
- 追加の環境変数やテンプレート指定は不要。標準の Ollama プロンプトで動作する。
- ベースモデルと LoRA のバージョンが異なると正しく適用されないため、必ず同一ベースを使用する。


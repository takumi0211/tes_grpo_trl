# Ollama で GPT‑OSS 20B を動かす手順（Cloud PC 向け）

更新日: 2025-11-02

本手順は、Ubuntu 22.04 などのクラウド GPU マシン（x86_64）で、ゼロから Ollama をセットアップし `ollama run gpt-oss-20b` で推論するまでの流れをまとめたものです。Ollama の公式 Linux ビルドは x86_64 向けのみ提供されているため、Grace Hopper (GH200) のような aarch64 環境では動作しません。ARM 系での実験が必要な場合は、x86_64 インスタンスを新たに用意するか、QEMU/仮想化で代替してください。

---

## 0. ハードウェアとリソース要件

- OS: Ubuntu 22.04 LTS (x86_64)
- GPU: NVIDIA Ampere 以降（RTX 6000 Ada、A100、H100 など）。20B モデルを 16bit で動かすなら 48GB 以上の VRAM を推奨。
- CPU: 16 コア以上を目安（量子化してもダウンロードと初回コンパイルに時間がかかるため）
- RAM: 64GB 以上（モデル初期化の一時メモリが多い）
- ディスク: 50GB 以上の空き領域（モデル、キャッシュ、量子化生成物）
- ネットワーク: Hugging Face から 10GB を超えるチェックポイントを取得できる帯域

> VRAM が足りない場合は、Modelfile で量子化 (`PARAMETER quantize q4` など) を指定します。ただし GPT‑OSS 20B の量子化ビルドは公式で提供されていないため、初回起動時に自動生成されます。

---

## 1. NVIDIA ドライバと CUDA ランタイム

最新ドライバと CUDA 12.2 以降を導入済みであることを前提にしています。未導入の場合は、公式ドライバを先にインストールしてください。

```
sudo ubuntu-drivers autoinstall
sudo reboot
```

再起動後に `nvidia-smi` で確認します。

---

## 2. NVIDIA Container Toolkit（任意）

Ollama 自体はコンテナ不要ですが、将来コンテナ化する予定がある場合は事前に導入しておきます。

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

---

## 3. Ollama のインストール

公式インストーラを用いてセットアップします。curl で取得できない環境では `.deb` パッケージをダウンロードして `sudo dpkg -i` で導入してください。

```
curl -fsSL https://ollama.com/install.sh | sh
```

インストール後は自動的に `ollama` サービスが systemd で起動します。

状態確認:
```
sudo systemctl status ollama
ollama --version
```

---

## 4. Hugging Face アクセストークン（必要に応じて）

`openai/gpt-oss-20b` は基本的に公開リポジトリですが、企業アカウントではトークンが必要になるケースがあります。トークンを利用する場合は、以下のように環境変数を設定します。

```
export HUGGING_FACE_HUB_TOKEN=hf_xxx...
```

systemd サービスとして永続設定したい場合は、`/etc/systemd/system/ollama.service.d/env.conf` を作成して `Environment=HUGGING_FACE_HUB_TOKEN=...` を追記し、`sudo systemctl daemon-reload && sudo systemctl restart ollama` で反映します。

---

## 5. Modelfile の作成

`gpt-oss-20b` を Ollama 用に登録するための `Modelfile` をプロジェクトディレクトリに作成します。量子化を後から変更したい場合は `PARAMETER quantize` を追加してください。

```
cat <<'EOF' > Modelfile
FROM hf.co/openai/gpt-oss-20b
TEMPLATE "{{ .Prompt }}"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
# Hopper/Ada で FlashAttention 3 を有効化したい場合
# PARAMETER rope_frequency_base 500000
EOF
```

> `FROM` に Hugging Face の完全修飾名を指定すると、初回 `ollama create` で自動的にダウンロードされます。`TEMPLATE` を最小限にしているため、チャット形式のプロンプトはランタイム側で整形する必要があります。

---

## 6. モデルの作成

```
ollama create gpt-oss-20b -f Modelfile
```

初回は 10GB 以上をダウンロードし、VRAM / RAM に応じた量子化ファイルを生成します。ログは `/var/log/ollama/ollama.log` に出力されます。

- `pulling manifest` → Hugging Face からメタデータ取得
- `pulling layer` → safetensors チェックポイント各 shard のダウンロード
- `quantizing` → VRAM に最適化された量子化の生成（時間がかかる場合あり）

途中で中断した場合は、再度 `ollama create` を実行すると差分のみ再取得します。

---

## 7. 推論の実行

### 対話モード
```
ollama run gpt-oss-20b
```

```
>>> Who are you?
```

### ワンショット実行
```
ollama run gpt-oss-20b --prompt "Summarize the latest GRPO training techniques."
```

推論中の GPU 使用率は別ウィンドウで `watch -n 5 nvidia-smi` を実行して確認します。

---

## 8. サーバー API 経由での利用（任意）

Ollama はポート `11434` の HTTP API を提供しています。

```
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss-20b",
  "prompt": "Write a haiku about reinforcement learning."
}'
```

外部からアクセスする場合は、ファイアウォール設定でポートを明示的に許可し、認証やプロキシを併用してください。

---

## 9. トラブルシューティング

- `no CUDA runtime found`: NVIDIA ドライバ・CUDA toolkit がインストールされているか確認し、`LD_LIBRARY_PATH` に `/usr/lib/x86_64-linux-gnu` など CUDA ライブラリのパスを追加する。
- `model requires at least 48GB VRAM`: 量子化 (`PARAMETER quantize q4_K_M`) を指定するか、より大きい VRAM の GPU に切り替える。
- `permission denied on /usr/share/ollama`: `ollama create` は sudo 不要。root で実行した履歴があると所有権が変わるため、`sudo chown -R $USER:$USER /usr/share/ollama` で修正。
- Hugging Face へのアクセスが遅い: `HF_ENDPOINT=https://hf-mirror.com` のようにミラーを指定するか、事前に `huggingface-cli download` でキャッシュを作成。

---

## 10. クリーンアップ

実験後にモデルを削除する場合:

```
ollama rm gpt-oss-20b
```

Ollama サービスを停止する場合:

```
sudo systemctl stop ollama
sudo systemctl disable ollama
```

完全にアンインストールしたい場合は、`/usr/bin/ollama` と `/usr/share/ollama`、`~/.ollama` を削除し、`sudo rm -f /etc/systemd/system/ollama.service` を実行した上で `sudo systemctl daemon-reload` を行います。

---

## 11. 補足

- Ollama は llama.cpp ベースのバックエンドを使用します。20B モデルの量子化には時間がかかるので、初回セットアップはメンテナンス時間中に行うのがおすすめです。
- `ollama pull openai/gpt-oss-20b` のように直接指定することもできますが、Modelfile を使うと量子化やテンプレートのカスタマイズが容易になります。
- ワークロードが長時間になる場合は、`OLLAMA_KEEP_ALIVE=1h` などの環境変数を設定して、モデルのアンロードを防ぐことができます。


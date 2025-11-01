git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "vllmはとりあえず使えない。GPU2台で実装したらmaxtokenが5000でも動いた。次はGPUへのモデル配置を考える。"
git push origin main
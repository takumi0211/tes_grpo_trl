git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "次はGPUへのモデル配置を工夫"
git push origin main
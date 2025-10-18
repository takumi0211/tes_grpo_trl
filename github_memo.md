git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "環境構築に手こずった。"
git push origin main
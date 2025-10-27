git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change vllm setting and batch"
git push origin main
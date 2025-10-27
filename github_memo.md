git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change vllm setting and batch from 16 to 12"
git push origin main
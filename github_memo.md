git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "good preparation for vllm"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add vLLM and change to FP16"
git push origin main
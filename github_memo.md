git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "use vLLM false"
git push origin main
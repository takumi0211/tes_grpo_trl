git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add flash attention 3"
git push origin main
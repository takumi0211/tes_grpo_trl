git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change hypara of tmep and top_k"
git push origin main
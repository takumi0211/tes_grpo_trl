git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change to rank1"
git push origin main
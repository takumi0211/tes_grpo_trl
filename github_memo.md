git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change to only attention"
git push origin main
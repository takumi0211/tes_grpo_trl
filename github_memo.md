git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change training flow"
git push origin main
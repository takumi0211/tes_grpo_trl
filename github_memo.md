git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "revert Num gene"
git push origin main
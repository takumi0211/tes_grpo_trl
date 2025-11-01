git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change 2 tensor"
git push origin main
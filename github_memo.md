git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add tool"
git push origin main
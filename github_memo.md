git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change hypara 4→3"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add run train model"
git push origin main
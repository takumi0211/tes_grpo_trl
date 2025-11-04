git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "well done adding run train model and output model"
git push origin main
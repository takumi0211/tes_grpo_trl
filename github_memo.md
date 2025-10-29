git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change train_batch_size"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "delete eval and clean train_grpo.py"
git push origin main
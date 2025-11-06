git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "before delete log in train_grpo.py"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add eval_model before codex"
git push origin main
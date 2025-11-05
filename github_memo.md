git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change LoRA adapter 3層だけに"
git push origin main
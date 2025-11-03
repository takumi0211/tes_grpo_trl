git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "well done use_liger_loss=True"
git push origin main
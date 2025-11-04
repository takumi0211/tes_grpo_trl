git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add run model from HGF"
git push origin main
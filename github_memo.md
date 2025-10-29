git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "well done with H100 pcle"
git push origin main
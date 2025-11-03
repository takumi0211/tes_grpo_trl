git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add log hoow many generate"
git push origin main
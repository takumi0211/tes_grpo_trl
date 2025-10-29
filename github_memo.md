git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add logging and narrow doen the number"
git push origin main
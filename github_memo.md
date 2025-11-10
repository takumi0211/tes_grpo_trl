git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add sample test and update csv"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "add step_GH200.md"
git push origin main
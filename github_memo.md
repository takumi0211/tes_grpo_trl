git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "well done flash attention3"
git push origin main
git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "back to BF16"
git push origin main
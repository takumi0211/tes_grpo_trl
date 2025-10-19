git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "縦回しの失敗"
git push origin main
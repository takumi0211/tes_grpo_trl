git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "縦回しに変更"
git push origin main
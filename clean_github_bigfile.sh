sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.weights' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.caffemodel' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.264' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.mp4' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.tgz' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.wk' --prune-empty --tag-name-filter cat -- --all
sudo git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch *.tar.gz' --prune-empty --tag-name-filter cat -- --all

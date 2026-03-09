https://www.youtube.com/watch?v=s8cACi76hig
cd Desktop
cd GitHub_Local_Test
git init
git status
git add .
#git commit -a -m "adding my first files which are the LICQF consensus .py and AMPPS introduction"
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git remote add origin https://github.com/zelaky/Local_Test
git update-server-info
git commit -m "Initial commit"
git push -u origin main
Username for 'https://github.com': zelaky
Password for 'https://zelaky@github.com': #### (token)

Tasks: 
1. Connect local folder to GitHub repo without overwriting original version (make back-up)
(after connecting .git is in local folder)
Commands
git rm -r "your subfolder name"
git add — adding updated files to git repo (upload changes)
git commit (note on what you changed)
git push (replacing is the action)
https://rogerdudler.github.io/git-guide/

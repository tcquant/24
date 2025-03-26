@echo off

REM Change directory to the location of your git repository
cd /d "C:\Users\Tanmay\Desktop\drive"

REM Fetch the current date and time
for /f "tokens=1-5 delims=.:/ " %%a in ("%date% %time%") do (
    set datetime=%%a-%%b-%%c_%%d-%%e
)

REM Stage all changes
git add .

REM Commit with the current date and time as the message
git commit -m "DateTime - %datetime%"

REM Push the changes to the remote repository
git push origin main

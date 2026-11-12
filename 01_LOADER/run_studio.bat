@echo off
chcp 65001 >nul
echo Starting Ministral-3-8B Swedish Weight Cartridge Live Studio...
python "%~dp0live_studio.py" %*
pause

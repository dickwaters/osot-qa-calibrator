@echo off
chcp 65001 >nul
echo ====================================================
echo   OSOT 工具 02：标注质检、IAA 一致性与阈值校准器
echo ====================================================
echo.
set "PY_BIN=D:\otos\osot-annotator\.venv\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=python"

set /p AUTO_FILE=请拖入机器初标 / 标注员A JSON（直接回车运行 Demo）: 
if "%AUTO_FILE%"=="" (
    "%PY_BIN%" "%~dp0osot_qa_calibrator.py" --demo --out-dir "%~dp0demo_qa_out"
    start "" "%~dp0demo_qa_out\osot_qa_calibration_report.html"
    pause
    exit /b 0
)

set /p GOLD_FILE=请拖入人工终稿 / 标注员B JSON: 
"%PY_BIN%" "%~dp0osot_qa_calibrator.py" --auto %AUTO_FILE% --gold %GOLD_FILE% --out-dir "%~dp0qa_out"
start "" "%~dp0qa_out\osot_qa_calibration_report.html"
pause

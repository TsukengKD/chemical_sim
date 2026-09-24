@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Проверка и установка пакетов (numpy, numba, pygame-ce)...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Не удалось установить пакеты. Проверьте, что Python установлен с python.org
  echo и отмечена галочка "Add Python to PATH". Подойдёт Python 3.10-3.14.
  pause
  exit /b 1
)
echo.
echo Запуск. Первый раз подготовка займёт около 30 секунд...
python main.py
pause

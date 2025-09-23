@echo off
echo ====================================================
echo NASA IMS Bearing Fault Detection - Quick Start
echo ====================================================

echo.
echo 1. Testing environment setup...
python test_setup.py

echo.
echo 2. Installing/updating requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo 3. Creating project directories...
python setup.py

echo.
echo 4. Running the main fault detection system...
python Code.py

echo.
echo ====================================================
echo Completed! Check the outputs above for any errors.
echo ====================================================
pause
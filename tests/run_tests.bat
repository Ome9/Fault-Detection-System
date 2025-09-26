@echo off
REM Test runner for Refined Bearing Fault Detection Model
REM This script runs the complete test suite and generates reports

echo ====================================================
echo REFINED MODEL TEST SUITE
echo ====================================================
echo.

REM Check if model files exist
echo Checking model files...
if not exist "..\refined_deployment\refined_model_data.h" (
    echo ERROR: Model header file not found!
    echo Please ensure refined_model_data.h exists in refined_deployment/
    pause
    exit /b 1
)

if not exist "..\refined_deployment\refined_model.tflite" (
    echo WARNING: TensorFlow Lite model not found
    echo This is optional for testing but recommended for deployment
)

echo ✅ Model files found
echo.

REM Create build directory
if not exist "build" mkdir build

REM Compile the test program
echo Compiling test program...
gcc -Wall -Wextra -std=c99 -O2 -g -I..\refined_deployment -o build\refined_model_test.exe refined_model_test.c -lm

if %ERRORLEVEL% neq 0 (
    echo ERROR: Compilation failed!
    echo Please check that GCC is installed and available in PATH
    pause
    exit /b 1
)

echo ✅ Compilation successful
echo.

REM Generate additional test data (optional)
echo Generating additional test data...
python generate_test_data.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Test data generation failed (Python required)
    echo Continuing with built-in test cases...
)
echo.

REM Run the main test suite
echo Running test suite...
echo ====================================================
build\refined_model_test.exe

if %ERRORLEVEL% neq 0 (
    echo ERROR: Test execution failed!
    pause
    exit /b 1
)

echo.
echo ====================================================
echo TEST SUITE COMPLETED
echo ====================================================
echo.
echo To run interactive mode: build\refined_model_test.exe interactive
echo To run individual tests: python generate_test_data.py
echo.
pause
@echo off
echo ===================================================
echo   Deploying to Hugging Face Spaces
echo ===================================================

echo [1/2] Checking Git Configurations...
git remote -v

echo.
echo [2/2] Pushing to Hugging Face Space (sanketDamre/Fraud-Detection-System)...
echo.
echo IMPORTANT: A popup may appear asking for your credentials.
echo If asked for a password, use your Hugging Face ACCESS TOKEN (User Settings -> Access Tokens).
echo.
git push -f space main

echo.
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo ✅ DEPLOYMENT SUCCESSFUL!
    echo Your app is building at:
    echo https://huggingface.co/spaces/sanketDamre/Fraud-Detection-System
    echo ===================================================
) else (
    echo.
    echo ❌ Deployment Failed. Please check your internet or credentials.
)
pause

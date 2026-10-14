@echo off
REM ScamGuard Demo Script for Windows
REM Usage: demo.bat

setlocal enabledelayedexpansion

set BACKEND_DIR=backend
set API_BASE=http://localhost:8000
set ACCESS_TOKEN=

echo ========================================
echo   ScamGuard Demo Runner (Windows)
echo ========================================
echo.

REM Check prerequisites
where curl >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] curl not found in PATH
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] python not found in PATH
    exit /b 1
)

echo [OK] Prerequisites found
echo.

REM Start backend
echo [INFO] Starting backend server...
cd %BACKEND_DIR%

REM Check if already running
curl -s %API_BASE%/health >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Backend already running
    cd ..
    goto :auth_demo
)

echo [INFO] Starting uvicorn in background...
start /B poetry run uvicorn app.main:app --reload --port 8000 > ..\backend.log 2>&1
cd ..

echo -n [INFO] Waiting for backend
for /L %%i in (1,1,30) do (
    curl -s %API_BASE%/health >nul 2>nul
    if %errorlevel% equ 0 (
        echo  Ready!
        goto :auth_demo
    )
    echo -n .
    timeout /t 1 /nobreak >nul
)
echo.
echo [ERROR] Backend failed to start
exit /b 1

:auth_demo
echo.
echo ========================================
echo   DEMO 1: Auth Flow
echo ========================================

set EMAIL=demo@scamguard.com
set PASS=DemoPass123!

echo [INFO] Registering user...
curl -s -X POST %API_BASE%/api/v1/auth/register -H "Content-Type: application/json" -d "{\"email\": \"%EMAIL%\", \"password\": \"%PASS%\", \"full_name\": \"Demo User\"}"
echo.

echo [INFO] Logging in...
for /f "delims=" %%a in ('curl -s -X POST %API_BASE%/api/v1/auth/login -H "Content-Type: application/json" -d "{\"email\": \"%EMAIL%\", \"password\": \"%PASS%\"}"') do set LOGIN_RESP=%%a
echo %LOGIN_RESP%

REM Extract access_token (simple parsing)
for /f "tokens=2 delims=:" %%a in ('echo %LOGIN_RESP% ^| findstr /R "access_token"') do set RAW_TOKEN=%%a
set ACCESS_TOKEN=%RAW_TOKEN:~1,-2%
echo [OK] Access Token: %ACCESS_TOKEN:~0,20%...

echo [INFO] Getting current user...
curl -s -X GET %API_BASE%/api/v1/auth/me -H "Authorization: Bearer %ACCESS_TOKEN%"
echo.

echo [INFO] Auth demo complete
echo.

:audio_demo
echo ========================================
echo   DEMO 2: Live Audio Deepfake Detection
echo ========================================

if "%ACCESS_TOKEN%"=="" (
    echo [ERROR] No access token
    goto :sms_demo
)

echo [INFO] Creating Python test script...
(
echo import asyncio
echo import websockets
echo import json
echo import numpy as np
echo import sys
echo import os
echo.
echo async def test_audio_ws(token^):
echo     uri = f"ws://localhost:8000/ws/audio?token={token}&session_id=demo&chunk_duration_ms=500&sample_rate=16000"
echo     try:
echo         async with websockets.connect(uri) as ws:
echo             print("[OK] WebSocket connected")
echo             for i in range(3^):
echo                 t = np.linspace(0, 0.5, 8000)
echo                 audio = (0.3 * np.sin(2*np.pi*300*t) + 0.2 * np.sin(2*np.pi*600*t) + 0.1 * np.random.randn(8000))
echo                 pcm16 = (audio * 32767).astype(np.int16).tobytes()
echo                 await ws.send(pcm16)
echo                 print(f"> Sent chunk {i+1} ({len(pcm16)} bytes)")
echo                 response = await asyncio.wait_for(ws.recv(), timeout=5.0)
echo                 data = json.loads(response)
echo                 print(f"< Received: score={data['anomaly_score']:.3f} risk={data['is_high_risk']} time={data['processing_time_ms']:.1f}ms")
echo                 await asyncio.sleep(0.1)
echo             print("[OK] Audio demo complete")
echo     except Exception as e:
echo         print(f"[ERROR] {e}")
echo         sys.exit(1)
echo.
echo if __name__ == "__main__":^
echo     token = os.environ.get("ACCESS_TOKEN", "")
echo     if not token:
echo         print("[ERROR] ACCESS_TOKEN not set")
echo         sys.exit(1)
echo     asyncio.run(test_audio_ws(token))
) > %TEMP%\test_audio_ws.py

echo [INFO] Installing websockets/numpy if needed...
python -m pip install websockets numpy -q 2>nul || echo [WARN] pip install had issues

echo [INFO] Running WebSocket client...
set ACCESS_TOKEN=%ACCESS_TOKEN%
python %TEMP%\test_audio_ws.py

echo [OK] Audio demo complete
echo.

:sms_demo
echo ========================================
echo   DEMO 3: SMS Scam Classification
echo ========================================

if "%ACCESS_TOKEN%"=="" (
    echo [ERROR] No access token
    goto :done
)

echo [INFO] Submitting scam SMS...
curl -s -X POST %API_BASE%/api/v1/nlp/score -H "Content-Type: application/json" -H "Authorization: Bearer %ACCESS_TOKEN%" -d "{\"text\": \"URGENT: Your bank account will be closed in 1 hour. Verify now at bit.ly/fake-bank-login\", \"sender\": \"+15551234567\"}"
echo.

echo [INFO] Waiting for background worker (3s)...
timeout /t 3 /nobreak >nul

echo [INFO] Fetching events...
curl -s -X GET %API_BASE%/api/v1/nlp/events -H "Authorization: Bearer %ACCESS_TOKEN%"
echo.

echo [INFO] Testing additional examples...
for %%t in (
    "This is IRS. You owe $5000. Pay now or we arrest you."
    "Your OTP is 123456. Reply with code to verify."
    "Congratulations! You won $1000 gift card. Claim at bit.ly/claim"
    "Hey, are we still meeting for lunch at 1pm?"
) do (
    echo [INFO] Text: %%t
    curl -s -X POST %API_BASE%/api/v1/nlp/score -H "Content-Type: application/json" -H "Authorization: Bearer %ACCESS_TOKEN%" -d "{\"text\": \"%%t\"}" >nul
    timeout /t 2 /nobreak >nul
    curl -s -X GET %API_BASE%/api/v1/nlp/events -H "Authorization: Bearer %ACCESS_TOKEN%"
    echo.
)

echo [OK] SMS demo complete
echo.

:done
echo ========================================
echo   ALL DEMOS COMPLETE
echo ========================================
echo.
echo Backend log: backend.log
echo Backend still running - close this window to stop
echo.
pause
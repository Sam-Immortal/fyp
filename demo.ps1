<#
.SYNOPSIS
    ScamGuard Demo Runner - PowerShell version for Windows
.DESCRIPTION
    Runs Auth, Audio WebSocket, and SMS NLP demos sequentially
.USAGE
    .\demo.ps1
#>

$ErrorActionPreference = "Stop"

$BACKEND_DIR = "backend"
$API_BASE = "http://localhost:8000"
$ACCESS_TOKEN = ""
$BACKEND_PROCESS = $null

function Write-Color($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Check-Prereqs {
    Write-Color "Checking prerequisites..." "Yellow"
    
    $missing = @()
    "curl", "python", "jq" | ForEach-Object {
        if (-not (Get-Command $_ -ErrorAction SilentlyContinue)) {
            $missing += $_
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Color "Missing: $($missing -join ', ')" "Red"
        Write-Color "Install with: choco install curl jq python" "Yellow"
        exit 1
    }
    
    Write-Color "Prerequisites OK" "Green"
}

function Start-Backend {
    Write-Color "Starting backend server..." "Yellow"
    Set-Location $BACKEND_DIR
    
    # Check if already running
    try {
        $resp = Invoke-RestMethod -Uri "$API_BASE/health" -Method Get -ErrorAction Stop
        Write-Color "Backend already running" "Green"
        Set-Location ..
        return
    } catch {
        # Not running, start it
    }
    
    Write-Color "Starting uvicorn..." "Yellow"
    $BACKEND_PROCESS = Start-Process poetry -ArgumentList "run uvicorn app.main:app --reload --port 8000" -PassThru -RedirectStandardOutput "..\backend.log" -RedirectStandardError "..\backend.log" -WorkingDirectory (Get-Location)
    
    Set-Location ..
    
    # Wait for health
    Write-Color -NoNewline "Waiting for backend"
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "$API_BASE/health" -Method Get -ErrorAction Stop
            Write-Color " Ready!" "Green"
            return
        } catch {
            Write-Color -NoNewline "."
            Start-Sleep 1
        }
    }
    Write-Color " Failed!" "Red"
    exit 1
}

function Demo-Auth {
    Write-Color "`n==========================================" "Blue"
    Write-Color "  DEMO 1: Auth Flow" "Blue"
    Write-Color "==========================================" "Blue"
    
    $EMAIL = "demo@scamguard.com"
    $PASS = "DemoPass123!"
    
    Write-Color "Registering user..." "Yellow"
    $regBody = @{ email = $EMAIL; password = $PASS; full_name = "Demo User" } | ConvertTo-Json
    $regResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/register" -Method Post -Body $regBody -ContentType "application/json"
    $regResp | ConvertTo-Json -Depth 3
    
    Write-Color "Logging in..." "Yellow"
    $loginBody = @{ email = $EMAIL; password = $PASS } | ConvertTo-Json
    $loginResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/login" -Method Post -Body $loginBody -ContentType "application/json"
    $loginResp | ConvertTo-Json -Depth 3
    
    $global:ACCESS_TOKEN = $loginResp.access_token
    $refreshToken = $loginResp.refresh_token
    
    Write-Color "Access Token: $($global:ACCESS_TOKEN.Substring(0,20))..." "Green"
    Write-Color "Refresh Token: $($refreshToken.Substring(0,20))..." "Green"
    
    Write-Color "Getting current user..." "Yellow"
    $meResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/me" -Method Get -Headers @{ Authorization = "Bearer $global:ACCESS_TOKEN" }
    $meResp | ConvertTo-Json -Depth 3
    
    Write-Color "Refreshing token..." "Yellow"
    $refreshBody = @{ refresh_token = $refreshToken } | ConvertTo-Json
    $refreshResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/refresh" -Method Post -Body $refreshBody -ContentType "application/json"
    $refreshResp | ConvertTo-Json -Depth 3
    
    Write-Color "Logging out (revokes refresh)..." "Yellow"
    $logoutBody = @{ refresh_token = $refreshToken } | ConvertTo-Json
    Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/logout" -Method Post -Body $logoutBody -ContentType "application/json"
    
    Write-Color "Auth demo complete" "Green"
}

function Demo-Audio {
    Write-Color "`n==========================================" "Blue"
    Write-Color "  DEMO 2: Live Audio Deepfake Detection" "Blue"
    Write-Color "==========================================" "Blue"
    
    if (-not $global:ACCESS_TOKEN) {
        Write-Color "No access token - run auth demo first" "Red"
        return
    }
    
    Write-Color "Creating Python test client..." "Yellow"
    
    $pyScript = @"
import asyncio
import websockets
import json
import numpy as np
import sys
import os

async def test_audio_ws(token):
    uri = f"ws://localhost:8000/ws/audio?token={token}&session_id=demo&chunk_duration_ms=500&sample_rate=16000"
    try:
        async with websockets.connect(uri) as ws:
            print("[OK] WebSocket connected")
            for i in range(3):
                t = np.linspace(0, 0.5, 8000)
                audio = (0.3 * np.sin(2*np.pi*300*t) + 0.2 * np.sin(2*np.pi*600*t) + 0.1 * np.random.randn(8000))
                pcm16 = (audio * 32767).astype(np.int16).tobytes()
                await ws.send(pcm16)
                print(f"> Sent chunk {i+1} ({len(pcm16)} bytes)")
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"< Received: score={data['anomaly_score']:.3f} risk={data['is_high_risk']} time={data['processing_time_ms']:.1f}ms")
                await asyncio.sleep(0.1)
            print("[OK] Audio demo complete")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    token = os.environ.get("ACCESS_TOKEN", "")
    if not token:
        print("[ERROR] ACCESS_TOKEN not set")
        sys.exit(1)
    asyncio.run(test_audio_ws(token))
"@
    
    $pyPath = "$env:TEMP\test_audio_ws.py"
    $pyScript | Set-Content -Path $pyPath -Encoding UTF8
    
    Write-Color "Installing websockets/numpy if needed..." "Yellow"
    & python -m pip install websockets numpy -q 2>$null
    
    Write-Color "Running WebSocket client (3 chunks)..." "Yellow"
    $env:ACCESS_TOKEN = $global:ACCESS_TOKEN
    & python $pyPath
    
    Write-Color "Audio demo complete" "Green"
}

function Demo-SMS {
    Write-Color "`n==========================================" "Blue"
    Write-Color "  DEMO 3: SMS Scam Classification" "Blue"
    Write-Color "==========================================" "Blue"
    
    if (-not $global:ACCESS_TOKEN) {
        Write-Color "No access token - run auth demo first" "Red"
        return
    }
    
    $headers = @{ Authorization = "Bearer $global:ACCESS_TOKEN"; "Content-Type" = "application/json" }
    
    Write-Color "Submitting scam SMS..." "Yellow"
    $body = @{ text = "URGENT: Your bank account will be closed in 1 hour. Verify now at bit.ly/fake-bank-login"; sender = "+15551234567" } | ConvertTo-Json
    $submitResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/nlp/score" -Method Post -Headers $headers -Body $body
    $submitResp | ConvertTo-Json -Depth 3
    
    Write-Color "Waiting for background worker (3s)..." "Yellow"
    Start-Sleep 3
    
    Write-Color "Fetching events..." "Yellow"
    $eventsResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/nlp/events" -Method Get -Headers $headers
    $eventsResp | ConvertTo-Json -Depth 3
    
    Write-Color "Testing additional examples..." "Yellow"
    
    $testTexts = @(
        "This is IRS. You owe $5000. Pay now or we arrest you.",
        "Your OTP is 123456. Reply with code to verify.",
        "Congratulations! You won $1000 gift card. Claim at bit.ly/claim",
        "Hey, are we still meeting for lunch at 1pm?"
    )
    
    foreach ($text in $testTexts) {
        Write-Color "Text: $text" "Yellow"
        $testBody = @{ text = $text } | ConvertTo-Json
        Invoke-RestMethod -Uri "$API_BASE/api/v1/nlp/score" -Method Post -Headers $headers -Body $testBody | Out-Null
        Start-Sleep 2
        $eventsResp = Invoke-RestMethod -Uri "$API_BASE/api/v1/nlp/events" -Method Get -Headers $headers
        $eventsResp[0] | ConvertTo-Json -Depth 3
    }
    
    Write-Color "SMS demo complete" "Green"
}

function Cleanup {
    Write-Color "`nCleaning up..." "Yellow"
    if ($BACKEND_PROCESS) {
        Stop-Process -Id $BACKEND_PROCESS.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item "$env:TEMP\test_audio_ws.py" -ErrorAction SilentlyContinue
    Write-Color "Done" "Green"
}

# Main
try {
    Check-Prereqs
    Start-Backend
    Demo-Auth
    Demo-Audio
    Demo-SMS
    
    Write-Color "`n==========================================" "Blue"
    Write-Color "  ALL DEMOS COMPLETE" "Green"
    Write-Color "==========================================" "Blue"
    Write-Color "Backend log: backend.log" "Gray"
    Write-Color "Backend process: $($BACKEND_PROCESS.Id) (still running - close window to stop)" "Gray"
} finally {
    Cleanup
}
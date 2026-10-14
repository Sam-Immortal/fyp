#!/usr/bin/env bash
# ScamGuard Demo Script - Runs Auth, Audio WS, and SMS NLP demos
# Usage: ./demo.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BACKEND_DIR="backend"
API_BASE="http://localhost:8000"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ScamGuard Demo Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
check_prereqs() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}curl not found${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}jq not found - JSON output will be raw${NC}"
        HAS_JQ=false
    else
        HAS_JQ=true
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}python3 not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Prerequisites OK${NC}"
    echo ""
}

# Start backend
start_backend() {
    echo -e "${YELLOW}Starting backend server...${NC}"
    cd "$BACKEND_DIR"
    
    # Check if already running
    if curl -s "$API_BASE/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Backend already running${NC}"
        cd ..
        return
    fi
    
    # Start in background
    poetry run uvicorn app.main:app --reload --port 8000 > ../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    # Wait for health check
    echo -n "Waiting for backend"
    for i in {1..30}; do
        if curl -s "$API_BASE/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            return
        fi
        echo -n "."
        sleep 1
    done
    echo -e " ${RED}Failed to start${NC}"
    exit 1
}

# Auth demo
demo_auth() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  DEMO 1: Auth Flow${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    EMAIL="demo@scamguard.com"
    PASS="DemoPass123!"
    
    echo -e "${YELLOW}Registering user...${NC}"
    REG_RESP=$(curl -s -X POST "$API_BASE/api/v1/auth/register" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$EMAIL\", \"password\": \"$PASS\", \"full_name\": \"Demo User\"}")
    
    if $HAS_JQ; then
        echo "$REG_RESP" | jq .
    else
        echo "$REG_RESP"
    fi
    
    echo -e "${YELLOW}Logging in...${NC}"
    LOGIN_RESP=$(curl -s -X POST "$API_BASE/api/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$EMAIL\", \"password\": \"$PASS\"}")
    
    if $HAS_JQ; then
        echo "$LOGIN_RESP" | jq .
        ACCESS_TOKEN=$(echo "$LOGIN_RESP" | jq -r .access_token)
        REFRESH_TOKEN=$(echo "$LOGIN_RESP" | jq -r .refresh_token)
    else
        echo "$LOGIN_RESP"
        # Extract tokens without jq (fragile)
        ACCESS_TOKEN=$(echo "$LOGIN_RESP" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
        REFRESH_TOKEN=$(echo "$LOGIN_RESP" | grep -o '"refresh_token":"[^"]*"' | cut -d'"' -f4)
    fi
    
    echo -e "${GREEN}Access Token: ${ACCESS_TOKEN:0:20}...${NC}"
    echo -e "${GREEN}Refresh Token: ${REFRESH_TOKEN:0:20}...${NC}"
    
    echo -e "${YELLOW}Getting current user...${NC}"
    ME_RESP=$(curl -s -X GET "$API_BASE/api/v1/auth/me" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if $HAS_JQ; then echo "$ME_RESP" | jq .; else echo "$ME_RESP"; fi
    
    echo -e "${YELLOW}Refreshing token...${NC}"
    REFRESH_RESP=$(curl -s -X POST "$API_BASE/api/v1/auth/refresh" \
        -H "Content-Type: application/json" \
        -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}")
    
    if $HAS_JQ; then echo "$REFRESH_RESP" | jq .; else echo "$REFRESH_RESP"; fi
    
    echo -e "${YELLOW}Logging out (revokes refresh)...${NC}"
    LOGOUT_RESP=$(curl -s -X POST "$API_BASE/api/v1/auth/logout" \
        -H "Content-Type: application/json" \
        -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}")
    
    echo -e "${GREEN}Auth demo complete${NC}"
    echo ""
    
    # Export for other demos
    export ACCESS_TOKEN
}

# Audio WS demo
demo_audio() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  DEMO 2: Live Audio Deepfake Detection${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${RED}No access token - run auth demo first${NC}"
        return
    fi
    
    echo -e "${YELLOW}Creating Python test client...${NC}"
    
    cat > /tmp/test_audio_ws.py << 'PYEOF'
import asyncio
import websockets
import json
import numpy as np
import sys

async def test_audio_ws(token):
    uri = f"ws://localhost:8000/ws/audio?token={token}&session_id=demo&chunk_duration_ms=500&sample_rate=16000"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ WebSocket connected")
            
            for i in range(3):
                # Generate synthetic speech-like audio
                t = np.linspace(0, 0.5, 8000)
                audio = (0.3 * np.sin(2*np.pi*300*t) + 
                         0.2 * np.sin(2*np.pi*600*t) + 
                         0.1 * np.random.randn(8000))
                pcm16 = (audio * 32767).astype(np.int16).tobytes()
                
                await ws.send(pcm16)
                print(f"→ Sent chunk {i+1} ({len(pcm16)} bytes)")
                
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"← Received: score={data['anomaly_score']:.3f} risk={data['is_high_risk']} time={data['processing_time_ms']:.1f}ms")
                
                await asyncio.sleep(0.1)
            
            print("✓ Audio demo complete")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import os
    token = os.environ.get("ACCESS_TOKEN", sys.argv[1] if len(sys.argv) > 1 else "")
    if not token:
        print("Usage: ACCESS_TOKEN=xxx python test_audio_ws.py")
        sys.exit(1)
    asyncio.run(test_audio_ws(token))
PYEOF

    echo -e "${YELLOW}Installing websockets if needed...${NC}"
    pip install websockets numpy -q 2>/dev/null || true
    
    echo -e "${YELLOW}Running WebSocket client (3 chunks)...${NC}"
    ACCESS_TOKEN="$ACCESS_TOKEN" python3 /tmp/test_audio_ws.py
    
    echo -e "${GREEN}Audio demo complete${NC}"
    echo ""
}

# SMS NLP demo
demo_sms() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  DEMO 3: SMS Scam Classification${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${RED}No access token - run auth demo first${NC}"
        return
    fi
    
    echo -e "${YELLOW}Submitting scam SMS for analysis...${NC}"
    
    SUBMIT_RESP=$(curl -s -X POST "$API_BASE/api/v1/nlp/score" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -d '{"text": "URGENT: Your bank account will be closed in 1 hour. Verify now at bit.ly/fake-bank-login", "sender": "+15551234567"}')
    
    if $HAS_JQ; then echo "$SUBMIT_RESP" | jq .; else echo "$SUBMIT_RESP"; fi
    
    echo -e "${YELLOW}Waiting for background worker (3s)...${NC}"
    sleep 3
    
    echo -e "${YELLOW}Fetching events...${NC}"
    EVENTS_RESP=$(curl -s -X GET "$API_BASE/api/v1/nlp/events" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if $HAS_JQ; then echo "$EVENTS_RESP" | jq .; else echo "$EVENTS_RESP"; fi
    
    # Test more examples
    echo -e "${YELLOW}Testing additional examples...${NC}"
    
    for text in \
        "This is IRS. You owe $5000. Pay now or we arrest you." \
        "Your OTP is 123456. Reply with code to verify." \
        "Congratulations! You won $1000 gift card. Claim at bit.ly/claim" \
        "Hey, are we still meeting for lunch at 1pm?"
    do
        echo -e "${YELLOW}Text: $text${NC}"
        curl -s -X POST "$API_BASE/api/v1/nlp/score" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -d "{\"text\": \"$text\"}" > /dev/null
        sleep 2
        EVENTS_RESP=$(curl -s -X GET "$API_BASE/api/v1/nlp/events" -H "Authorization: Bearer $ACCESS_TOKEN")
        if $HAS_JQ; then echo "$EVENTS_RESP" | jq '.[0]'; else echo "$EVENTS_RESP"; fi
    done
    
    echo -e "${GREEN}SMS demo complete${NC}"
    echo ""
}

# Cleanup
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    rm -f /tmp/test_audio_ws.py
    echo -e "${GREEN}Done${NC}"
}

# Main
main() {
    check_prereqs
    start_backend
    demo_auth
    demo_audio
    demo_sms
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}  ALL DEMOS COMPLETE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Backend log: backend.log"
    echo "Backend PID: $BACKEND_PID (still running - kill manually if needed)"
}

# Trap cleanup on exit
trap cleanup EXIT

main "$@"
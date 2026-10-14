"""
Live demo client for ScamGuard's /ws/audio deepfake-detection endpoint.

This streams a real audio file to the SAME WebSocket endpoint a live phone
call would use, paced in real time, so you can demo the detection pipeline
end to end without needing live call-audio capture on Android (which, as
established, isn't feasible on stock Android today).

Setup (one-time):
    pip install websockets soundfile numpy requests

Usage:
    python demo_ws_audio.py --flac path/to/some_utterance.flac \
        --email alice@example.com --password Password123

    # or, if you already have a token from Swagger:
    python demo_ws_audio.py --flac path/to/some_utterance.flac --token eyJ...

What it does:
    1. Logs in via POST /auth/login (unless --token is given) to get a
       real access token, the same way the Android app would.
    2. Loads the audio file and, if it's shorter than the model's fixed
       4.04s (64,600-sample) window, loops it to fill one window --
       exactly the same method app/audio/inference.py uses server-side,
       so a short ASVspoof2019 utterance still produces a real score
       instead of silently never reaching the threshold.
    3. Opens the WebSocket, same as a real call would, and streams the
       audio in small real-time-paced chunks.
    4. Prints each score as it arrives, live, as the server sends it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import numpy as np
import requests
import soundfile as sf
import websockets

AASIST_SAMPLE_RATE = 16000
AASIST_MAX_LEN = 64600  # ~4.04s at 16kHz -- must match app/audio/inference.py

# How much audio to send per WebSocket frame, in seconds. Smaller = more
# granular live updates on the wire; doesn't need to match the model's
# scoring window since the server buffers across frames itself.
SEND_CHUNK_SECONDS = 0.32


def login(base_url: str, email: str, password: str) -> str:
    """Log in exactly like the Android app would, and return the access token."""
    resp = requests.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"Login failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        print(
            "If this user doesn't exist yet, register it first via Swagger "
            "at /docs, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    return resp.json()["access_token"]


def load_and_prepare_audio(flac_path: str) -> tuple[np.ndarray, int]:
    """Load the file and loop it to fill one scoring window if it's short."""
    audio, sr = sf.read(flac_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # collapse to mono if needed

    duration_s = len(audio) / sr
    print(f"Loaded {flac_path}: {duration_s:.2f}s at {sr}Hz, {len(audio)} samples")

    native_window = int(AASIST_MAX_LEN * sr / AASIST_SAMPLE_RATE)
    if len(audio) < native_window:
        num_repeats = int(native_window / len(audio)) + 1
        audio = np.tile(audio, num_repeats)[:native_window]
        print(
            f"  (shorter than the model's ~4.04s window -- looped to fill "
            f"one window, {len(audio)} samples, same method the server uses)"
        )

    return audio, sr


async def stream_audio(ws_url: str, audio: np.ndarray, sr: int) -> None:
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    bytes_per_sample = 2
    chunk_bytes = int(SEND_CHUNK_SECONDS * sr) * bytes_per_sample

    print(f"\nConnecting to {ws_url.split('?')[0]} ...")
    async with websockets.connect(ws_url, max_size=None) as ws:
        print("Connected. Streaming audio in real time...\n")

        async def sender():
            for i in range(0, len(pcm), chunk_bytes):
                await ws.send(pcm[i : i + chunk_bytes])
                await asyncio.sleep(SEND_CHUNK_SECONDS)
            # Give the server a moment to flush the final partial window,
            # then close our send side.
            await asyncio.sleep(0.5)

        async def receiver():
            try:
                async for message in ws:
                    data = json.loads(message)
                    flag = "\033[91mHIGH RISK\033[0m" if data["is_high_risk"] else "monitoring"
                    print(
                        f"  chunk {data['chunk_index']:>2}  "
                        f"score={data['anomaly_score']:.4f}  "
                        f"threshold={data['threshold']}  "
                        f"[{flag}]  "
                        f"({data['processing_time_ms']:.1f} ms)"
                    )
            except websockets.exceptions.ConnectionClosed:
                pass

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await send_task
        await asyncio.sleep(1.0)  # let any trailing score arrive
        recv_task.cancel()

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flac", required=True, help="Path to a .flac/.wav audio file")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", help="Existing JWT access token (skips login)")
    parser.add_argument("--email", help="Login email (if --token not given)")
    parser.add_argument("--password", help="Login password (if --token not given)")
    args = parser.parse_args()

    if args.token:
        token = args.token
    else:
        if not args.email or not args.password:
            parser.error("Provide --token, or both --email and --password")
        token = login(args.base_url, args.email, args.password)
        print("Logged in, got access token.")

    audio, sr = load_and_prepare_audio(args.flac)

    session_id = str(uuid.uuid4())
    ws_base = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_base}/ws/audio?token={token}&session_id={session_id}&sample_rate={sr}"

    asyncio.run(stream_audio(ws_url, audio, sr))


if __name__ == "__main__":
    main()

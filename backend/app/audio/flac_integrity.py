"""Strict FLAC decoding and integrity verification (no soundfile dependency).

Why this exists: libsndfile rejects some ASVspoof 2021 FLAC files ("flac decoder lost
sync"). A lenient ffmpeg decode of such a file can silently conceal damage, so a file that
only ffmpeg can read must be *proven* intact before it is scored.

Proof of integrity, strongest first:
  1. ffmpeg decodes with every error treated as fatal (-xerror, CRC and bitstream checks).
  2. The decoded sample count equals the total_samples declared in the FLAC STREAMINFO.
  3. The MD5 of the decoded PCM equals the MD5 stored in STREAMINFO (the same check
     `flac -t` performs). An all-zero stored MD5 means "not recorded".
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

STRICT_FLAGS = ["-xerror", "-err_detect", "crccheck+bitstream+explode"]
_PCM_FORMATS = {16: ("s16le", "pcm_s16le", np.dtype("<i2"), 32768.0)}


class FlacIntegrityError(RuntimeError):
    """Raised when a file cannot be proven intact. `status` is a short reason code."""

    def __init__(self, status: str, detail: str = "") -> None:
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class StreamInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int  # 0 means "unknown"
    md5: bytes  # 16 bytes; all zero means "not recorded"


def read_streaminfo(path) -> StreamInfo:
    """Parse the mandatory STREAMINFO block (skipping a leading ID3v2 tag if present)."""
    try:
        with open(path, "rb") as f:
            head = f.read(10)
            if head[:3] == b"ID3" and len(head) == 10:
                size = (
                    ((head[6] & 0x7F) << 21)
                    | ((head[7] & 0x7F) << 14)
                    | ((head[8] & 0x7F) << 7)
                    | (head[9] & 0x7F)
                )
                f.seek(10 + size)
                magic = f.read(4)
            else:
                magic = head[:4]
                f.seek(4)
            if magic != b"fLaC":
                raise FlacIntegrityError("bad_header", f"magic={magic!r}")
            block = f.read(4)
            if len(block) < 4 or (block[0] & 0x7F) != 0:
                raise FlacIntegrityError("bad_header", "first metadata block is not STREAMINFO")
            length = int.from_bytes(block[1:4], "big")
            body = f.read(length)
    except OSError as exc:
        raise FlacIntegrityError("unreadable", str(exc)) from exc
    if length != 34 or len(body) != 34:
        raise FlacIntegrityError("bad_header", f"STREAMINFO length {length}/{len(body)}")
    packed = int.from_bytes(body[10:18], "big")
    return StreamInfo(
        sample_rate=(packed >> 44) & 0xFFFFF,
        channels=((packed >> 41) & 0x7) + 1,
        bits_per_sample=((packed >> 36) & 0x1F) + 1,
        total_samples=packed & 0xFFFFFFFFF,
        md5=bytes(body[18:34]),
    )


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FlacIntegrityError("ffmpeg_missing", "ffmpeg not found on PATH")
    return exe


def decode_pcm_strict(path, bits: int = 16, timeout: float = 120.0) -> bytes:
    """Decode to interleaved little-endian PCM at native rate/channels; any error is fatal."""
    if bits not in _PCM_FORMATS:
        raise FlacIntegrityError("unsupported_bit_depth", str(bits))
    fmt, codec, _, _ = _PCM_FORMATS[bits]
    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        *STRICT_FLAGS,
        "-i",
        str(path),
        "-f",
        fmt,
        "-acodec",
        codec,
        "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FlacIntegrityError("decode_timeout", str(path)) from exc
    err = res.stderr.decode("utf-8", errors="ignore").strip()
    if res.returncode != 0 or err:
        raise FlacIntegrityError("decode_error", (err or f"rc={res.returncode}")[:300])
    return res.stdout


def verify_and_decode(
    path, accept_md5_mismatch: bool = False
) -> tuple[np.ndarray, int, StreamInfo, str]:
    """Return (mono float32 audio, sample_rate, streaminfo, md5_state).

    md5_state is "verified" (md5 matched), "not_recorded" (no md5 in header; the strict decode
    and length check still passed) or, only when accept_md5_mismatch=True, "md5_mismatch"
    (every frame CRC and the length are fine but the header MD5 disagrees; the caller must flag
    such files). Raises FlacIntegrityError otherwise.
    """
    info = read_streaminfo(path)
    if info.bits_per_sample not in _PCM_FORMATS:
        raise FlacIntegrityError("unsupported_bit_depth", str(info.bits_per_sample))
    pcm = decode_pcm_strict(path, info.bits_per_sample)
    _, _, dtype, scale = _PCM_FORMATS[info.bits_per_sample]
    frame_bytes = dtype.itemsize * info.channels
    if len(pcm) % frame_bytes:
        raise FlacIntegrityError("length_mismatch", "PCM not a whole number of frames")
    n_frames = len(pcm) // frame_bytes
    if info.total_samples and n_frames != info.total_samples:
        raise FlacIntegrityError(
            "length_mismatch", f"decoded {n_frames} frames, header declares {info.total_samples}"
        )
    if info.md5 != bytes(16):
        if hashlib.md5(pcm).digest() == info.md5:
            state = "verified"
        elif accept_md5_mismatch:
            state = "md5_mismatch"
        else:
            raise FlacIntegrityError("md5_mismatch", "decoded audio does not match stored MD5")
    else:
        state = "not_recorded"
    samples = np.frombuffer(pcm, dtype=dtype).reshape(-1, info.channels).astype(np.float32)
    mono = samples.mean(axis=1) / scale  # axis=1: average channels, keep time axis
    return mono.astype(np.float32), info.sample_rate, info, state

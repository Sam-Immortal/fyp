"""Tests for strict FLAC verification (app.audio.flac_integrity).

Builds tiny FLAC files with ffmpeg, so they are skipped when ffmpeg is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pytest

from app.audio.flac_integrity import FlacIntegrityError, read_streaminfo, verify_and_decode

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("flac")
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 2) / 16000
    sig = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(d / "src.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(d / "src.wav"), str(d / "ok.flac")], check=True
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(d / "src.wav"),
            "-ac",
            "2",
            str(d / "stereo.flac"),
        ],
        check=True,
    )
    data = bytearray((d / "ok.flac").read_bytes())
    bad = bytearray(data)
    mid = len(bad) // 2
    for i in range(mid, mid + 40):
        bad[i] ^= 0xFF
    (d / "corrupt.flac").write_bytes(bytes(bad))
    (d / "truncated.flac").write_bytes(bytes(data[: int(len(data) * 0.6)]))
    (d / "badheader.flac").write_bytes(b"RIFF" + bytes(data[4:]))
    md5bad = bytearray(data)
    md5bad[26] ^= 0xFF  # one byte of the STREAMINFO MD5 only; frames and length untouched
    (d / "md5bad.flac").write_bytes(bytes(md5bad))
    return d, pcm.astype(np.float32) / 32768.0


def test_streaminfo(clips):
    d, ref = clips
    info = read_streaminfo(d / "ok.flac")
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (16000, 1, 16)
    assert info.total_samples == ref.size


def test_healthy_file_is_bit_exact_and_md5_verified(clips):
    d, ref = clips
    audio, sr, _info, state = verify_and_decode(d / "ok.flac")
    assert sr == 16000 and state == "verified"
    assert np.array_equal(audio, ref)


def test_stereo_is_downmixed_over_channels_not_time(clips):
    d, ref = clips
    audio, _sr, info, _state = verify_and_decode(d / "stereo.flac")
    assert info.channels == 2
    assert audio.shape == ref.shape  # the old axis=0 mean would have produced shape (2,)


@pytest.mark.parametrize(
    ("name", "statuses"),
    [
        ("corrupt.flac", {"length_mismatch", "md5_mismatch", "decode_error"}),
        ("truncated.flac", {"decode_error", "length_mismatch"}),
        ("badheader.flac", {"bad_header"}),
    ],
)
def test_damaged_files_are_rejected(clips, name, statuses):
    d, _ = clips
    with pytest.raises(FlacIntegrityError) as exc:
        verify_and_decode(d / name)
    assert exc.value.status in statuses


def test_lenient_ffmpeg_would_have_accepted_the_corrupt_file(clips):
    """Documents why the strict path is needed: a plain decode returns rc=0 and short audio.
    
    On ffmpeg >= 9.0, the lenient decoder may already conceal the corruption and
    return the full expected number of samples, in which case this test is skipped.
    """
    d, ref = clips
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(d / "corrupt.flac"),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0
    sample_count = len(r.stdout) // 4
    if sample_count == ref.size:
        pytest.skip("this ffmpeg version conceals the corruption")
    assert sample_count != ref.size


def test_md5_only_mismatch_is_rejected_by_default(clips):
    d, _ = clips
    with pytest.raises(FlacIntegrityError) as exc:
        verify_and_decode(d / "md5bad.flac")
    assert exc.value.status == "md5_mismatch"


def test_md5_only_mismatch_can_be_accepted_and_is_flagged(clips):
    d, ref = clips
    audio, _sr, _info, state = verify_and_decode(d / "md5bad.flac", accept_md5_mismatch=True)
    assert state == "md5_mismatch"
    assert np.array_equal(audio, ref)  # the audio itself is intact


def test_accept_flag_does_not_rescue_real_damage(clips):
    d, _ = clips
    for name in ("corrupt.flac", "truncated.flac", "badheader.flac"):
        with pytest.raises(FlacIntegrityError):
            verify_and_decode(d / name, accept_md5_mismatch=True)

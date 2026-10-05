"""Anti-forensic attacks on decoded video (Step 5).

Clips are handled as raw yuv420p in sensor orientation, center-cropped to the
same window as the analysis frames, so the attacker's fingerprint lines up with
the pixels it is added to or subtracted from. Only luma (Y) is attacked; chroma
is passed through. Every variant, including the alpha = 0 control, goes through
the same libx264 encoder, so compression alone cannot separate the classes.
"""
import subprocess

import numpy as np

PRNU_STD = 0.01   # attacker fingerprints are scaled to a 1% PRNU; alpha multiplies this


def read_clip(path, crop, seconds):
    """Decode the first `seconds` of a video as (y, uv): y (n, c, c), uv (n, 2 * (c/2)^2) uint8."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(path), "-t", str(seconds),
        "-an", "-vf", f"crop={crop}:{crop},format=yuv420p", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])
    luma, frame = crop * crop, crop * crop * 3 // 2
    n = len(proc.stdout) // frame
    if n == 0:
        raise RuntimeError("no frames decoded")
    raw = np.frombuffer(proc.stdout[: n * frame], np.uint8).reshape(n, frame)
    return raw[:, :luma].reshape(n, crop, crop), raw[:, luma:]


def attacker_fingerprint(k):
    """Zero-mean fingerprint rescaled to a PRNU of standard deviation PRNU_STD."""
    k = k - k.mean()
    return (k / k.std() * PRNU_STD).astype(np.float32)


def prnu_frames(y, k, alpha, sign):
    """Y' = Y + sign * alpha * Y * K, frame by frame (sign -1: removal, +1: injection)."""
    gain = (1.0 + sign * alpha * k).astype(np.float32)
    for frame in y:
        yield np.clip(np.rint(frame * gain), 0, 255).astype(np.uint8)


def encode(y_frames, uv, out_path, crop, fps, crf, preset, gop, vf=None):
    """Encode luma frames (+ original chroma) with libx264, one I-frame every `gop` frames."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{crop}x{crop}", "-r", f"{fps:.3f}", "-i", "pipe:0",
        *(["-vf", vf] if vf else []),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for y, c in zip(y_frames, uv):
            proc.stdin.write(y.tobytes())
            proc.stdin.write(c.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(err[-500:])

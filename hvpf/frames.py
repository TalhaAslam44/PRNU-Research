"""I-frame extraction into sensor-aligned grayscale crops.

PRNU is a pixel-wise pattern, so every frame must be in the sensor's own
orientation and geometry before it is compared with a fingerprint:

* decode with -noautorotate (rotation metadata is ignored, pixels stay as the sensor wrote them);
* YouTube/WhatsApp copies may have the rotation baked into the pixels and are
  downscaled, so they are rotated back and rescaled to their native counterpart's size;
* then a fixed center crop of the luma (gray) plane is kept.
"""
import subprocess

import numpy as np

# ffmpeg filter that rotates pixels clockwise by the given angle
ROTATE_CW = {0: None, 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}


def build_filter(width, height, native_w, native_h, rotate_cw, crop, step):
    """Filter chain for one video. width/height: stored size of this file."""
    chain = [f"select='not(mod(n\\,{step}))'"] if step > 1 else []
    if ROTATE_CW[rotate_cw]:
        chain.append(ROTATE_CW[rotate_cw])
        if rotate_cw in (90, 270):
            width, height = height, width
    if (width, height) != (native_w, native_h):
        chain.append(f"scale={native_w}:{native_h}:flags=bicubic")
    chain.append(f"crop={crop}:{crop}")          # centered by default
    chain.append("format=gray")                  # luma only
    return ",".join(chain)


def extract_iframes(path, vf, crop, max_frames):
    """Decode only key frames and return a uint8 array (n, crop, crop)."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-noautorotate", "-skip_frame", "nokey", "-i", str(path),
        "-an", "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(max_frames),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])
    n = len(proc.stdout) // (crop * crop)
    if n == 0:
        raise RuntimeError("no frames decoded")
    return np.frombuffer(proc.stdout[: n * crop * crop], np.uint8).reshape(n, crop, crop)

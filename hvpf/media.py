"""ffprobe / ffmpeg helpers."""
import json
import subprocess
from fractions import Fraction

VIDEO_EXTS = {".mp4", ".mov", ".3gp", ".m4v", ".avi", ".mkv", ".mts", ".m2ts"}  # .mts: AVCHD camcorders (ACID)


def _rate(s):
    try:
        f = Fraction(s)
        return round(float(f), 3) if f.denominator else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _rotation(stream):
    """Clockwise rotation (0/90/180/270) a player applies for display."""
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is not None:
        return int(rotate) % 360
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(-float(sd["rotation"])) % 360
    return 0


def probe(path):
    """Probe the first video stream: geometry, codec, fps, duration, I-frame count.

    Reads every packet header (no decoding), so truncated or corrupt files show up
    as demuxer errors or a missing stream.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames"
        ":stream_side_data=rotation:stream_tags=rotate"
        ":format=duration,bit_rate,size,format_name"
        ":packet=flags",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    errors = proc.stderr.strip()
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    if proc.returncode != 0 or not streams:
        return {"ok": False, "error": errors[:300] or "no video stream"}

    s = streams[0]
    packets = info.get("packets", [])
    duration = float(fmt.get("duration", 0) or 0)
    row = {
        "codec": s.get("codec_name"),
        "profile": s.get("profile"),
        "width": s.get("width"),
        "height": s.get("height"),
        "pix_fmt": s.get("pix_fmt"),
        "fps": _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate")),
        "n_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else len(packets),
        "n_iframes": sum("K" in p.get("flags", "") for p in packets),
        "duration_s": round(duration, 3),
        "bitrate_kbps": round(int(fmt.get("bit_rate", 0) or 0) / 1000),
        "size_mb": round(int(fmt.get("size", 0) or 0) / 1e6, 2),
        "container": fmt.get("format_name"),
        "rotation": _rotation(s),
        "ok": True,
        "error": errors[:300],
    }
    if duration < 1 or row["n_iframes"] == 0 or not row["width"]:
        row["ok"] = False
        row["error"] = row["error"] or "empty or undecodable stream"
    return row

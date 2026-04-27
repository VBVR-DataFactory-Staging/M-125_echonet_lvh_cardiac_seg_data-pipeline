"""Video / frame helpers for M-125 EchoNet-LVH cardiac-cycle prediction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
#  AVI -> frame list
# --------------------------------------------------------------------------

def load_avi_frames(path: str | Path) -> List[np.ndarray]:
    """Load all frames of an .avi as a list of BGR uint8 arrays."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def get_avi_fps(path: str | Path) -> float:
    """Read native FPS from an .avi (returns 0.0 if unknown)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    try:
        return float(fps)
    except Exception:
        return 0.0


def resize_bgr(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a BGR frame to (h_tgt, w_tgt) using bilinear."""
    import cv2
    h_tgt, w_tgt = target_hw
    if frame.shape[:2] == (h_tgt, w_tgt):
        return frame
    return cv2.resize(frame, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
#  Frame-count resampling
# --------------------------------------------------------------------------

def resample_frames(frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
    """Uniformly resample a list of frames to target_count via nearest indexing."""
    if not frames or target_count <= 0:
        return []
    n = len(frames)
    if n == target_count:
        return list(frames)
    idxs = np.linspace(0, n - 1, num=target_count).round().astype(int)
    return [frames[int(i)] for i in idxs]


def extract_one_cycle(
    frames: List[np.ndarray],
    ed_frame: int,
    es_frame: Optional[int],
    native_fps: float,
) -> List[np.ndarray]:
    """Pick one full ED-to-ED cardiac cycle from a clip.

    Strategy:
      - Use ED frame as start.
      - If we know ES frame, the ED→ES interval is half a cycle. Cycle len
        ≈ 2 * (ES - ED) frames (assumes symmetric systole/diastole). This
        is approximate but fine for prediction-task ground truth.
      - If ES is missing or before ED, fall back to ~1.0 s of native fps,
        clamped to [12, 60] frames.
    """
    n = len(frames)
    if n == 0:
        return []
    ed = max(0, min(ed_frame, n - 1))

    if es_frame is not None and es_frame > ed:
        half = es_frame - ed
        cycle_len = max(2 * half, 8)
    else:
        # Heuristic 1 second of native fps, clamped.
        sec_frames = int(round(native_fps)) if native_fps > 0 else 30
        cycle_len = max(12, min(60, sec_frames))

    end = min(n, ed + cycle_len + 1)
    return frames[ed:end]


# --------------------------------------------------------------------------
#  Video encoding
# --------------------------------------------------------------------------

def write_h264_video(frames: Iterable[np.ndarray], out_path: str | Path, fps: int) -> None:
    """Encode BGR frames to h264 yuv420p mp4 via ffmpeg stdin."""
    import cv2
    frames = list(frames)
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2 = w - (w % 2)
    h2 = h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()

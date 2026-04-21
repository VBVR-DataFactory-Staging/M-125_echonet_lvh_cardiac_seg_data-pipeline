"""Shared utilities for Med-VR phase2 pipelines (M-028~M-055).

Provides:
- NIfTI → (image frames, label frames) extraction
- Overlay drawing for seg/bbox/mask
- MP4 writing via ffmpeg
- Standard example_output 7-file layout writer

All outputs go to datasets/_example_output/M-0XX_{task}/task_XXXX/.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np
import cv2

REPO_ROOT = Path("/home/ubuntu/vbvr/medical")
DATA_ROOT = REPO_ROOT / "datasets"
EXAMPLE_ROOT = DATA_ROOT / "_example_output"

# ---------- NIfTI helpers ----------

def load_nifti_volume(path: Path):
    import nibabel as nib
    vol = nib.load(str(path))
    arr = vol.get_fdata()
    # Reorder to (z, y, x) with z as slice axis
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    return arr, vol.affine

def window_ct(slice2d, wl=40, ww=400):
    lo, hi = wl - ww / 2, wl + ww / 2
    s = np.clip(slice2d, lo, hi)
    s = ((s - lo) / (hi - lo) * 255).astype(np.uint8)
    return s

def window_minmax(slice2d):
    s = slice2d.astype(np.float32)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    if hi <= lo:
        hi = lo + 1
    s = np.clip((s - lo) / (hi - lo), 0, 1) * 255
    return s.astype(np.uint8)

def to_rgb(gray):
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

# ---------- Overlay helpers ----------

COLORS = {
    "red":    (0,   0,   255),
    "green":  (0,   200, 0),
    "blue":   (255, 100, 0),
    "yellow": (0,   255, 255),
    "cyan":   (255, 255, 0),
    "magenta":(255, 0,   255),
    "orange": (0,   140, 255),
    "purple": (200, 0,   200),
    "lime":   (100, 255, 100),
    "pink":   (180, 105, 255),
    "brown":  (60,  80,  140),
    "white":  (255, 255, 255),
    "gray":   (128, 128, 128),
    "navy":   (128, 0,   0),
    "teal":   (128, 128, 0),
}

def overlay_mask(rgb_bgr, mask, color=(0, 200, 0), alpha=0.45):
    out = rgb_bgr.copy()
    mask_bool = mask > 0
    layer = out.copy()
    layer[mask_bool] = color
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    # contour
    m8 = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out

def overlay_multi(rgb_bgr, label_map, color_list, alpha=0.5):
    """label_map: int array; each unique >0 value gets a color from color_list (list of (name,bgr))."""
    out = rgb_bgr.copy()
    for idx, (name, color) in enumerate(color_list, start=1):
        m = (label_map == idx)
        if not m.any():
            continue
        layer = out.copy()
        layer[m] = color
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        m8 = m.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 1)
    return out

def draw_bbox(rgb_bgr, bboxes, color=(0, 200, 0), labels=None):
    out = rgb_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if labels:
            cv2.putText(out, labels[i], (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def fit_square(img, size=512, is_mask=False):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    if r.ndim == 2:
        canvas = np.zeros((size, size), dtype=r.dtype)
    else:
        canvas = np.zeros((size, size, 3), dtype=r.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas

# ---------- MP4 writer ----------

def pick_annotated_idx(gt_frame_count_or_flags):
    """Given a list of bools / annotated flags, pick middle of the annotated ones."""
    flags = gt_frame_count_or_flags
    idx = [i for i, f in enumerate(flags) if f]
    if not idx:
        return 0
    return idx[len(idx) // 2]


def write_mp4(frames_bgr, out_path: Path, fps: int):
    """Write frames as H.264 MP4 via ffmpeg for broad playback compatibility.

    OpenCV's mp4v fourcc produces files QuickTime / Chrome cannot decode
    (shows blank/green frames). Using libx264 + yuv420p fixes that.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    # ffmpeg needs even dimensions for yuv420p
    pad_w = w + (w & 1)
    pad_h = h + (h & 1)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-vf", f"pad={pad_w}:{pad_h}:0:0:color=black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_bgr:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed for {out_path}: {err}")

# ---------- Task output writer ----------

def write_task(pipeline_id: str, task_name: str, task_idx: int,
               first_frame, final_frame,
               first_video_frames, last_video_frames, gt_video_frames,
               prompt: str, metadata: dict, fps: int):
    """Write the 7-file standard layout."""
    out_dir = EXAMPLE_ROOT / f"{pipeline_id}_{task_name}" / f"task_{task_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "first_frame.png"), first_frame)
    cv2.imwrite(str(out_dir / "final_frame.png"), final_frame)
    write_mp4(first_video_frames, out_dir / "first_video.mp4", fps)
    write_mp4(last_video_frames,  out_dir / "last_video.mp4",  fps)
    write_mp4(gt_video_frames,    out_dir / "ground_truth.mp4", fps)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps({**metadata, "fps": fps, "pipeline_id": pipeline_id},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

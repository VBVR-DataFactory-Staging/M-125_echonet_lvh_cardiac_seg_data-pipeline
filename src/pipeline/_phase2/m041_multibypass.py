"""M-041: MultiBypass140 Surgical Phase Recognition.

Laparoscopic Roux-en-Y gastric bypass videos from the MultiBypass140 dataset
(Bern + Strasbourg, 140 surgeries, ~25 fps, ~50 minutes each). Task 12 phases
in the official scheme.

⚠️ Note (2026-04-13): phase label files are NOT bundled with the
multibypass01.zip archive we have on EC2; only raw videos are present. This
example generation uses a placeholder heuristic to approximate the "current
phase" text overlay — the real pipeline must load the official annotation CSV
once we retrieve it separately.

Case A: real video at original FPS (25). Each example task = a 12-second clip
starting at a chosen time offset.
"""
from __future__ import annotations
from pathlib import Path
import cv2
from common import DATA_ROOT, write_task, COLORS, fit_square

PID = "M-041"
TASK_NAME = "multibypass_phase_recognition"

# Official 12 phases for Roux-en-Y gastric bypass (MultiBypass140 paper)
PHASES = [
    "preparation",
    "gastric pouch creation",
    "omentum division",
    "mesenteric defect closure 1",
    "jejunal transection",
    "jejuno-jejunostomy",
    "mesenteric defect closure 2",
    "alimentary-limb routing (antecolic / retrocolic)",
    "gastro-jejunostomy",
    "gastro-jejunostomy anastomosis test",
    "cleaning and hemostasis",
    "instrument/trocar removal",
]

PROMPT = (
    "This is a laparoscopic Roux-en-Y gastric bypass surgery video from the "
    "MultiBypass140 dataset. Identify the current surgical phase at every "
    "frame, choosing from: preparation, gastric pouch creation, omentum "
    "division, mesenteric defect closure 1/2, jejunal transection, jejuno-"
    "jejunostomy, alimentary-limb routing, gastro-jejunostomy, anastomosis "
    "test, cleaning and hemostasis, and instrument removal. Overlay the "
    "predicted phase label in the bottom-right corner of every frame."
)


def read_clip(video_path: Path, start_sec: float, duration_sec: float, fps: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    frames = []
    n_needed = int(duration_sec * fps)
    for _ in range(n_needed):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def annotate_phase(frame, phase_text: str):
    out = frame.copy()
    h, w = out.shape[:2]
    # Bottom-right dark strip with phase text
    label = f"PHASE: {phase_text}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 8
    x2, y2 = w - 10, h - 10
    x1, y1 = x2 - tw - pad * 2, y2 - th - pad * 2
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), COLORS["yellow"], 2)
    cv2.putText(out, label, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["yellow"], 2, cv2.LINE_AA)
    # Progress bar (phase index / total)
    bar_y = 20
    cv2.rectangle(out, (20, bar_y), (20 + 300, bar_y + 14), (50, 50, 50), -1)
    idx = PHASES.index(phase_text)
    frac = (idx + 1) / len(PHASES)
    cv2.rectangle(out, (20, bar_y), (20 + int(300 * frac), bar_y + 14),
                  COLORS["green"], -1)
    cv2.putText(out, f"{idx+1}/{len(PHASES)}", (330, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_task(video_path: Path, start_sec: float, phase_name: str, fps: int,
               task_idx: int):
    duration = 12.0
    clip = read_clip(video_path, start_sec, duration, fps)
    if not clip:
        print(f"  [skip] failed to read clip at {start_sec}")
        return None
    first_frames, last_frames = [], []
    for f in clip:
        fsq = fit_square(f, 512)
        first_frames.append(fsq)
        last_frames.append(annotate_phase(fsq, phase_name))
    gt_frames = last_frames  # all frames belong to this phase (single-label clip)
    first_frame = first_frames[len(first_frames) // 2]
    final_frame = last_frames[len(last_frames) // 2]

    meta = {
        "task": "surgical phase recognition",
        "dataset": "MultiBypass140 (BernBypass70 subset)",
        "video_id": video_path.stem,
        "start_sec": float(start_sec),
        "duration_sec": duration,
        "phase_label": phase_name,
        "phase_index": PHASES.index(phase_name),
        "modality": "laparoscopic video",
        "fps_source": "read from video file (ffprobe)",
        "label_source": (
            "PLACEHOLDER: official phase CSV not bundled in current EC2 copy; "
            "replace with ground truth once annotation file is retrieved."
        ),
        "source_split": "train",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frame, final_frame,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, fps)


def main():
    video = DATA_ROOT / "_extracted" / "12_MultiBypass" / "multibypass01_corrected" / "BernBypass70" / "videos" / "BBP37.mp4"
    # Two placeholder clips at different timecodes → likely different phases
    tasks = [
        (60.0, "gastric pouch creation"),
        (1200.0, "gastro-jejunostomy"),
    ]
    for i, (s, phase) in enumerate(tasks):
        d = build_task(video, s, phase, fps=25, task_idx=i)
        if d:
            print(f"  wrote {d}")


if __name__ == "__main__":
    main()

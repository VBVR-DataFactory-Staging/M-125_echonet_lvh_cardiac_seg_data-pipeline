"""M-125 pipeline: EchoNet-LVH echocardiogram -> heartbeat-cycle prediction task.

Each sample:
    Input:  first_frame.png = ED frame (parasternal long-axis view, 512x512)
    Output: ground_truth.mp4 = full cardiac cycle (~3 s @ 16 fps)

All three videos (first_video, last_video, ground_truth) are produced for the
standard 7-file VBVR layout. first_video = held ED-frame loop (static seed
clip). last_video = ground_truth.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import core.pipeline as _core_pipeline
from core.pipeline import BasePipeline, SampleProcessor, TaskSample
from core.download import run_download

from .config import TaskConfig
from . import transforms


# ---------------------------------------------------------------------------
# Scrub x-access-token:ghp_xxx@ credentials embedded in git remote URLs so
# tokens don't leak into every sample's metadata.json. (HARNESS Bug 5.)
# Older core/pipeline.py revisions don't have _git_info; this is a no-op there.
# ---------------------------------------------------------------------------
_URL_CRED_RE = re.compile(r"(https?://)[^/@\s]+@")
if hasattr(_core_pipeline, "_git_info"):
    _original_git_info = _core_pipeline._git_info

    def _sanitized_git_info() -> dict:
        info = _original_git_info()
        repo = info.get("repo")
        if isinstance(repo, str):
            info["repo"] = _URL_CRED_RE.sub(r"\1", repo)
        return info

    _core_pipeline._git_info = _sanitized_git_info


TMP_PREFIX = "vbvr_tmp_m125_"


class TaskPipeline(BasePipeline):
    """Convert one EchoNet-LVH .avi into one heartbeat-cycle task sample."""

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config

    # -- 1) Download -------------------------------------------------------
    def download(self) -> Iterator[dict]:
        yield from run_download(self.task_config)

    # -- 2) Process --------------------------------------------------------
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        avi_path = raw_sample["avi_path"]
        hashed = raw_sample["hashed_name"]
        ed_frame = int(raw_sample.get("ed_frame") or 0)
        es_frame = raw_sample.get("es_frame")
        native_fps = float(raw_sample.get("fps_native") or 0.0)

        target_hw = (int(self.task_config.height), int(self.task_config.width))
        target_n = int(self.task_config.num_frames)
        out_fps = int(self.task_config.fps)

        # Decode the .avi.
        raw_frames = transforms.load_avi_frames(avi_path)
        if not raw_frames:
            return None
        # Resize all to target HW first (so video sizes are uniform).
        raw_frames = [transforms.resize_bgr(f, target_hw) for f in raw_frames]

        # Pull one cycle starting at ED.
        cycle_frames = transforms.extract_one_cycle(
            raw_frames, ed_frame, es_frame, native_fps)
        if len(cycle_frames) < 2:
            # Fallback: just take a chunk from start.
            cycle_frames = raw_frames[: max(8, min(48, len(raw_frames)))]
        if len(cycle_frames) < 2:
            return None

        # Resample cycle to target frame count.
        gt_frames = transforms.resample_frames(cycle_frames, target_n)
        if len(gt_frames) < 2:
            return None

        ed_bgr = gt_frames[0]
        # final_frame = last frame of GT (visually should be near ED again).
        final_bgr = gt_frames[-1]

        # first_video: static ED frame held for target_n frames (seed loop).
        seed_loop = [ed_bgr.copy() for _ in range(target_n)]
        # last_video: same as ground truth.
        last_frames = gt_frames

        domain = self.task_config.domain
        global_idx = int(getattr(self.task_config, "start_index", 0)) + idx
        task_id = f"echonet_lvh_{global_idx:05d}"

        tmp_root = Path(tempfile.mkdtemp(prefix=TMP_PREFIX))
        first_video_path = tmp_root / "first_video.mp4"
        last_video_path = tmp_root / "last_video.mp4"
        gt_video_path = tmp_root / "ground_truth.mp4"

        transforms.write_h264_video(seed_loop, first_video_path, fps=out_fps)
        transforms.write_h264_video(last_frames, last_video_path, fps=out_fps)
        transforms.write_h264_video(gt_frames, gt_video_path, fps=out_fps)

        prompt_text = self.task_config.task_prompt_template

        metadata = {
            "dataset": "EchoNet-LVH",
            "source_org": "Stanford AIMI",
            "hashed_file_name": hashed,
            "zip_member": raw_sample.get("zip_member"),
            "ed_frame_index": ed_frame,
            "es_frame_index": es_frame,
            "native_num_frames": raw_sample.get("frames"),
            "native_frame_rate": native_fps,
            "native_width": raw_sample.get("width_native"),
            "native_height": raw_sample.get("height_native"),
            "split": raw_sample.get("split"),
            "lv_measurements_cm": {
                "IVSd": raw_sample.get("ivsd_cm"),
                "LVIDd": raw_sample.get("lvidd_cm"),
                "LVIDs": raw_sample.get("lvids_cm"),
                "LVPWd": raw_sample.get("lvpwd_cm"),
            },
            "task_format": "image_plus_text_to_video",
            "fps": out_fps,
            "num_frames": target_n,
            "width": int(self.task_config.width),
            "height": int(self.task_config.height),
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=domain,
            first_image=ed_bgr,
            prompt=prompt_text,
            final_image=final_bgr,
            first_video=str(first_video_path),
            last_video=str(last_video_path),
            ground_truth_video=str(gt_video_path),
            metadata=metadata,
        )

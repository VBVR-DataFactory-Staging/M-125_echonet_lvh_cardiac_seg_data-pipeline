"""Config for M-125 EchoNet-LVH cardiac-cycle prediction pipeline.

Input:  single EchoNet-LVH parasternal-LV echocardiogram frame at end-diastole (ED)
Output: predicted one full cardiac cycle as a video, ~48 frames at 16 fps (3 s).

Raw data lives as a single 73.79 GB ZIP on S3. The downloader streams individual
.avi members via boto3 range fetch + zlib inflate; we deliberately do NOT set
``s3_prefix`` so the v3 bootstrap won't try to ``aws s3 sync`` the whole zip
into raw/.
"""

from pathlib import Path

from pydantic import Field

from core.pipeline import PipelineConfig


TASK_PROMPT_TEMPLATE = (
    "Given this Stanford EchoNet-LVH parasternal-long-axis echocardiogram frame "
    "at end-diastole (left ventricle maximally filled), predict one complete "
    "cardiac cycle showing systolic contraction (ventricular walls thicken and "
    "the left-ventricular chamber shrinks) followed by diastolic relaxation "
    "(walls thin and the chamber re-expands). Preserve the LV chamber geometry, "
    "the interventricular septum (IVS) and posterior wall (LVPW), and the "
    "overall acoustic-window shape. The generated video should end at "
    "approximately the same end-diastolic state it started from. "
    "No text overlays in the generated video."
)


class TaskConfig(PipelineConfig):
    """EchoNet-LVH cardiac-cycle prediction settings."""

    domain: str = Field(default="echonet_lvh_cardiac_cycle_prediction")

    # ZIP source on S3 — streamed via range fetch, NOT bulk-synced.
    # We intentionally leave s3_prefix empty so the v3 bootstrap skips the
    # "aws s3 sync raw/" step (would try to drag the 73 GB zip).
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(default="")
    s3_zip_key: str = Field(default="M-125/echonet_lvh/echonetlvh/EchoNet-LVH.zip")
    raw_dir: Path = Field(default=Path("raw"))

    # Video output spec (3 s @ 16 fps, Wan2.2-compatible).
    fps: int = Field(default=16, ge=1)
    num_frames: int = Field(default=48, ge=2)

    # Output frame size — EchoNet-LVH native is 1024x768 (or 800x600); rescale.
    width: int = Field(default=512, ge=64)
    height: int = Field(default=512, ge=64)

    # Cap total samples for one EC2 run.
    max_samples: int = Field(default=300, ge=1)

    task_prompt_template: str = Field(default=TASK_PROMPT_TEMPLATE)

"""Core pipeline module — process raw data into the standardized VBVR format.

Provides the base pipeline class, sample processor, output writer, image
utilities, validator, and shared schemas.  The actual dataset-specific
processing logic lives in ``src.pipeline``; this module always delegates
to it via :func:`run_pipeline`.
"""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


# ============================================================================
#  Pydantic schemas
# ============================================================================

class PipelineConfig(BaseModel):
    """Pipeline configuration — base class for task-specific configs."""

    num_samples: Optional[int] = None
    domain: str = "dataset"
    output_dir: Path = Path("data/questions")
    split: str = "test"


class TaskSample(BaseModel):
    """A single task sample in the standardized format.

    Mirrors the TaskPair schema from template-data-generator so both
    generators and pipelines produce the same on-disk layout.
    """

    task_id: str
    domain: str
    prompt: str
    first_image: Any  # PIL Image
    final_image: Optional[Any] = None  # PIL Image
    first_video: Optional[str] = None  # Path to first segment video
    last_video: Optional[str] = None  # Path to last segment video
    ground_truth_video: Optional[str] = None  # Path to full video
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
#  Image utilities
# ============================================================================

def convert_to_pil_image(
    image_input: Union[Image.Image, np.ndarray, str, Path, None],
    mode: str = "RGB",
) -> Optional[Image.Image]:
    """Convert various image formats to PIL Image.

    Args:
        image_input: PIL Image, numpy array, or file path.
        mode: Target colour mode (default: RGB).

    Returns:
        PIL Image in *mode*, or ``None`` if conversion fails.
    """
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        return image_input.convert(mode) if image_input.mode != mode else image_input

    if isinstance(image_input, np.ndarray):
        return numpy_to_pil(image_input, mode)

    if isinstance(image_input, (str, Path)):
        return load_from_path(Path(image_input), mode)

    return None


def numpy_to_pil(arr: np.ndarray, mode: str = "RGB") -> Optional[Image.Image]:
    """Convert numpy array to PIL Image."""
    if arr.dtype in [np.float32, np.float64]:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA")
        else:
            return None
    else:
        return None

    return img.convert(mode) if img.mode != mode else img


def load_from_path(path: Path, mode: str = "RGB") -> Optional[Image.Image]:
    """Load image from file path."""
    if not path.exists():
        return None
    img = Image.open(path)
    return img.convert(mode) if img.mode != mode else img


# ============================================================================
#  Sample processor
# ============================================================================

class SampleProcessor:
    """Utilities for building TaskSample objects from raw data."""

    @staticmethod
    def build_sample(
        task_id: str,
        domain: str,
        first_image: Any,
        prompt: str,
        final_image: Any = None,
        first_video: Optional[str] = None,
        last_video: Optional[str] = None,
        ground_truth_video: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskSample]:
        """Build a validated TaskSample from raw inputs.

        Converts images via ``convert_to_pil_image`` and returns ``None``
        if required fields are missing or invalid.
        """
        first_pil = convert_to_pil_image(first_image)
        if first_pil is None:
            return None

        if not prompt or not prompt.strip():
            return None

        final_pil = convert_to_pil_image(final_image) if final_image is not None else None

        return TaskSample(
            task_id=task_id,
            domain=domain,
            prompt=prompt.strip(),
            first_image=first_pil,
            final_image=final_pil,
            first_video=first_video,
            last_video=last_video,
            ground_truth_video=ground_truth_video,
            metadata=metadata,
        )


# ============================================================================
#  Output writer
# ============================================================================

class OutputWriter:
    """Writes task samples to the standardized directory layout.

    Produces::

        data/questions/{domain}_task/{task_id}/
        ├── first_frame.png
        ├── final_frame.png
        ├── prompt.txt
        ├── first_video.mp4
        ├── last_video.mp4
        ├── ground_truth.mp4
        └── metadata.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_sample(self, sample: TaskSample) -> Path:
        """Write a single sample to disk."""
        task_dir = self.output_dir / f"{sample.domain}_task" / sample.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Write images
        if sample.first_image is not None:
            sample.first_image.save(task_dir / "first_frame.png")

        if sample.final_image is not None:
            sample.final_image.save(task_dir / "final_frame.png")

        # Write prompt
        (task_dir / "prompt.txt").write_text(sample.prompt)

        # Write videos if provided
        for video_path, filename in [
            (sample.first_video, "first_video.mp4"),
            (sample.last_video, "last_video.mp4"),
            (sample.ground_truth_video, "ground_truth.mp4"),
        ]:
            if video_path is not None and Path(video_path).exists():
                video_src = Path(video_path)
                shutil.copy(video_src, task_dir / filename)

        # Write metadata
        if sample.metadata is not None:
            (task_dir / "metadata.json").write_text(
                json.dumps(sample.metadata, ensure_ascii=False, indent=2)
            )

        return task_dir

    def write_dataset(self, samples: List[TaskSample]) -> Path:
        """Write all samples to disk."""
        for sample in samples:
            self.write_sample(sample)
        return self.output_dir


# ============================================================================
#  Validator
# ============================================================================

def validate_task_data(
    first_frame: Image.Image,
    prompt: str,
    final_frame: Optional[Image.Image] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Validate task data matches standardized format.

    Args:
        first_frame: Initial state image (required).
        prompt: Natural language instruction (required).
        final_frame: Final state image (optional).
        metadata: Task metadata dict (optional).

    Returns:
        ``True`` if valid.
    """
    if first_frame is None:
        return False

    if not prompt or not prompt.strip():
        return False

    return True


def validate_task_directory(task_dir: Path) -> bool:
    """Validate task directory structure.

    Args:
        task_dir: Path to task directory.

    Returns:
        ``True`` if the required files are present.
    """
    if not task_dir.exists() or not task_dir.is_dir():
        return False

    if not (task_dir / "first_frame.png").exists():
        return False

    if not (task_dir / "prompt.txt").exists():
        return False

    return True


# ============================================================================
#  Abstract base pipeline
# ============================================================================

class BasePipeline(ABC):
    """Base class for dataset pipelines.

    Subclass this and implement :meth:`download` and :meth:`process_sample`.
    The :meth:`run` method orchestrates the full workflow:
    download → process → write.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ── 1) Download ───────────────────────────────────────────────────────

    @abstractmethod
    def download(self) -> Iterator[dict]:
        """Download / stream raw samples from the source.

        Yields:
            Raw sample dicts (structure depends on the source dataset).
        """

    # ── 2) Process ────────────────────────────────────────────────────────

    @abstractmethod
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        """Transform one raw sample into a standardized :class:`TaskSample`.

        Return ``None`` to skip invalid or unusable samples.

        Args:
            raw_sample: A single dict yielded by :meth:`download`.
            idx: Zero-based index of the sample.

        Returns:
            A :class:`TaskSample`, or ``None`` to skip.
        """

    # ── Orchestration (don't override) ────────────────────────────────────

    def run(self) -> List[TaskSample]:
        """Execute the full pipeline: download → process → write.

        Returns:
            List of successfully processed :class:`TaskSample` objects.
        """
        writer = OutputWriter(self.config.output_dir)
        samples: List[TaskSample] = []
        processed = 0

        for idx, raw in enumerate(self.download()):
            sample = self.process_sample(raw, idx)
            if sample is None:
                print(f"  Skipped sample {idx}")
                continue

            writer.write_sample(sample)
            samples.append(sample)
            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} samples...")

        print(
            f"Done! Processed {processed} samples "
            f"-> {self.config.output_dir}/{self.config.domain}_task/"
        )
        return samples


# ============================================================================
#  Orchestration — delegates to src.pipeline
# ============================================================================

def run_pipeline(config) -> List[TaskSample]:
    """Standard pipeline entry point.

    Imports and runs the custom pipeline defined in ``src.pipeline``.

    Args:
        config: A :class:`PipelineConfig` (or subclass) instance.

    Returns:
        List of successfully processed :class:`TaskSample` objects.
    """
    from src.pipeline import TaskPipeline

    pipeline = TaskPipeline(config)
    return pipeline.run()

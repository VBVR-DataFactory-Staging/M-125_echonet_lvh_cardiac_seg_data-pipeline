"""Core pipeline utilities — KEEP as-is when forking.

Two modules:
    core.download  — fetch raw data (HuggingFace, S3, etc.)
    core.pipeline  — process raw data into standardized format
"""

from .download import (
    HuggingFaceDownloader,
    download_from_s3,
    run_download,
)

from .pipeline import (
    PipelineConfig,
    TaskSample,
    BasePipeline,
    SampleProcessor,
    OutputWriter,
    convert_to_pil_image,
    numpy_to_pil,
    load_from_path,
    validate_task_data,
    validate_task_directory,
    run_pipeline,
)

__all__ = [
    # Download
    "HuggingFaceDownloader",
    "download_from_s3",
    "run_download",
    # Pipeline
    "PipelineConfig",
    "TaskSample",
    "BasePipeline",
    "SampleProcessor",
    "OutputWriter",
    "convert_to_pil_image",
    "numpy_to_pil",
    "load_from_path",
    "validate_task_data",
    "validate_task_directory",
    "run_pipeline",
]

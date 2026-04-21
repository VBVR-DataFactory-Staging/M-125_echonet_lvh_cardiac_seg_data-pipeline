"""S3 raw-data downloader for M-041."""
from pathlib import Path
from typing import Iterator, Optional

from core.download import download_from_s3


class TaskDownloader:
    """Fetches raw data from S3 and yields sample dicts.

    Sub-classes of :class:`BasePipeline` drive this via :meth:`download`.
    The yielded dict structure is task-specific and is consumed by
    :meth:`process_sample` in ``src/pipeline/pipeline.py``.
    """

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)

    def ensure_raw(self):
        """Ensure the raw data is present locally. Downloads from S3 if not."""
        if not self.raw_dir.exists() or not any(self.raw_dir.iterdir()):
            print(f"Raw data not found locally, syncing from s3://{self.config.s3_bucket}/{self.config.s3_prefix} ...")
            download_from_s3(
                bucket_name=self.config.s3_bucket,
                s3_prefix=self.config.s3_prefix,
                local_dir=self.raw_dir,
            )
        else:
            print(f"Raw data already present at {self.raw_dir}, skipping sync.")

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Yield raw sample dicts from the local raw directory.

        The exact schema is defined per-task in ``src/pipeline/pipeline.py``.
        """
        self.ensure_raw()
        yield {"raw_dir": str(self.raw_dir)}


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)

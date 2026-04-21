"""Core download module — fetch raw data from external sources.

Provides generic download utilities (HTTP public S3, HuggingFace).
The actual dataset-specific download logic lives in ``src.download``;
this module always delegates to it via :func:`run_download`.

No AWS credentials required — all downloads use public HTTP URLs.
"""

from pathlib import Path
from typing import Iterator, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
import xml.etree.ElementTree as ET


# ============================================================================
#  HuggingFace downloader
# ============================================================================

class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub into the ``raw/`` directory.

    Requires ``pip install datasets huggingface-hub`` (optional dependency).
    """

    def __init__(self, repo_id: str, split: str = "test", raw_dir: Path = Path("raw")):
        self.repo_id = repo_id
        self.split = split
        self.raw_dir = Path(raw_dir)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        from datasets import load_dataset

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {self.repo_id} (split: {self.split}) → {self.raw_dir}/")
        dataset = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=str(self.raw_dir / ".cache"),
        )

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Streaming {len(dataset)} samples...")

        for item in dataset:
            yield item


# ============================================================================
#  Public S3 download (no credentials required)
# ============================================================================

def _list_s3_public(bucket: str, prefix: str, region: str = "us-east-2") -> List[str]:
    """List objects in a public S3 bucket via the REST XML API.

    Returns a list of S3 keys under *prefix*.
    """
    base = f"https://{bucket}.s3.{region}.amazonaws.com"
    keys: List[str] = []
    continuation = None

    while True:
        url = f"{base}?list-type=2&prefix={prefix}"
        if continuation:
            url += f"&continuation-token={continuation}"

        resp = urlopen(Request(url))
        tree = ET.parse(resp)
        root = tree.getroot()
        ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

        for content in root.findall(f"{ns}Contents"):
            key = content.find(f"{ns}Key").text
            if key and not key.endswith("/"):
                keys.append(key)

        is_truncated = root.find(f"{ns}IsTruncated")
        if is_truncated is not None and is_truncated.text == "true":
            token_el = root.find(f"{ns}NextContinuationToken")
            continuation = token_el.text if token_el is not None else None
        else:
            break

    return keys


def download_from_s3(
    bucket_name: str,
    s3_prefix: str,
    local_dir: Path,
    region: str = "us-east-2",
) -> int:
    """Download dataset from a **public** S3 bucket via HTTP.

    No AWS credentials required. The bucket must allow public read.

    Args:
        bucket_name: S3 bucket name.
        s3_prefix: S3 prefix to download from.
        local_dir: Local directory to save files.
        region: AWS region (default: us-east-2).

    Returns:
        Number of files downloaded.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://{bucket_name}.s3.{region}.amazonaws.com"
    print(f"Listing files at {base_url}/{s3_prefix}...")

    keys = _list_s3_public(bucket_name, s3_prefix, region)
    print(f"Found {len(keys)} files to download...")

    downloaded = 0
    for key in keys:
        relative_path = key.replace(s3_prefix, "", 1).lstrip("/")
        if not relative_path:
            continue

        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        file_url = f"{base_url}/{key}"
        try:
            resp = urlopen(Request(file_url))
            local_path.write_bytes(resp.read())
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{len(keys)} files...")
        except URLError as e:
            print(f"  Failed: {key} ({e})")

    print(f"\n✓ Download complete: {downloaded} files")
    return downloaded


# ============================================================================
#  Orchestration — delegates to src.download
# ============================================================================

def run_download(config) -> Iterator[dict]:
    """Standard download entry point.

    Imports and calls the custom downloader defined in ``src.download``.

    Args:
        config: A :class:`PipelineConfig` (or subclass) instance.

    Yields:
        Raw sample dicts from the custom downloader.
    """
    from src.download import create_downloader

    downloader = create_downloader(config)
    yield from downloader.download(limit=config.num_samples)

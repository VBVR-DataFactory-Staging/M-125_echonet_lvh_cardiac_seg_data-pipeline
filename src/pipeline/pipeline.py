"""Task pipeline for M-041.

This module is a thin wrapper that delegates all real work to the vendored
phase2 logic in ``src/pipeline/_phase2/m041_multibypass.py``. The phase2 module
generates the standard 7-file task layout into
``datasets/_example_output/M-041_multibypass_phase_recognition/`` relative to the pipeline's
``DATA_ROOT``.

The :class:`TaskPipeline` class below exposes the standard Med-VR
BasePipeline interface (download + process_sample + run) so it stays
compatible with the wider harness.
"""
from pathlib import Path
from typing import Iterator, Optional
import sys

from core.pipeline import BasePipeline, TaskSample
from src.download.downloader import create_downloader
from src.pipeline.config import TaskConfig

# Make the vendored phase2 code importable.
_PHASE2 = Path(__file__).parent / "_phase2"
if str(_PHASE2) not in sys.path:
    sys.path.insert(0, str(_PHASE2))

import importlib
_phase2_mod = importlib.import_module("m041_multibypass")


class TaskPipeline(BasePipeline):

    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.downloader = create_downloader(self.config)

    def download(self) -> Iterator[dict]:
        yield from self.downloader.iter_samples(limit=self.config.num_samples)

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        """Delegate processing to the vendored phase2 ``main()``.

        Calls phase2 main() once on first sample; subsequent invocations
        are no-ops since phase2 writes the full output in one pass.
        """
        if idx > 0:
            return None  # phase2 main() is invoked once
        if hasattr(_phase2_mod, "main"):
            _phase2_mod.main()
        return None  # phase2 writes files directly, no TaskSample returned

    def run(self):
        # Override BasePipeline.run to just call phase2.main() directly.
        if hasattr(_phase2_mod, "main"):
            _phase2_mod.main()
        return []

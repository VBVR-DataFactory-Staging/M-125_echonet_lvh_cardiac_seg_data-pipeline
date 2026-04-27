#!/usr/bin/env python3
"""Generate M-125 EchoNet-LVH cardiac-cycle prediction task samples."""

import argparse
import os
import sys
from pathlib import Path

# bootstrap.sh redirects stdout/stderr to a log file, which makes stdout fully
# buffered — print() output never lands in S3 until Python exits. Force
# line-buffered output so we get live progress.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass
print("[generate.py] starting", flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig  # noqa: E402
print("[generate.py] imports OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="M-125 EchoNet-LVH cardiac-cycle prediction")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/questions")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    # Force empty generator so OutputWriter writes
    #   <output>/<domain>_task/<task_id>/  (matches bootstrap meta auto-detect)
    parser.add_argument("--generator", type=str, default="")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    print("Generating M-125 EchoNet-LVH cardiac-cycle dataset...", flush=True)

    kwargs = dict(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        seed=args.seed,
        start_index=args.start_index,
        generator=args.generator,
        fps=args.fps,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
    )

    config = TaskConfig(**kwargs)
    pipeline = TaskPipeline(config)
    samples = pipeline.run()
    gen_part = getattr(config, "generator", "") or ""
    print(f"Wrote {len(samples)} samples to {config.output_dir}/{gen_part}", flush=True)


if __name__ == "__main__":
    main()

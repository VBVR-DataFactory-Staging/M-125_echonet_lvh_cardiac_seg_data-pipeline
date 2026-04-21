#!/usr/bin/env python3
"""Dataset generation entry point for M-041 (surgical_phase_recognition).

Usage:
    python examples/generate.py
    python examples/generate.py --num-samples 10
    python examples/generate.py --output data/my_output
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(description="Generate M-041 dataset")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/questions")
    args = parser.parse_args()

    print("Generating M-041 (surgical_phase_recognition) dataset...")
    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
    )
    pipeline = TaskPipeline(config)
    pipeline.run()
    print("Done.")


if __name__ == "__main__":
    main()

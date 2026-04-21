# M-041 — Surgical Phase Recognition

MultiBypass140 laparoscopic Roux-en-Y surgical phase recognition.

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark. It produces standardized video-
reasoning task samples from the underlying raw medical dataset.

## Task

**Prompt shown to the model**:

> This is a laparoscopic Roux-en-Y gastric bypass surgery video. Identify the current surgical phase at every frame. The full-frame border changes color according to the phase (preparation: gray; gastric pouch creation: red; omentum division: orange; mesenteric defect closure: yellow; jejunal transection: green; jejuno-jejunostomy: cyan; gastro-jejunostomy: blue; anastomosis test: purple; cleaning and hemostasis: magenta; instrument removal: brown), and a surgical-progress bar is shown at the bottom.

## S3 Raw Data

```
s3://med-vr-datasets/M-041_MultiBypass140/raw/
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate samples (downloads raw from S3 on first run)
python examples/generate.py

# Generate only N samples
python examples/generate.py --num-samples 10

# Custom output directory
python examples/generate.py --output data/my_output
```

## Output Layout

```
data/questions/surgical_phase_recognition_task/
├── task_0000/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── first_video.mp4
│   ├── last_video.mp4
│   ├── ground_truth.mp4
│   ├── prompt.txt
│   └── metadata.json
├── task_0001/
└── ...
```

## Example Output

See [`examples/example_output/`](examples/example_output/) for 2 reference
samples committed alongside the code.

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `"surgical_phase_recognition"` | Task domain string used in output paths. |
| `s3_bucket` | `"med-vr-datasets"` | S3 bucket containing raw data. |
| `s3_prefix` | `"M-041_MultiBypass140/raw/"` | S3 key prefix for raw data. |
| `fps` | `25` | Output video FPS. |
| `raw_dir` | `Path("raw")` | Local raw cache directory. |
| `num_samples` | `None` | Max samples (None = all). |

## Repository Structure

```
M-041_multibypass_phase_recognition_data-pipeline/
├── core/                ← shared pipeline framework (verbatim)
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 raw-data downloader
│   └── pipeline/
│       ├── config.py        ← task config
│       ├── pipeline.py      ← TaskPipeline
│       ├── transforms.py    ← visualization helpers (shim)
│       └── _phase2/         ← vendored phase2 prototype logic
├── examples/
│   ├── generate.py
│   └── example_output/      ← committed reference samples
├── requirements.txt
├── README.md
└── LICENSE
```

# M-125 — EchoNet-LVH Cardiac-Cycle Prediction

Stanford AIMI **EchoNet-LVH** parasternal-long-axis echocardiograms turned into
a video-prediction task: given a single end-diastolic (ED) frame, predict the
full cardiac cycle (systolic contraction → diastolic re-expansion).

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark.

## Task design (Option B — Cycle Prediction, C5 tab)

Aligned with **M-126** (EchoNet-Pediatric segmentation/cycle) and **M-130**
(CAMUS heartbeat-cycle prediction).

**Prompt shown to the model**:

> Given this Stanford EchoNet-LVH parasternal-long-axis echocardiogram frame at
> end-diastole (left ventricle maximally filled), predict one complete cardiac
> cycle showing systolic contraction (ventricular walls thicken and the
> left-ventricular chamber shrinks) followed by diastolic relaxation (walls
> thin and the chamber re-expands). Preserve the LV chamber geometry, the
> interventricular septum (IVS) and posterior wall (LVPW), and the overall
> acoustic-window shape. The generated video should end at approximately the
> same end-diastolic state it started from. No text overlays in the generated
> video.

**Per-sample 7-file layout**:
```
echonet_lvh_cardiac_cycle_prediction_task/echonet_lvh_<NNNNN>/
├── first_frame.png      ED frame, 512x512
├── final_frame.png      Last frame of GT cycle (≈ ED again)
├── prompt.txt           the prompt above
├── first_video.mp4      ED frame held for 48 frames (seed loop)
├── last_video.mp4       == ground_truth.mp4
├── ground_truth.mp4     full cardiac cycle, 48 frames @ 16 fps
└── metadata.json        LV measurements (IVSd/LVIDd/LVIDs/LVPWd cm), split, etc.
```

## S3 Raw Data

```
s3://med-vr-datasets/M-125/echonet_lvh/echonetlvh/EchoNet-LVH.zip   (73.79 GB)
  ├── Batch1/  (3,150 .avi)
  ├── Batch2/  (2,986 .avi)
  ├── Batch3/  (2,908 .avi)
  ├── Batch4/  (2,960 .avi)
  └── MeasurementsList.csv   (per-video LV measurements + split + ED/ES frame)
```

The downloader **streams individual .avi members** via boto3 range fetch +
zlib inflate (`src/download/downloader.py`). It deliberately does **not** sync
the full 73 GB zip — EC2 disk footprint stays around `num_samples * 10 MB`.

## Pipeline notes

- ED frame is taken from `MeasurementsList.csv` (the `LVIDd` row's `Frame`).
- Cycle length is approximated as `2 * (ES - ED)` when both are present, else
  ~1 s of native FPS, clamped to [12, 60] frames.
- All output videos are 512×512 H.264 yuv420p MP4 (Wan2.2-compatible).

## Local quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Stream 3 samples from S3, write to data/questions/
python examples/generate.py --num-samples 3 --output data/questions
```

## EC2 launch (production)

Use the standard harness:
```
./.claude/skills/aws-ec2/launch M-125_echonet_lvh_cardiac_seg_data-pipeline --big
```

(Pipeline is `--big` because it reads from a 73 GB ZIP. We don't sync the zip,
so a c5.4xlarge with 500 GB EBS is plenty.)

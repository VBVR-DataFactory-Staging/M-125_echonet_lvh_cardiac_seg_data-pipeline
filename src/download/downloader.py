"""Stream-from-zip downloader for M-125 EchoNet-LVH.

The Stanford AIMI EchoNet-LVH release is a single 73.79 GB ZIP on S3:
    s3://med-vr-datasets/M-125/echonet_lvh/echonetlvh/EchoNet-LVH.zip

It contains 12,000 echocardiogram .avi files in 4 sub-folders (Batch1..Batch4)
plus a single ``MeasurementsList.csv`` with per-video LV measurements:
    HashedFileName, Calc, CalcValue, Frame, X1, X2, Y1, Y2, Frames, FPS,
    Width, Height, split

The Calc column has values: LVIDd (LV internal-diam diastole), LVIDs (systole),
IVSd (interventricular-septum diastole), LVPWd (posterior-wall diastole). The
Frame column is the frame index where that measurement was taken — LVIDd's
Frame is end-diastole (ED), LVIDs's Frame is end-systole (ES).

We avoid extracting the whole zip (~150 GB of scratch) by:
  1. Reading the Zip64 central directory via S3 range requests once.
  2. For each selected video, range-fetching just that member's local file
     header + compressed bytes, then inflating with zlib.

This makes the EC2 disk footprint ~O(num_samples * 10 MB) instead of 150 GB.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Iterator, List, Optional

import boto3


def _read_central_directory(s3, bucket: str, key: str) -> List[dict]:
    """Read the Zip64 central directory of a remote zip and return entries.

    Returns a list of dicts: {fname, csize, usize, off, method}
    """
    head = s3.head_object(Bucket=bucket, Key=key)
    total = head["ContentLength"]
    print(f"  [zip] {bucket}/{key} = {total/1e9:.2f} GB", flush=True)

    # Pull last 16 MiB to find EOCD / Zip64 EOCD.
    tail_size = min(16 * 1024 * 1024, total)
    start = total - tail_size
    tail = s3.get_object(Bucket=bucket, Key=key,
                         Range=f"bytes={start}-{total-1}")["Body"].read()

    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("ZIP EOCD record not found")
    sig, disk, cd_disk, n_this, n_total, cd_size, cd_offset, comment_len = \
        struct.unpack("<IHHHHIIH", tail[eocd_pos:eocd_pos + 22])

    # Promote to Zip64 if any field is sentinel.
    if cd_offset == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        loc_pos = tail.rfind(b"PK\x06\x07")
        if loc_pos < 0:
            raise RuntimeError("Zip64 locator not found")
        _sig, _disk_z, z64_off, _td = struct.unpack("<IIQI",
                                                     tail[loc_pos:loc_pos + 20])
        if z64_off >= start:
            z64 = tail[z64_off - start: z64_off - start + 56]
        else:
            z64 = s3.get_object(Bucket=bucket, Key=key,
                                Range=f"bytes={z64_off}-{z64_off+55}")["Body"].read()
        (_sig, _esize, _vmade, _vneed, _disk, _cd_disk,
         n_total, _ntot, cd_size, cd_offset) = struct.unpack(
            "<IQHHIIQQQQ", z64[:56])

    print(f"  [zip] {n_total} entries, CD={cd_size:,} bytes @ off {cd_offset:,}",
          flush=True)
    cd = s3.get_object(Bucket=bucket, Key=key,
                       Range=f"bytes={cd_offset}-{cd_offset+cd_size-1}")["Body"].read()

    entries: List[dict] = []
    pos = 0
    SIG = b"PK\x01\x02"
    while pos < len(cd):
        if cd[pos:pos + 4] != SIG:
            break
        (_sig, _vm, _vn, _flags, method, _mt, _md, _crc,
         csize, usize, fn_len, ex_len, cm_len, _ds,
         _int, _ext, local_off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        fname = cd[pos + 46:pos + 46 + fn_len].decode("utf-8", errors="replace")
        extra = cd[pos + 46 + fn_len:pos + 46 + fn_len + ex_len]

        real_csize, real_usize, real_off = csize, usize, local_off
        ep = 0
        while ep + 4 <= len(extra):
            tag, sz = struct.unpack("<HH", extra[ep:ep + 4])
            if tag == 0x0001:
                data = extra[ep + 4:ep + 4 + sz]
                dp = 0
                if usize == 0xFFFFFFFF and dp + 8 <= len(data):
                    real_usize = struct.unpack("<Q", data[dp:dp + 8])[0]; dp += 8
                if csize == 0xFFFFFFFF and dp + 8 <= len(data):
                    real_csize = struct.unpack("<Q", data[dp:dp + 8])[0]; dp += 8
                if local_off == 0xFFFFFFFF and dp + 8 <= len(data):
                    real_off = struct.unpack("<Q", data[dp:dp + 8])[0]
            ep += 4 + sz

        entries.append({
            "fname": fname,
            "csize": real_csize,
            "usize": real_usize,
            "off": real_off,
            "method": method,
        })
        pos += 46 + fn_len + ex_len + cm_len

    return entries


def _fetch_member(s3, bucket: str, key: str, entry: dict) -> bytes:
    """Range-fetch and inflate one zip member."""
    off = entry["off"]
    # Local file header is 30 bytes + filename + extra. Filename in CD ==
    # filename in LFH usually, but extra fields can differ — pull a
    # small buffer to read the LFH precisely.
    lfh_buf = s3.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes={off}-{off+30+1024-1}")["Body"].read()
    if lfh_buf[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local file header @ {off} for {entry['fname']}")
    (_sig, _ver, _flags, _method, _mt, _md, _crc,
     _csz, _usz, fn_len, ex_len) = struct.unpack("<IHHHHHIIIHH", lfh_buf[:30])
    data_start = off + 30 + fn_len + ex_len
    csize = entry["csize"]
    comp = s3.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes={data_start}-{data_start+csize-1}")["Body"].read()

    method = entry["method"]
    if method == 8:    # deflate
        return zlib.decompress(comp, -zlib.MAX_WBITS)
    if method == 0:    # store
        return comp
    raise RuntimeError(f"unsupported compression method {method} for {entry['fname']}")


def _load_measurements_csv(csv_bytes: bytes) -> dict:
    """Parse MeasurementsList.csv → {hashed_file_name: per-video info dict}."""
    import io
    import csv

    out: dict = {}
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8", errors="replace")))
    for row in reader:
        name = row.get("HashedFileName")
        if not name:
            continue
        rec = out.setdefault(name, {
            "frames": None,
            "fps": None,
            "width": None,
            "height": None,
            "split": None,
            "ed_frame": None,   # from LVIDd row
            "es_frame": None,   # from LVIDs row
            "ivsd": None,
            "lvidd": None,
            "lvids": None,
            "lvpwd": None,
        })
        try:
            rec["frames"] = int(row.get("Frames") or 0) or rec["frames"]
            rec["fps"] = float(row.get("FPS") or 0) or rec["fps"]
            rec["width"] = int(row.get("Width") or 0) or rec["width"]
            rec["height"] = int(row.get("Height") or 0) or rec["height"]
            rec["split"] = row.get("split") or rec["split"]

            calc = (row.get("Calc") or "").strip()
            try:
                frame = int(row.get("Frame") or -1)
            except ValueError:
                frame = -1
            try:
                cval = float(row.get("CalcValue") or "nan")
            except ValueError:
                cval = float("nan")

            if calc == "LVIDd" and frame >= 0:
                rec["ed_frame"] = frame
                rec["lvidd"] = cval
            elif calc == "LVIDs" and frame >= 0:
                rec["es_frame"] = frame
                rec["lvids"] = cval
            elif calc == "IVSd":
                rec["ivsd"] = cval
            elif calc == "LVPWd":
                rec["lvpwd"] = cval
        except Exception:
            continue
    return out


class TaskDownloader:
    def __init__(self, config):
        self.config = config
        self.s3 = boto3.client("s3", region_name="us-east-2")
        self._cd: Optional[List[dict]] = None
        self._measurements: Optional[dict] = None

    def _ensure_cd(self) -> List[dict]:
        if self._cd is None:
            cache = Path(self.config.raw_dir) / "_zip_cd_cache.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists() and cache.stat().st_size > 1000:
                self._cd = json.loads(cache.read_text())
                print(f"  [zip] loaded CD cache ({len(self._cd)} entries)", flush=True)
            else:
                self._cd = _read_central_directory(
                    self.s3, self.config.s3_bucket, self.config.s3_zip_key)
                cache.write_text(json.dumps(self._cd))
                print(f"  [zip] cached CD to {cache}", flush=True)
        return self._cd

    def _ensure_measurements(self) -> dict:
        if self._measurements is None:
            cache = Path(self.config.raw_dir) / "_measurements.json"
            if cache.exists() and cache.stat().st_size > 1000:
                self._measurements = json.loads(cache.read_text())
                print(f"  [csv] loaded measurements cache ({len(self._measurements)} videos)",
                      flush=True)
            else:
                cd = self._ensure_cd()
                csv_entry = next((e for e in cd
                                  if e["fname"].endswith("MeasurementsList.csv")), None)
                if csv_entry is None:
                    raise RuntimeError("MeasurementsList.csv not found in zip")
                csv_bytes = _fetch_member(
                    self.s3, self.config.s3_bucket, self.config.s3_zip_key, csv_entry)
                self._measurements = _load_measurements_csv(csv_bytes)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(self._measurements))
                print(f"  [csv] parsed {len(self._measurements)} videos, cached", flush=True)
        return self._measurements

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Yield raw-sample dicts. Each sample has its .avi already extracted
        to a temp file path (caller must clean up via the temp dir lifecycle)."""
        cd = self._ensure_cd()
        meas = self._ensure_measurements()

        # Build hash → cd entry map (skip directories + the csv).
        avi_by_hash: dict = {}
        for e in cd:
            n = e["fname"]
            if not n.endswith(".avi"):
                continue
            stem = Path(n).stem  # e.g. "0X1027077445AE5512"
            avi_by_hash[stem] = e

        # Iterate videos with valid measurements (need ED frame).
        max_samples = int(getattr(self.config, "max_samples", 300))
        if limit is not None:
            max_samples = min(max_samples, limit)

        # Stable order: sort by hash so reruns pick the same samples.
        candidate_hashes = sorted(h for h in meas.keys() if h in avi_by_hash)
        print(f"  [m125] {len(candidate_hashes)} videos with measurements + AVI present",
              flush=True)

        tmp_root = Path(tempfile.mkdtemp(prefix="m125_avi_"))
        yielded = 0
        for h in candidate_hashes:
            if yielded >= max_samples:
                break
            entry = avi_by_hash[h]
            rec = meas[h]
            # Sanity: need an ED frame. Fall back to frame 0 if missing.
            ed_frame = rec.get("ed_frame")
            if ed_frame is None or ed_frame < 0:
                ed_frame = 0

            try:
                avi_bytes = _fetch_member(self.s3, self.config.s3_bucket,
                                          self.config.s3_zip_key, entry)
            except Exception as exc:
                print(f"  [m125] skip {h}: fetch failed: {exc}", flush=True)
                continue

            local_avi = tmp_root / f"{h}.avi"
            local_avi.write_bytes(avi_bytes)

            yield {
                "hashed_name": h,
                "avi_path": str(local_avi),
                "ed_frame": int(ed_frame),
                "es_frame": rec.get("es_frame"),
                "frames": rec.get("frames"),
                "fps_native": rec.get("fps"),
                "width_native": rec.get("width"),
                "height_native": rec.get("height"),
                "split": rec.get("split"),
                "ivsd_cm": rec.get("ivsd"),
                "lvidd_cm": rec.get("lvidd"),
                "lvids_cm": rec.get("lvids"),
                "lvpwd_cm": rec.get("lvpwd"),
                "zip_member": entry["fname"],
            }
            yielded += 1


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)

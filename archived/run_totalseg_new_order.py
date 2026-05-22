#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator (breasts) on *_TDM.nii.gz files."
    )
    parser.add_argument("input_root", type=Path, help="Root folder containing subject_xxx subfolders.")
    parser.add_argument("output_root", type=Path, help="Folder to store breast masks only.")
    parser.add_argument("--device", default="gpu:0", help='Device for TotalSegmentator, e.g., "gpu:0" or "cpu".')
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast mode.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects whose mask already exists.")
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    tdm_files = sorted(input_root.rglob("*_TDM.nii.gz"))

    if not tdm_files:
        print(f"No *_TDM.nii.gz files found under: {input_root}")
        return

    print(f"Found {len(tdm_files)} TDM file(s). Starting segmentation...\n")

    for tdm_path in tdm_files:
        subject_id = tdm_path.parent.name
        output_mask_path = output_root / f"{subject_id}_TDM_breast_mask.nii.gz"

        if args.skip_existing and output_mask_path.exists():
            print(f"[SKIP] {subject_id}: mask already exists.")
            continue

        tmp_out_dir = output_root / f"{subject_id}_tmp"
        tmp_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN ] {subject_id}: {tdm_path.name}")

        cmd = [
            "TotalSegmentator",
            "-i", str(tdm_path),
            "-o", str(tmp_out_dir),
            "-ta", "breasts",
            "--device", args.device,
            "--statistics",
        ]
        if args.fast:
            cmd.insert(4, "--fast")

        try:
            subprocess.run(cmd, check=True)

            # Move the breast mask to final destination
            breast_files = list(tmp_out_dir.glob("*breast*.nii.gz"))
            if breast_files:
                shutil.move(str(breast_files[0]), output_mask_path)
                print(f"[DONE] Saved breast mask to {output_mask_path}")
            else:
                print(f"[WARN] No breast mask found for {subject_id}")

        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {subject_id}: {e}")
        finally:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)

    print("All done.")

if __name__ == "__main__":
    main()

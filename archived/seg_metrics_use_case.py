#!/usr/bin/env python3
"""
metrics_use_case.py

CLI wrapper around segmentation_metrics.py to compute Dice, HD, and HD95 across two folders.
"""

import argparse
from src.seg_metrics import compute_metrics_batch


def main():
    ap = argparse.ArgumentParser(
        description="Compute mean Dice, HD, and HDp (default p=95) across two folders of NIfTI masks."
    )
    ap.add_argument("folder_a", help="Folder A (e.g., ground truth)")
    ap.add_argument("folder_b", help="Folder B (e.g., predictions)")
    ap.add_argument("--percentile", type=int, default=95, help="Percentile for robust Hausdorff (e.g., 95)")
    ap.add_argument("--include-background", action="store_true", help="Include background channel in metrics")
    ap.add_argument("--csv", default=None, help="Optional path to save per-case metrics CSV")
    args = ap.parse_args()

    _, summary = compute_metrics_batch(
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        include_background=args.include_background,
        hd_percentile=args.percentile,
        save_csv=args.csv,
    )

    print("\n==== Summary ====")
    print(f"Cases: {summary['n']}")
    print(f"Dice:  mean={summary['dice_mean']:.4f},  std={summary['dice_std']:.4f}")
    print(f"HD:    mean={summary['hd_mean']:.2f} mm, std={summary['hd_std']:.2f} mm")
    print(f"HD{summary['hd_percentile']}:  mean={summary['hdp_mean']:.2f} mm, std={summary['hdp_std']:.2f} mm")


if __name__ == "__main__":
    main()

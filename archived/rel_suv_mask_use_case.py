# rel_suv_mask_use_case.py
import argparse
from pathlib import Path
from src.rel_suv_mask import generate_masks_with_breast_roi

def main():
    p = argparse.ArgumentParser(
        description="Create PET tumor masks by thresholding SUV inside mandatory TDM-derived breast masks."
    )
    p.add_argument("output_root", type=Path,
                   help="Root containing <subj_id> folders with <subj_id>_TEP_SUV.nii.gz inside.")
    p.add_argument("breast_masks_root", type=Path,
                   help="Folder (can be nested) containing <subj_id>_TDM_breast_mask.nii.gz files.")
    p.add_argument("out_masks_dir", type=Path,
                   help="Destination folder for PET tumor masks (named <subj_id>_tumor_mask.nii.gz).")

    p.add_argument("--relative-threshold", type=float, default=0.45,
                   help="Fraction of SUVmax used for thresholding inside the breast ROI (default: 0.45).")
    p.add_argument("--keep", choices=["largest", "all"], default="largest",
                   help="Keep only the largest component or all components (default: largest).")
    p.add_argument("--min-volume-ml", type=float, default=0.0,
                   help="Remove components smaller than this volume (in mL). Default: 0 (disabled).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing tumor masks if present.")
    p.add_argument("--log", type=Path, default=Path("pet_tumor_masking_log.txt"),
                   help="Path to a log file capturing successes and skips.")

    args = p.parse_args()

    generate_masks_with_breast_roi(
        output_root=args.output_root,
        breast_masks_root=args.breast_masks_root,
        out_masks_dir=args.out_masks_dir,
        log_path=args.log,
        relative_threshold=args.relative_threshold,
        keep=args.keep,
        min_volume_ml=args.min_volume_ml,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()

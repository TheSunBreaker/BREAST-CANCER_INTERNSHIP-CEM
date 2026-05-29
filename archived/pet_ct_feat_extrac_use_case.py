"""
run_use_case.py

Scans a route like:

output_root/
  subject_001/
    subject_001_IRM.nii.gz
    subject_001_TEP.nii.gz
    subject_001_TEP_SUV.nii.gz
    subject_001_TDM.nii.gz
  subject_002/
    ...

Plus two mask folders:
  - breast masks in CT space: <SUBJ_ID>_TDM_breast_mask.nii.gz
  - tumor masks in PET-SUV space: <SUBJ_ID>_tumor_mask.nii  (accepts .nii or .nii.gz)

Resamples all to 1×1×1 mm³, aligns CT+breast+tumor to PET-isotropic reference, and writes CSV + Excel.

Example:
python pet_ct_feat_extrac_use_case.py \
  --route-root /data/output_root \
  --breast-masks /data/breast_masks \
  --tumor-masks /data/tumor_masks \
  --out-csv /data/features.csv \
  --out-xlsx /data/features.xlsx
"""
import os
import argparse
from typing import List, Dict

#from src.pet_ct_feat_extrac import (
from src.pet_ct_feat_extrac_v2 import (
    load_nifti,
    case_features,
    save_dataset,
    discover_subjects,
)

def run_batch(args) -> List[Dict]:
    cases = discover_subjects(
        route_root=args.route_root,
        breast_masks_dir=args.breast_masks,
        tumor_masks_dir=args.tumor_masks,
        ct_suffix=args.ct_suffix,
        pet_suv_suffix=args.pet_suv_suffix,
    )
    if not cases:
        raise RuntimeError("No valid subjects found. Check filenames and mask folders.")

    rows: List[Dict] = []
    for c in cases:
        ct_img, _, _ = load_nifti(c["ct"])
        br_img, _, _ = load_nifti(c["breast"])
        pet_img, _, _ = load_nifti(c["pet"])
        tu_img, _, _ = load_nifti(c["tumor"])

        row = case_features(
            case_id=c["case_id"],
            pet_img=pet_img,
            ct_img=ct_img,
            breast_mask_img=br_img,
            tumor_mask_img=tu_img,
            ring_mm_1=args.ring_mm_1,
            ring_mm_2=args.ring_mm_2,
            enable_pyradiomics=not args.no_pyradiomics,
            pet_binwidth_suv=args.pet_binwidth_suv,
            ct_binwidth_hu=args.ct_binwidth_hu,
            enable_log=args.enable_log,
            enable_wavelet=args.enable_wavelet
        )
        rows.append(row)
        print(f"[OK] {c['case_id']} processed ({len(row)} features).")
    return rows

def parse_args():
    ap = argparse.ArgumentParser(description="PET/CT isotropic (1mm) feature extraction -> CSV + Excel")

    # required inputs
    ap.add_argument("--route-root", type=str, required=True, help="Root with per-subject subfolders.")
    ap.add_argument("--breast-masks", type=str, required=True, help="Folder with CT-space breast masks (<SUBJ_ID>_TDM_breast_mask.nii.gz).")
    ap.add_argument("--tumor-masks", type=str, required=True, help="Folder with PET-SUV-space tumor masks (<SUBJ_ID>_tumor_mask.nii).")

    # filename suffixes inside each subject folder
    ap.add_argument("--ct-suffix", type=str, default="_TDM", help="CT filename suffix (default: _TDM).")
    ap.add_argument("--pet-suv-suffix", type=str, default="_TEP_SUV", help="PET SUV filename suffix (default: _TEP_SUV).")

    # rings
    ap.add_argument("--ring-mm-1", type=float, default=5.0, help="First ring outer radius (mm), inner=0.")
    ap.add_argument("--ring-mm-2", type=float, default=10.0, help="Second ring outer radius (mm), inner=ring-mm-1.")

    # PyRadiomics options
    ap.add_argument("--no-pyradiomics", action="store_true", help="Disable PyRadiomics features.")
    ap.add_argument("--pet-binwidth-suv", type=float, default=0.25, help="PyRadiomics binWidth for PET (SUV).")
    ap.add_argument("--ct-binwidth-hu", type=float, default=25.0, help="PyRadiomics binWidth for CT (HU).")
    ap.add_argument("--enable-log", action="store_true", help="Also extract LoG features (sigma=1,2).")
    ap.add_argument("--enable-wavelet", action="store_true", help="Also extract Wavelet features.")

    # outputs
    ap.add_argument("--out-csv", type=str, required=True, help="Output CSV path.")
    ap.add_argument("--out-xlsx", type=str, required=True, help="Output Excel path (.xlsx).")

    args = ap.parse_args()

    # Validate dirs
    for d in [args.route_root, args.breast_masks, args.tumor_masks]:
        if not os.path.isdir(d):
            ap.error(f"Not a directory: {d}")

    return args

def main():
    args = parse_args()
    rows = run_batch(args)
    save_dataset(rows, args.out_csv, args.out_xlsx)
    print(f"[DONE] Saved:\n  CSV  -> {args.out_csv}\n  XLSX -> {args.out_xlsx}")

if __name__ == "__main__":
    main()

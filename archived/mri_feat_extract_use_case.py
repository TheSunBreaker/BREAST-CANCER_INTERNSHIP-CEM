#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
use_case_irm_radiomics.py
Example use-case that IMPORTS the library (irm_radiomics.py) and runs the pipeline.
"""

from pathlib import Path
import argparse
from src.mri_feat_extract import extract_for_dataset

def run_use_case(
    output_root: Path,
    nnunet_root: Path,
    out_dir: Path,
    peri_mm: float = 5.0,
    save_peri_masks: bool = False,
    jobs: int | None = None,
    bin_width: float = 25.0,
    normalize: bool = True
):
    df_feats, df_errs = extract_for_dataset(
        output_root=output_root,
        nnunet_root=nnunet_root,
        out_dir=out_dir,
        peri_mm=peri_mm,
        save_peri_masks=save_peri_masks,
        n_jobs=jobs,
        extractor_params={"binWidth": bin_width, "normalize": normalize, "label": 1}
    )
    return df_feats, df_errs


def main():
    ap = argparse.ArgumentParser(
        description="Use-case runner: IRM radiomics with nnU-Net test masks."
    )
    ap.add_argument("--output_root", required=True, type=Path,
                    help="Root with subject_xxx folders containing *_IRM.nii.gz")
    ap.add_argument("--nnunet_root", required=True, type=Path,
                    help="Root that contains 'nnunetv2/nnUNet_raw/Dataset001/imagesTs_pred3dfullres'")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Directory to write CSV/XLSX and (optionally) peri masks")
    ap.add_argument("--peri_mm", default=5.0, type=float,
                    help="Peritumoral ring thickness in mm (default 5.0)")
    ap.add_argument("--save_peri_masks", action="store_true",
                    help="If set, saves peritumoral masks as NIfTI")
    ap.add_argument("--jobs", default=None, type=int,
                    help="Parallel jobs (default: CPU-1)")
    ap.add_argument("--bin_width", default=25.0, type=float,
                    help="PyRadiomics binWidth (default 25)")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable PyRadiomics intensity normalization")

    args = ap.parse_args()

    run_use_case(
        output_root=args.output_root,
        nnunet_root=args.nnunet_root,
        out_dir=args.out_dir,
        peri_mm=args.peri_mm,
        save_peri_masks=args.save_peri_masks,
        jobs=args.jobs,
        bin_width=args.bin_width,
        normalize=not args.no_normalize
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
irm_radiomics.py
Reusable radiomics extractor:
- Finds *_IRM.nii.gz under output_root/subject_xxx/
- Pairs with nnU-Net test predictions at:
  <nnunet_root>/nnunetv2/nnUNet_raw/Dataset001/imagesTs_pred3dfullres/<subject_id>.nii.gz
- Extracts tumor + peritumoral ring features (PyRadiomics)
- Saves CSV + Excel (+ optional peri masks)
"""

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from radiomics import featureextractor, logger as pyr_logger
pyr_logger.setLevel("WARNING")  # only warnings/errors from PyRadiomics


# --------------------------- Discovery utils ---------------------------

def find_subjects_with_irm(output_root: Path) -> Dict[str, Path]:
    """
    Recursively find any *_IRM.nii.gz under output_root, regardless of folder name.
    Returns {subject_id: irm_path}, where subject_id is taken from the filename stem.
    Example: AUBCE_IRM.nii.gz -> subject_id = 'AUBCE'
    """
    mapping: Dict[str, Path] = {}
    for irm_path in output_root.rglob("*_IRM.nii.gz"):
        # e.g., "AUBCE_IRM.nii.gz" -> "AUBCE_IRM" -> "AUBCE"
        stem = irm_path.stem  # "AUBCE_IRM"
        if stem.endswith("_IRM"):
            subject_id = stem[:-4]
        else:
            # fallback if the pattern is odd
            subject_id = stem.split("_IRM")[0]
        mapping[subject_id] = irm_path
    return mapping


def seg_path_for_subject(nnunet_root: Path, subject_id: str) -> Optional[Path]:
    """
    Build expected segmentation path for a subject_id.
    """
    seg_dir = nnunet_root / "nnunetv2" / "nnUNet_raw" / "Dataset001" / "imagesTs_pred3dfullres"
    seg_path = seg_dir / f"{subject_id}.nii.gz"
    return seg_path if seg_path.exists() else None


# --------------------------- Mask helpers ---------------------------

def binary_peri_ring(mask_img: sitk.Image, dilation_mm: float, label: int = 1) -> sitk.Image:
    """
    Create peritumoral ring mask = Dilate(mask==label, dilation_mm) - original(mask==label).
    """
    spacing = mask_img.GetSpacing()  # (x, y, z)
    radius_vox = [int(math.ceil(dilation_mm / s)) for s in spacing]

    bin_mask = sitk.BinaryThreshold(mask_img, lowerThreshold=label, upperThreshold=label,
                                    insideValue=1, outsideValue=0)

    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radius_vox)
    dilate.SetForegroundValue(1)
    dilated = dilate.Execute(bin_mask)

    peri = sitk.Subtract(dilated, bin_mask)
    peri = sitk.Clamp(peri, lowerBound=0, upperBound=1)
    return peri


# --------------------------- PyRadiomics config ---------------------------

def make_extractor(binWidth: float = 25.0, normalize: bool = True, label: int = 1) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Configure a compact PyRadiomics extractor (first-order + common textures).
    """
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=binWidth,
        normalize=normalize,
        interpolator='sitkBSpline',
        label=label
    )
    ext.disableAllFeatures()
    ext.enableFeaturesByName(firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[])
    return ext


def execute_extract(extractor, img: sitk.Image, mask: sitk.Image, prefix: str) -> Dict[str, float]:
    """
    Run extractor and prefix feature names. Filter diagnostics.
    """
    mask.CopyInformation(img)
    feats = extractor.execute(img, mask)

    out = {}
    for k, v in feats.items():
        if k.startswith("diagnostics"):
            continue
        name = f"{prefix}_{re.sub(r'^original_', '', k)}"
        try:
            out[name] = float(v)
        except Exception:
            continue
    return out


# --------------------------- Worker ---------------------------

def _process_one(args) -> Tuple[str, Dict[str, float], Optional[str]]:
    """
    Worker: returns (subject_id, features, error_msg)
    """
    (subject_id, irm_path, seg_path, peri_mm, save_peri_dir, extractor_params) = args
    try:
        img = sitk.ReadImage(str(irm_path))
        mask_raw = sitk.ReadImage(str(seg_path))

        # Align mask to image grid if needed
        if (img.GetSize() != mask_raw.GetSize()
            or img.GetSpacing() != mask_raw.GetSpacing()
            or img.GetOrigin()  != mask_raw.GetOrigin()
            or img.GetDirection()!= mask_raw.GetDirection()):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(sitk.Transform())
            mask = resampler.Execute(mask_raw)
        else:
            mask = mask_raw

        # Binarize any positive label -> 1
        mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535,
                                    insideValue=1, outsideValue=0)

        peri = binary_peri_ring(mask, dilation_mm=peri_mm, label=1)

        if save_peri_dir is not None:
            save_peri_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(peri, str(save_peri_dir / f"{subject_id}_peri_mask.nii.gz"))

        extractor = make_extractor(**extractor_params)
        feats_tumor = execute_extract(extractor, img, mask, "tumor")
        feats_peri  = execute_extract(extractor, img, peri, "peri")

        feats = {"subject_id": subject_id}
        feats.update(feats_tumor)
        feats.update(feats_peri)

        return subject_id, feats, None

    except Exception as e:
        return subject_id, {}, f"{type(e).__name__}: {e}"


# --------------------------- Orchestrator ---------------------------

def extract_for_dataset(
    output_root: Path,
    nnunet_root: Path,
    out_dir: Path,
    peri_mm: float = 5.0,
    save_peri_masks: bool = False,
    n_jobs: Optional[int] = None,
    extractor_params: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finds IRM images + nnUNet masks, extracts features in parallel, and writes CSV/XLSX.
    Returns: (df_results, df_errors)
    """
    extractor_params = extractor_params or {"binWidth": 25.0, "normalize": True, "label": 1}

    irm_map = find_subjects_with_irm(output_root)
    tasks = []
    for sid, irm_path in irm_map.items():
        segp = seg_path_for_subject(nnunet_root, sid)
        if segp is not None:
            tasks.append(
                (sid, irm_path, segp, peri_mm,
                 (out_dir / "peritumoral_masks") if save_peri_masks else None,
                 extractor_params)
            )

    if not tasks:
        raise RuntimeError("No (IRM image, segmentation) pairs found. Check paths and naming.")

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    errors: List[Dict[str, str]] = []

    with Pool(processes=n_jobs) as pool:
        for subject_id, feats, err in tqdm(pool.imap_unordered(_process_one, tasks),
                                           total=len(tasks), desc="Extracting"):
            if err is None:
                results.append(feats)
            else:
                errors.append({"subject_id": subject_id, "error": err})

    df_results = pd.DataFrame(results).sort_values("subject_id") if results else pd.DataFrame()
    df_errors  = pd.DataFrame(errors).sort_values("subject_id") if errors else pd.DataFrame()

    csv_path  = out_dir / "radiomics_results.csv"
    xlsx_path = out_dir / "radiomics_results.xlsx"
    meta_path = out_dir / "run_metadata.json"

    df_results.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="features", index=False)
        if not df_errors.empty:
            df_errors.to_excel(writer, sheet_name="errors", index=False)

    meta = {
        "n_subjects_input": len(irm_map),
        "n_pairs_processed": len(results),
        "n_errors": len(errors),
        "peri_ring_mm": peri_mm,
        "extractor_params": extractor_params
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:\n- {csv_path}\n- {xlsx_path}\n- {meta_path}")
    if save_peri_masks:
        print(f"- Peri masks in: {out_dir / 'peritumoral_masks'}")

    return df_results, df_errors

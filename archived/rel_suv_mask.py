# src/rel_suv_mask.py
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Literal

def _voxel_volume_ml(img: sitk.Image) -> float:
    sx, sy, sz = img.GetSpacing()
    return float(sx) * float(sy) * float(sz) / 1000.0  # mm^3 -> mL

def _largest_component(bin_img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(bin_img)
    if arr.max() == 0:
        return bin_img
    cc = sitk.ConnectedComponent(bin_img)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    out = rel == 1
    out = sitk.Cast(out, sitk.sitkUInt8)  # 0/1
    out.CopyInformation(bin_img)
    return out

def _remove_small_components(lbl_img: sitk.Image, min_volume_ml: float) -> sitk.Image:
    """
    lbl_img can be binary or labeled. We will remove any connected component
    whose volume < min_volume_ml and return a binary image.
    """
    if min_volume_ml <= 0:
        # force binary
        return sitk.Cast(lbl_img > 0, sitk.sitkUInt8)

    # Ensure labeled connectivity
    if lbl_img.GetPixelID() != sitk.sitkUInt32 and sitk.GetArrayFromImage(lbl_img).max() in (0, 1):
        cc = sitk.ConnectedComponent(lbl_img > 0)
    else:
        cc = lbl_img

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    vx_ml = _voxel_volume_ml(cc)

    cc_arr = sitk.GetArrayFromImage(cc)
    keep = np.zeros_like(cc_arr, dtype=np.uint8)

    for lab in stats.GetLabels():
        vox = stats.GetNumberOfPixels(lab)
        vol_ml = vox * vx_ml
        if vol_ml >= min_volume_ml:
            keep[cc_arr == lab] = 1

    out = sitk.GetImageFromArray(keep)
    out.CopyInformation(cc)
    return out

def generate_masks_with_breast_roi(
    output_root: Path,
    breast_masks_root: Path,
    out_masks_dir: Path,
    log_path: Path,
    relative_threshold: float = 0.45,
    keep: Literal["largest", "all"] = "largest",
    min_volume_ml: float = 0.0,
    overwrite: bool = False,
):
    """
    For each <subj_id>/ under output_root:
      - read PET at <subj_id>/<subj_id>_TEP_SUV.nii.gz
      - read CT breast mask at breast_masks_root/<subj_id>_TDM_breast_mask.nii.gz
        (fallback: search recursively for that exact filename)
      - resample breast mask to PET space (NN)
      - compute SUVmax in ROI, threshold at relative_threshold*SUVmax
      - keep largest/all; remove tiny comps < min_volume_ml
      - save tumor mask to out_masks_dir/<subj_id>_tumor_mask.nii.gz
      - log outcomes + summary
    """
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("========= PET Tumor Masking (SUV relative threshold in breast ROI) =========\n")
        log_file.write(f"output_root       : {output_root}\n")
        log_file.write(f"breast_masks_root : {breast_masks_root}\n")
        log_file.write(f"out_masks_dir     : {out_masks_dir}\n")
        log_file.write(f"params            : rel_thr={relative_threshold}, keep={keep}, min_vol_ml={min_volume_ml}, overwrite={overwrite}\n\n")

        subjects = [p for p in output_root.iterdir() if p.is_dir()]
        if not subjects:
            log_file.write("[INFO] No subject folders found under output_root.\n")
            print(f"No subject folders under: {output_root}")
            return

        n_total = n_ok = n_skip_pet = n_skip_mask = n_skip_empty_roi = n_exists = n_err = 0

        for subj_dir in sorted(subjects):
            subj_id = subj_dir.name
            pet_path = subj_dir / f"{subj_id}_TEP_SUV.nii.gz"
            out_mask_path = out_masks_dir / f"{subj_id}_tumor_mask.nii.gz"

            # PET check
            if not pet_path.exists():
                log_file.write(f"[SKIP] {subj_id}: Missing PET SUV file -> {pet_path}\n")
                n_skip_pet += 1
                continue

            # TDM mask check (exact path first)
            mask_exact = breast_masks_root / f"{subj_id}_TDM_breast_mask.nii.gz"
            if mask_exact.exists():
                breast_mask_path = mask_exact
            else:
                # fallback: nested search for that exact filename
                hits = list(breast_masks_root.rglob(f"{subj_id}_TDM_breast_mask.nii.gz"))
                if hits:
                    breast_mask_path = sorted(hits)[0]
                else:
                    log_file.write(f"[SKIP] {subj_id}: Missing TDM breast mask -> {mask_exact}\n")
                    n_skip_mask += 1
                    continue

            if out_mask_path.exists() and not overwrite:
                log_file.write(f"[SKIP] {subj_id}: Output exists -> {out_mask_path}\n")
                n_exists += 1
                continue

            try:
                # Read images
                pet = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)
                breast_mask = sitk.ReadImage(str(breast_mask_path), sitk.sitkUInt8)

                # Resample breast mask to PET space (identity transform, NN)
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(pet)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(sitk.Transform())
                resampler.SetDefaultPixelValue(0)
                breast_mask_pet = resampler.Execute(breast_mask)

                mask_np = sitk.GetArrayFromImage(breast_mask_pet)
                if np.count_nonzero(mask_np) == 0:
                    log_file.write(f"[SKIP] {subj_id}: Breast mask empty after resampling\n")
                    n_skip_empty_roi += 1
                    continue

                # --- Local-peak thresholding inside breast ROI ---

                    # PET within ROI
                    pet_np = sitk.GetArrayFromImage(pet)
                    breast_pet = np.where(mask_np > 0, pet_np, 0.0)

                    positive = breast_pet[breast_pet > 0]
                    if positive.size == 0:
                        log_file.write(f"[SKIP] {subj_id}: No positive SUV within breast ROI\n")
                        n_skip_empty_roi += 1
                        continue

                    # 1) Seed lesions with a looser global fraction to catch colder lesions
                    seed_frac = 0.30  # you can tune this (e.g., 0.25–0.35)
                    global_max = float(positive.max())
                    seed_thr = seed_frac * global_max

                    seed_np = ((breast_pet >= seed_thr) & (mask_np > 0)).astype(np.uint8)
                    seed_img = sitk.GetImageFromArray(seed_np)
                    seed_img.CopyInformation(pet)

                    # 2) Connected components on the seed to get candidate lesions
                    cc = sitk.ConnectedComponent(seed_img)
                    cc_arr = sitk.GetArrayFromImage(cc)
                    labels = np.unique(cc_arr)
                    labels = labels[labels != 0]

                    # 3) For each candidate, refine with  (relative_threshold × local_peak)
                    tumor_final_np = np.zeros_like(cc_arr, dtype=np.uint8)
                    for lab in labels:
                        region = (cc_arr == lab)
                        local_peak = float(breast_pet[region].max())
                        local_thr = float(relative_threshold) * local_peak
                        refined = ((breast_pet >= local_thr) & region).astype(np.uint8)
                        tumor_final_np |= refined  # union of refined lesions

                    tumor_img = sitk.GetImageFromArray(tumor_final_np)
                    tumor_img.CopyInformation(pet)

                    # keep these handy for logging later
                    _suv_global_max = global_max
                    _seed_thr_used = seed_thr


                # Keep largest/all
                if keep == "largest":
                    tumor_img = _largest_component(tumor_img)
                else:
                    tumor_img = sitk.Cast(tumor_img > 0, sitk.sitkUInt8)

                # Filter by min volume
                if min_volume_ml > 0:
                    tumor_img = _remove_small_components(tumor_img, min_volume_ml)

                # Save
                sitk.WriteImage(tumor_img, str(out_mask_path))

                # Log basic metrics
                voxels = int((sitk.GetArrayFromImage(tumor_img) > 0).sum())
                vol_ml = voxels * _voxel_volume_ml(tumor_img)
                # count lesions in the final mask (after keep/filter steps)
                n_lesions = int(sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(tumor_img))).max())
                log_file.write(
                    f"[OK] {subj_id}: mode=local-peak, global_max={_suv_global_max:.3f}, "
                    f"seed_frac=0.30, rel_thr={relative_threshold:.2f}, lesions={n_lesions}, "
                    f"vox={voxels}, vol={vol_ml:.3f} mL -> {out_mask_path}\n"
                )
                n_ok += 1

            except Exception as e:
                log_file.write(f"[ERROR] {subj_id}: {e}\n")
                n_err += 1

        # Summary
        log_file.write("\n---------------- Summary ----------------\n")
        log_file.write(f"Subjects scanned   : {n_total or len(subjects)}\n")
        log_file.write(f"Masked OK          : {n_ok}\n")
        log_file.write(f"Skipped PET missing: {n_skip_pet}\n")
        log_file.write(f"Skipped mask miss. : {n_skip_mask}\n")
        log_file.write(f"Skipped empty ROI  : {n_skip_empty_roi}\n")
        log_file.write(f"Skipped exists     : {n_exists}\n")
        log_file.write(f"Errors             : {n_err}\n")
        log_file.write("-----------------------------------------\n")

    print(f"Done. Log written to {log_path}")

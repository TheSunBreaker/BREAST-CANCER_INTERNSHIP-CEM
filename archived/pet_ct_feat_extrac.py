"""
features_pipeline.py

End-to-end PET/CT feature extractor with isotropic resampling to 1×1×1 mm³,
aligned to the PET-isotropic reference grid.
- Images (PET SUV, CT) -> linear interpolation
- Masks (breast, tumor) -> nearest neighbor
- NumPy 'clinical' features (MTV/TLG, first-order, shape, asymmetries)
- PyRadiomics (IBSI-style) for tumor and peritumoral rings on PET & CT
- Subject discovery for routes like:

output_root/
  subject_001/
    subject_001_IRM.nii.gz
    subject_001_TEP.nii.gz
    subject_001_TEP_SUV.nii.gz  (PET in SUV)  <-- used
    subject_001_TDM.nii.gz      (CT)          <-- used

External mask folders:
- breast masks (CT space): <SUBJ_ID>_TDM_breast_mask.nii.gz
- tumor masks (PET-SUV space): <SUBJ_ID>_tumor_mask.nii  (also accepts .nii.gz)

Outputs: CSV + Excel with features.

Dependencies:
  pip install numpy pandas nibabel scipy SimpleITK pyradiomics xlsxwriter
"""

from __future__ import annotations
import os
import glob
import math
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel import processing as nibproc
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import distance_transform_edt, label


# --------------------------
# I/O
# --------------------------

def load_nifti(path: str):
    """Load NIfTI and return (nib image, float32 array, spacing)."""
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    sp = tuple(float(x) for x in img.header.get_zooms()[:3])
    return img, arr, sp

# --- Reorientation & resampling helpers (to 1mm and PET reference) ---

def _to_RAS(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient to RAS+ (avoids axis flips before resampling)."""
    return nib.as_closest_canonical(img)

def _resample_to_output(img: nib.Nifti1Image, spacing=(1.0,1.0,1.0), is_mask=False) -> nib.Nifti1Image:
    """Isotropic resample (linear for images, NN for masks)."""
    order = 0 if is_mask else 1
    return nibproc.resample_to_output(img, voxel_sizes=spacing, order=order)

def _resample_from_to(ref: nib.Nifti1Image, moving: nib.Nifti1Image, is_mask=False) -> nib.Nifti1Image:
    """Resample moving onto ref grid (affine + shape)."""
    order = 0 if is_mask else 1
    return nibproc.resample_from_to(moving, ref, order=order)

def isotropic_and_align_to_pet(
    pet_img: nib.Nifti1Image,
    ct_img: nib.Nifti1Image,
    breast_mask_img: nib.Nifti1Image,
    tumor_mask_img: nib.Nifti1Image,
    iso=(1.0,1.0,1.0),
):
    """
    1) Reorient all to RAS+
    2) Resample each to isotropic 'iso'
    3) Use PET-isotropic as reference grid; resample CT + masks onto it
    Returns: (pet_np, ct_np, breast_bool, tumor_bool, spacing(mm), pet_iso_ref_img)
    """
    # 1) RAS+
    pet_r  = _to_RAS(pet_img)
    ct_r   = _to_RAS(ct_img)
    br_r   = _to_RAS(breast_mask_img)
    tu_r   = _to_RAS(tumor_mask_img)

    # 2) Each to isotropic 1mm in its own space
    pet_iso_img = _resample_to_output(pet_r, spacing=iso, is_mask=False)
    ct_iso_img  = _resample_to_output(ct_r,  spacing=iso, is_mask=False)
    br_iso_img  = _resample_to_output(br_r,  spacing=iso, is_mask=True)
    tu_iso_img  = _resample_to_output(tu_r,  spacing=iso, is_mask=True)

    # 3) Align CT & masks onto PET-isotropic reference grid
    ct_on_pet = _resample_from_to(pet_iso_img, ct_iso_img, is_mask=False)
    br_on_pet = _resample_from_to(pet_iso_img, br_iso_img, is_mask=True)
    tu_on_pet = _resample_from_to(pet_iso_img, tu_iso_img, is_mask=True)

    # numpy arrays
    pet_np = pet_iso_img.get_fdata(dtype=np.float32)
    ct_np  = ct_on_pet.get_fdata(dtype=np.float32)
    br_np  = (br_on_pet.get_fdata(dtype=np.float32) > 0.5)
    tu_np  = (tu_on_pet.get_fdata(dtype=np.float32) > 0.5)

    # guarantee same shapes
    assert pet_np.shape == ct_np.shape == br_np.shape == tu_np.shape, \
        f"Aligned shapes differ: PET {pet_np.shape}, CT {ct_np.shape}, breast {br_np.shape}, tumor {tu_np.shape}"

    spacing = tuple(float(x) for x in pet_iso_img.header.get_zooms()[:3])  # (1,1,1)
    return pet_np, ct_np, br_np, tu_np, spacing, pet_iso_img


# --------------------------
# Region logic (rings/ipsi/contra)
# --------------------------

def make_ring_mask(tumor_mask: np.ndarray,
                   breast_mask: np.ndarray,
                   spacing: Tuple[float,float,float],
                   inner_mm: float,
                   outer_mm: float) -> np.ndarray:
    """Peritumoral ring in [inner_mm, outer_mm] OUTSIDE the tumor, constrained to breast."""
    outside = (~tumor_mask) & breast_mask
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    ring = breast_mask & (dist_mm > inner_mm) & (dist_mm <= outer_mm)
    return ring

def split_breasts(breast_mask: np.ndarray,
                  tumor_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split breast into ipsilateral (overlaps tumor) and contralateral."""
    labeled, n = label(breast_mask.astype(np.uint8))
    if n >= 2:
        overlaps = [np.logical_and(labeled == (i+1), tumor_mask).sum() for i in range(n)]
        ipsi_label = int(np.argmax(overlaps)) + 1
        ipsi = labeled == ipsi_label
        contra = (labeled != ipsi_label) & (labeled > 0)
        return ipsi, contra

    coords = np.argwhere(breast_mask)
    if coords.size == 0:
        return breast_mask & False, breast_mask & False
    minc = coords.min(axis=0); maxc = coords.max(axis=0)
    axis = int(np.argmax(maxc - minc))
    mid = (minc[axis] + maxc[axis]) / 2.0
    tcoords = np.argwhere(tumor_mask)

    if tcoords.size == 0:
        ipsi = breast_mask & (np.indices(breast_mask.shape)[axis] < mid)
        return ipsi, breast_mask & (~ipsi)

    tcent = tcoords.mean(axis=0)[axis]
    if tcent < mid:
        ipsi = breast_mask & (np.indices(breast_mask.shape)[axis] < mid)
    else:
        ipsi = breast_mask & (np.indices(breast_mask.shape)[axis] >= mid)
    return ipsi, breast_mask & (~ipsi)


# --------------------------
# NumPy features
# --------------------------

def voxel_volume_mm3(spacing: Tuple[float,float,float]) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])

def first_order(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = arr[mask]
    if vals.size == 0:
        return {
            f"{prefix}_n": 0, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan, f"{prefix}_p10": np.nan, f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan, f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(vals.size),
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_p10": float(np.percentile(vals, 10)),
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_p90": float(np.percentile(vals, 90)),
        f"{prefix}_max": float(vals.max()),
    }

def _surface_area_voxel(mask: np.ndarray, spacing: Tuple[float,float,float]) -> float:
    sx, sy, sz = spacing
    area = 0.0
    a01 = (~mask[1:,:,:]) & mask[:-1,:,:]; a10 = (~mask[:-1,:,:]) & mask[1:,:,:]
    area += (a01.sum() + a10.sum()) * (sy*sz)
    b01 = (~mask[:,1:,:]) & mask[:,:-1,:]; b10 = (~mask[:,:-1,:]) & mask[:,1:,:]
    area += (b01.sum() + b10.sum()) * (sx*sz)
    c01 = (~mask[:,:,1:]) & mask[:,:,:-1]; c10 = (~mask[:,:,:-1]) & mask[:,:,1:]
    area += (c01.sum() + c10.sum()) * (sx*sy)
    return float(area)

def shape_features(mask: np.ndarray, spacing: Tuple[float,float,float], prefix: str) -> Dict[str, float]:
    vv = voxel_volume_mm3(spacing)
    voxels = int(mask.sum())
    vol_mm3 = voxels * vv
    vol_ml = vol_mm3 / 1000.0
    if voxels == 0:
        return {
            f"{prefix}_voxels": 0, f"{prefix}_volume_ml": 0.0, f"{prefix}_surface_mm2": np.nan,
            f"{prefix}_sphericity": np.nan, f"{prefix}_bbox_x_mm": np.nan, f"{prefix}_bbox_y_mm": np.nan,
            f"{prefix}_bbox_z_mm": np.nan, f"{prefix}_bbox_volume_ml": np.nan,
            f"{prefix}_compactness_bbox": np.nan,
        }
    coords = np.argwhere(mask)
    minc = coords.min(axis=0); maxc = coords.max(axis=0)
    dims_vox = (maxc - minc + 1).astype(np.float32)
    dims_mm = dims_vox * np.array(spacing, dtype=np.float32)
    bbox_vol_ml = float(np.prod(dims_mm) / 1000.0)
    surface = _surface_area_voxel(mask, spacing)
    sphericity = ((math.pi ** (1/3.0)) * (6.0 * vol_mm3) ** (2/3.0)) / surface if surface > 0 else np.nan
    compact_bbox = (vol_ml / bbox_vol_ml) if bbox_vol_ml > 0 else np.nan
    return {
        f"{prefix}_voxels": voxels,
        f"{prefix}_volume_ml": float(vol_ml),
        f"{prefix}_surface_mm2": float(surface),
        f"{prefix}_sphericity": float(sphericity),
        f"{prefix}_bbox_x_mm": float(dims_mm[0]),
        f"{prefix}_bbox_y_mm": float(dims_mm[1]),
        f"{prefix}_bbox_z_mm": float(dims_mm[2]),
        f"{prefix}_bbox_volume_ml": float(bbox_vol_ml),
        f"{prefix}_compactness_bbox": float(compact_bbox),
    }

def suv_peak_3x3x3(pet: np.ndarray, tumor_mask: np.ndarray) -> float:
    if tumor_mask.sum() == 0:
        return np.nan
    t = pet.copy(); t[~tumor_mask] = -np.inf
    x,y,z = np.unravel_index(int(np.argmax(t)), t.shape)
    xs = slice(max(0, x-1), min(pet.shape[0], x+2))
    ys = slice(max(0, y-1), min(pet.shape[1], y+2))
    zs = slice(max(0, z-1), min(pet.shape[2], z+2))
    vals = pet[xs, ys, zs][tumor_mask[xs, ys, zs]]
    return float(np.mean(vals)) if vals.size else np.nan

def mtv_tlg(pet: np.ndarray, tumor_mask: np.ndarray, spacing: Tuple[float,float,float], mode: str="41pct"):
    vals = pet[tumor_mask]
    if vals.size == 0:
        return 0.0, 0.0
    thr = float(vals.max() * 0.41) if mode == "41pct" else 2.5
    sub = tumor_mask & (pet >= thr)
    if sub.sum() == 0:
        return 0.0, 0.0
    vol_ml = sub.sum() * voxel_volume_mm3(spacing) / 1000.0
    tlg = float(np.mean(pet[sub]) * vol_ml)
    return float(vol_ml), float(tlg)

def asymmetry_metric(ipsi_vals: np.ndarray, contra_vals: np.ndarray) -> float:
    if ipsi_vals.size == 0 or contra_vals.size == 0:
        return np.nan
    denom = float(np.mean(contra_vals))
    if denom == 0:
        return np.nan
    return float(np.mean(ipsi_vals) / denom)


# --------------------------
# PyRadiomics helpers (IBSI)
# --------------------------

def _np_to_sitk(img_np: np.ndarray, spacing: Tuple[float,float,float], is_mask: bool):
    import SimpleITK as sitk
    if is_mask:
        im = sitk.GetImageFromArray(img_np.astype(np.uint8))
    else:
        im = sitk.GetImageFromArray(img_np.astype(np.float32))
    im.SetSpacing(tuple(float(s) for s in spacing))
    return im

def _make_extractor(bin_width: float, enable_wavelet: bool=False, enable_log: bool=False):
    from radiomics import featureextractor
    settings = {
        "binWidth": float(bin_width),
        "normalize": False,
        "resampledPixelSpacing": None,  # already isotropic
        "interpolator": "sitkBSpline",
        "label": 1,
        "preCrop": True,
        "correctMask": True,
        "geometryTolerance": 1e-6,
    }
    extr = featureextractor.RadiomicsFeatureExtractor(**settings)
    img_types = {"Original": {}}
    if enable_log:
        img_types["LoG"] = {"sigma": [1.0, 2.0]}
    if enable_wavelet:
        img_types["Wavelet"] = {}
    extr.enableImageTypes(**img_types)
    extr.disableAllFeatures()
    for cls in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extr.enableFeatureClassByName(cls)
    return extr

def _pyrad_execute(extractor, image_np: np.ndarray, mask_bool: np.ndarray, spacing, prefix: str) -> Dict[str, float]:
    if mask_bool.sum() == 0:
        return {f"{prefix}EMPTY": 1}
    img_sitk = _np_to_sitk(image_np, spacing, is_mask=False)
    msk_sitk = _np_to_sitk(mask_bool.astype(np.uint8), spacing, is_mask=True)
    result = extractor.execute(img_sitk, msk_sitk, label=1)
    out: Dict[str, float] = {}
    for k, v in result.items():
        if str(k).startswith("diagnostics_"):
            continue
        try:
            out[f"{prefix}{k}"] = float(v)
        except Exception:
            try:
                out[f"{prefix}{k}"] = float(np.asarray(v).item())
            except Exception:
                pass
    return out


# --------------------------
# Case -> feature row
# --------------------------

def case_features(case_id: str,
                  pet_img: nib.Nifti1Image,
                  ct_img: nib.Nifti1Image,
                  breast_mask_img: nib.Nifti1Image,
                  tumor_mask_img: nib.Nifti1Image,
                  ring_mm_1: float = 5.0,
                  ring_mm_2: float = 10.0,
                  enable_pyradiomics: bool = True,
                  pet_binwidth_suv: float = 0.25,
                  ct_binwidth_hu: float = 25.0,
                  enable_log: bool = False,
                  enable_wavelet: bool = False) -> Dict[str, float]:
    """
    Full pipeline for a single case:
    1) Reorient to RAS+, resample each to 1mm, align CT+breast+tumor to PET-1mm reference
    2) Build rings + ipsi/contra
    3) Compute NumPy and PyRadiomics features
    """
    # ---- 1) Isotropic resampling + alignment to PET reference
    pet, ct, breast, tumor, spacing, _pet_ref = isotropic_and_align_to_pet(
        pet_img, ct_img, breast_mask_img, tumor_mask_img, iso=(1.0,1.0,1.0)
    )

    # Keep tumor inside breast (same grid)
    tumor = tumor & breast

    # CT stabilization for textures
    ct = np.clip(ct, -150.0, 250.0)

    # Regions
    ipsi, contra = split_breasts(breast, tumor)
    ring1 = make_ring_mask(tumor, breast, spacing, 0.0, ring_mm_1)          # 0–5 mm
    ring2 = make_ring_mask(tumor, breast, spacing, ring_mm_1, ring_mm_2)    # 5–10 mm
    breast_bg = breast & (~tumor)
    ipsi_bg = ipsi & (~tumor)
    contra_bg = contra & (~tumor)

    feats: Dict[str, float] = {"case_id": case_id}

    # ---- Shape (NumPy)
    feats.update(shape_features(tumor, spacing, prefix="tumor_shape"))

    # ---- PET (NumPy)
    feats.update(first_order(pet, tumor, "pet_tumor"))
    feats["pet_tumor_SUVpeak3x3x3"] = suv_peak_3x3x3(pet, tumor)
    mtv41, tlg41 = mtv_tlg(pet, tumor, spacing, "41pct"); feats["pet_tumor_MTV_41pct_ml"] = mtv41; feats["pet_tumor_TLG_41pct"] = tlg41
    mtv25, tlg25 = mtv_tlg(pet, tumor, spacing, "2.5");  feats["pet_tumor_MTV_2p5_ml"]  = mtv25; feats["pet_tumor_TLG_2p5"]  = tlg25
    feats.update(first_order(pet, ring1, "pet_ring_0to5mm"))
    feats.update(first_order(pet, ring2, "pet_ring_5to10mm"))
    feats.update(first_order(pet, breast_bg, "pet_breast_bg"))
    feats.update(first_order(pet, ipsi_bg, "pet_ipsi_bg"))
    feats.update(first_order(pet, contra_bg, "pet_contra_bg"))
    feats["pet_asym_ipsi_over_contra_mean"] = asymmetry_metric(pet[ipsi_bg], pet[contra_bg])
    cmean = feats.get("pet_contra_bg_mean", np.nan)
    tmean = feats.get("pet_tumor_mean", np.nan)
    feats["pet_TBR_tumorMean_over_contraMean"] = float(tmean / cmean) if (not np.isnan(cmean) and cmean != 0) else np.nan

    # ---- CT (NumPy)
    feats.update(first_order(ct, tumor, "ct_tumor_HU"))
    feats.update(first_order(ct, ring1, "ct_ring_0to5mm_HU"))
    feats.update(first_order(ct, ring2, "ct_ring_5to10mm_HU"))
    feats.update(first_order(ct, breast_bg, "ct_breast_bg_HU"))
    feats.update(first_order(ct, ipsi_bg, "ct_ipsi_bg_HU"))
    feats.update(first_order(ct, contra_bg, "ct_contra_bg_HU"))
    feats["ct_asym_ipsi_over_contra_meanHU"] = asymmetry_metric(ct[ipsi_bg], ct[contra_bg])
    cHU = feats.get("ct_contra_bg_HU_mean", np.nan)
    tHU = feats.get("ct_tumor_HU_mean", np.nan)
    feats["ct_tumor_to_contraHU_diff"] = float(tHU - cHU) if not np.isnan(cHU) else np.nan

    # ---- PyRadiomics (IBSI)
    if enable_pyradiomics:
        pet_extr = _make_extractor(pet_binwidth_suv, enable_wavelet=enable_wavelet, enable_log=enable_log)
        ct_extr  = _make_extractor(ct_binwidth_hu,  enable_wavelet=enable_wavelet, enable_log=enable_log)

        feats.update(_pyrad_execute(pet_extr, pet, tumor, spacing, prefix="pyrad_pet_tumor__"))
        feats.update(_pyrad_execute(pet_extr, pet, ring1, spacing, prefix="pyrad_pet_ring0to5__"))
        feats.update(_pyrad_execute(pet_extr, pet, ring2, spacing, prefix="pyrad_pet_ring5to10__"))

        feats.update(_pyrad_execute(ct_extr,  ct,  tumor, spacing, prefix="pyrad_ct_tumor__"))
        feats.update(_pyrad_execute(ct_extr,  ct,  ring1, spacing, prefix="pyrad_ct_ring0to5__"))
        feats.update(_pyrad_execute(ct_extr,  ct,  ring2, spacing, prefix="pyrad_ct_ring5to10__"))

    # meta
    feats["ring_inner_mm"] = float(ring_mm_1)
    feats["ring_outer_mm"] = float(ring_mm_2)
    feats["pyradiomics_enabled"] = int(enable_pyradiomics)
    feats["isotropic_spacing_mm"] = 1.0

    return feats


# --------------------------
# Subject discovery & saving
# --------------------------

def _match_file(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def discover_subjects(route_root: str,
                      breast_masks_dir: str,
                      tumor_masks_dir: str,
                      ct_suffix: str = "_TDM",
                      pet_suv_suffix: str = "_TEP_SUV") -> List[Dict[str, str]]:
    """
    Return list of dicts with keys: case_id, ct, pet, breast, tumor.
    Accepts .nii or .nii.gz files.
    """
    subs = [d for d in sorted(os.listdir(route_root)) if os.path.isdir(os.path.join(route_root, d))]
    cases: List[Dict[str, str]] = []

    for sub in subs:
        subdir = os.path.join(route_root, sub)
        subj_id = sub

        ct_path = _match_file([
            os.path.join(subdir, f"{subj_id}{ct_suffix}.nii.gz"),
            os.path.join(subdir, f"{subj_id}{ct_suffix}.nii"),
        ])
        pet_path = _match_file([
            os.path.join(subdir, f"{subj_id}{pet_suv_suffix}.nii.gz"),
            os.path.join(subdir, f"{subj_id}{pet_suv_suffix}.nii"),
        ])
        if ct_path is None or pet_path is None:
            print(f"[WARN] {subj_id}: missing CT or PET_SUV; skipping.")
            continue

        breast_path = _match_file([
            os.path.join(breast_masks_dir, f"{subj_id}_TDM_breast_mask.nii.gz"),
            os.path.join(breast_masks_dir, f"{subj_id}_TDM_breast_mask.nii"),
        ])
        tumor_path = _match_file([
            os.path.join(tumor_masks_dir, f"{subj_id}_tumor_mask.nii"),
            os.path.join(tumor_masks_dir, f"{subj_id}_tumor_mask.nii.gz"),
        ])
        if breast_path is None or tumor_path is None:
            print(f"[WARN] {subj_id}: missing masks; skipping.")
            continue

        cases.append({
            "case_id": subj_id,
            "ct": ct_path,
            "pet": pet_path,
            "breast": breast_path,
            "tumor": tumor_path,
        })

    return cases

def save_dataset(rows: List[Dict[str, float]], out_csv: str, out_xlsx: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "case_id" in df.columns:
        cols = ["case_id"] + [c for c in df.columns if c != "case_id"]
        df = df[cols]
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="features")
    return df

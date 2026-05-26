#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  Extracteur de Caractéristiques Radiomiques PET/CT (V4 - Complète & Exécutable)
===============================================================================
RÔLE :
  Extrait un tableau de bord quantitatif (Shape, SUV, CT HU, PyRadiomics) 
  à partir des images formatées pour nnU-Net.

POURQUOI UTILISER LES DONNÉES ISSUES DU TRAITEMENT DE MISE EN STRUCTURE NN-UNET COMME SOURCE ?
  En se branchant APRES la mise en structure nnU-Net, on garantit que les 
  radiomiques sont extraites sur l'exacte même géométrie et vérité terrain 
  que celle vue par le réseau de neurones. L'alignement PET/CT est déjà garanti.

TRAITEMENT INTERNE :
  Le script force un ré-échantillonnage à 2x2x2 mm (isotropique) en mémoire.
  C'est une condition sine qua non pour que les matrices de texture GLCM 
  (PyRadiomics) soient invariantes par rotation et cliniquement valides.
===============================================================================

Approximation SUVpeak basée sur un voisinage 3x3x3 voxels
sur grille isotropique 2 mm.

ATTENTION, ACTUELLEMENT, LA GLCM DISTANCE EST MISE PAR DEFAUT A 1, MAIS PLUS TARD? IL SERA INTERESSANT DE LA TESTER AVEC LES 2 ET 3 EN PLUS
"""

from __future__ import annotations
import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import label, binary_dilation, generate_binary_structure, distance_transform_edt

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
ISO_SPACING = (2.0, 2.0, 2.0)  # Résolution isométrique cible pour l'analyse de texture
DILATION_NEIGHBORHOOD = 1      # Connectivité 6 (face partagée) pour la création de l'anneau

def _crop_to_mask_bbox(arrs: List[np.ndarray], mask: np.ndarray, spacing_zyx: Tuple[float,float,float], margin_mm=12.0):
    """
    [RESTAURÉ & OPTIMISÉ]
    Réduit la taille des matrices Numpy autour de la boîte englobante du masque (ici le sein).
    Bénéfice : Réduit drastiquement la consommation de RAM et accélère PyRadiomics et les dilatations 3D.
    """
    if mask.sum() == 0:
        return arrs, (slice(0,arrs[0].shape[0]), slice(0,arrs[0].shape[1]), slice(0,arrs[0].shape[2]))
        
    idx = np.argwhere(mask)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    
    # Calcul de la marge en voxels (en utilisant l'espacement ZYX correspondant aux axes NumPy)
    vox_margin = np.array([max(1, int(round(margin_mm / s))) for s in spacing_zyx], dtype=int)
    
    lo = np.maximum(mins - vox_margin, 0)
    hi = np.minimum(maxs + vox_margin + 1, mask.shape)
    
    slicer = tuple(slice(lo[d], hi[d]) for d in range(3))
    return [a[slicer] for a in arrs], slicer

def asymmetry_metric(ipsi_vals: np.ndarray, contra_vals: np.ndarray) -> float:
    """[RESTAURÉ] Calcule le ratio d'intensité entre le côté malade et le côté sain."""
    if ipsi_vals.size == 0 or contra_vals.size == 0: 
        return np.nan
    denom = float(contra_vals.mean())
    return float(ipsi_vals.mean() / denom) if denom != 0 else np.nan

# =============================================================================
# MODULE 1 : GÉOMÉTRIE ET ALIGNEMENT (Strictement SimpleITK)
# =============================================================================
def isotropic_and_align_to_pet(
    pet_img: sitk.Image,
    ct_img: sitk.Image,
    breast_mask_img: sitk.Image,
    tumor_mask_img: sitk.Image,
    iso_spacing: tuple = ISO_SPACING
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image, tuple]:
    """
    Crée une grille spatiale isométrique basée sur le PET, puis projette toutes
    les autres modalités (CT, Masques) sur cette nouvelle grille.
    Préserve strictement l'Origine et la Direction Cosine des images.
    """
    orig_size = pet_img.GetSize()
    orig_spacing = pet_img.GetSpacing()

    # Calcul de la nouvelle matrice de voxels pour couvrir le même volume physique
    new_size = [
        max(1, int(round(orig_size[0] * (orig_spacing[0] / iso_spacing[0])))),
        max(1, int(round(orig_size[1] * (orig_spacing[1] / iso_spacing[1])))),
        max(1, int(round(orig_size[2] * (orig_spacing[2] / iso_spacing[2]))))
    ]

    resampler_pet = sitk.ResampleImageFilter()
    resampler_pet.SetSize(new_size)
    resampler_pet.SetOutputSpacing(iso_spacing)
    resampler_pet.SetOutputOrigin(pet_img.GetOrigin())
    resampler_pet.SetOutputDirection(pet_img.GetDirection())
    resampler_pet.SetInterpolator(sitk.sitkBSpline)
    resampler_pet.SetDefaultPixelValue(0.0)

    pet_iso = resampler_pet.Execute(pet_img)

    def align_to_ref(moving_img: sitk.Image, is_mask: bool, pad_value: float) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_iso)
      
        # NearestNeighbor CRITIQUE pour les masques (garde les pixels strictement à 0 ou 1)
        # B-Spline pour les signaux physiques (CT)
        if is_mask:
            interp = sitk.sitkNearestNeighbor
        else:
            interp = sitk.sitkBSpline
        
        resampler.SetInterpolator(interp)
      
        resampler.SetDefaultPixelValue(pad_value)
        return resampler.Execute(moving_img)

    # -1000 HU correspond à l'air (évite les artefacts d'interpolation aux bords du corps)
    ct_iso     = align_to_ref(ct_img, is_mask=False, pad_value=-1000.0)
    breast_iso = align_to_ref(breast_mask_img, is_mask=True, pad_value=0.0)
    tumor_iso  = align_to_ref(tumor_mask_img, is_mask=True, pad_value=0.0)

    # Sécurité typage binaire (Force UInt8, obligatoire pour PyRadiomics)
    breast_iso = sitk.Cast(breast_iso > 0, sitk.sitkUInt8)
    tumor_iso  = sitk.Cast(tumor_iso > 0, sitk.sitkUInt8)
    
    return pet_iso, ct_iso, breast_iso, tumor_iso, pet_iso.GetSpacing()

# =============================================================================
# MODULE 2 : TOPOLOGIE (Latéralité et Anneaux)
# =============================================================================
def _ring_by_distance(
    tumor_mask: np.ndarray,
    breast_mask: np.ndarray,
    spacing_zyx: Tuple[float,float,float],
    inner_mm: float,
    outer_mm: float
) -> np.ndarray:
    """
    =========================================================================
    Construction IBSI-compatible d'une couronne péritumorale
    =========================================================================

    Principe :
    ----------
    Utilise une distance euclidienne réelle (EDT = Euclidean Distance Transform)
    afin de construire des anneaux métriquement exacts en millimètres.

    Pourquoi c'est supérieur à la dilatation morphologique :
    --------------------------------------------------------
    - indépendant de la connectivité voxel
    - invariant à la rotation
    - robuste aux résolutions anisotropes
    - conforme aux pratiques radiomics IBSI

    spacing :
    ---------
    spacing NumPy -> ordre ZYX
    """

    if tumor_mask.sum() == 0 or breast_mask.sum() == 0:
        return np.zeros_like(breast_mask, dtype=bool)

    # ------------------------------------------------------------
    # Distance euclidienne réelle depuis la tumeur
    # ------------------------------------------------------------

    outside = ~tumor_mask

    dist_mm = distance_transform_edt(
        outside,
        sampling=spacing_zyx
    )

    # ------------------------------------------------------------
    # Couronne contrainte au sein
    # ------------------------------------------------------------

    ring = (
        breast_mask &
        (~tumor_mask) &
        (dist_mm > inner_mm) &
        (dist_mm <= outer_mm)
    )

    return ring
  
def split_breasts(breast_mask: np.ndarray, tumor_mask: np.ndarray):
    """
    RESTAURÉ DE LA V1: Sépare le sein ipsilatéral (avec tumeur) du controlatéral.
    Intègre une sécurité géométrique si les deux seins se touchent (sternum).
    """
    labeled, n = label(breast_mask.astype(np.uint8))
    
    # Cas idéal : 2 volumes ou plus bien distincts
    if n >= 2:
        overlaps = [np.logical_and(labeled == (i+1), tumor_mask).sum() for i in range(n)]
        ipsi_label = int(np.argmax(overlaps)) + 1
        ipsi = labeled == ipsi_label
        contra = (labeled != ipsi_label) & (labeled > 0)
        return ipsi, contra
        
    # FALLBACK (Cas difficile) : Si les seins se touchent, coupe géométrique
    coords = np.argwhere(breast_mask)
    if coords.size == 0:
        return breast_mask & False, breast_mask & False
        
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    axis = int(np.argmax(maxc - minc)) # Axe le plus étendu (généralement Gauche-Droite)
    mid = (minc[axis] + maxc[axis]) / 2.0
    
    tcoords = np.argwhere(tumor_mask)
    if tcoords.size == 0:
        # Si pas de tumeur, coupe bêtement à la moitié
        ipsi = breast_mask & (np.indices(breast_mask.shape)[axis] < mid)
        return ipsi, breast_mask & (~ipsi)
        
    # Coupe en fonction du centre de la tumeur
    tcent = tcoords.mean(axis=0)[axis]
    ipsi = breast_mask & ((np.indices(breast_mask.shape)[axis] < mid) if tcent < mid else (np.indices(breast_mask.shape)[axis] >= mid))
    return ipsi, breast_mask & (~ipsi)

# =============================================================================
# MODULE 3 : FEATURES NUMPY (Formes, Métabolisme, Densité)
# =============================================================================
def voxel_volume_mm3(spacing_zyx):
    return float(spacing_zyx[0]*spacing_zyx[1]*spacing_zyx[2])

def _surface_area_voxel(mask: np.ndarray, spacing_zyx: Tuple[float,float,float]) -> float:
    sz, sy, sx = spacing_zyx; area = 0.0
    a01 = (~mask[1:,:,:]) & mask[:-1,:,:]; a10 = (~mask[:-1,:,:]) & mask[1:,:,:]; area += (a01.sum()+a10.sum())*(sy*sz)
    b01 = (~mask[:,1:,:]) & mask[:,:-1,:]; b10 = (~mask[:,:-1,:]) & mask[:,1:,:]; area += (b01.sum()+b10.sum())*(sx*sz)
    c01 = (~mask[:,:,1:]) & mask[:,:,:-1]; c10 = (~mask[:,:,:-1]) & mask[:,:,1:]; area += (c01.sum()+c10.sum())*(sx*sy)
    return float(area)

def shape_features(mask: np.ndarray, spacing_zyx: Tuple[float,float,float], prefix: str) -> Dict[str, float]:
    vv = voxel_volume_mm3(spacing_zyx); voxels = int(mask.sum()); vol_mm3 = voxels * vv; vol_ml = vol_mm3 / 1000.0
    if voxels == 0:
        return {
            f"{prefix}_voxels": 0,
            f"{prefix}_volume_ml": 0.0,
            f"{prefix}_surface_mm2": np.nan,
            f"{prefix}_sphericity": np.nan,
            f"{prefix}_bbox_x_mm": np.nan,
            f"{prefix}_bbox_y_mm": np.nan,
            f"{prefix}_bbox_z_mm": np.nan,
            f"{prefix}_bbox_volume_ml": np.nan,
            f"{prefix}_compactness_bbox": np.nan
        }
                
    coords = np.argwhere(mask)
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    
    dims_vox_zyx = (maxc - minc + 1).astype(np.float32)
    
    spacing_arr_zyx = np.array(spacing_zyx, dtype=np.float32)
    
    dims_mm_zyx = dims_vox_zyx * spacing_arr_zyx
    
    dim_z_mm, dim_y_mm, dim_x_mm = dims_mm_zyx
    
    surface = _surface_area_voxel(mask, spacing_zyx)
    
    sphericity = (
        ((math.pi ** (1/3.0)) * (6.0 * vol_mm3) ** (2/3.0)) / surface
        if surface > 0 else np.nan
    )
    
    bbox_vol_ml = float(
        (dim_x_mm * dim_y_mm * dim_z_mm) / 1000.0
    )
    
    compact_bbox = (
        vol_ml / bbox_vol_ml
        if bbox_vol_ml > 0 else np.nan
    )
    
    return {
        f"{prefix}_voxels": voxels,
        f"{prefix}_volume_ml": float(vol_ml),
        f"{prefix}_surface_mm2": float(surface),
        f"{prefix}_sphericity": float(sphericity),
    
        f"{prefix}_bbox_x_mm": float(dim_x_mm),
        f"{prefix}_bbox_y_mm": float(dim_y_mm),
        f"{prefix}_bbox_z_mm": float(dim_z_mm),
    
        f"{prefix}_bbox_volume_ml": float(bbox_vol_ml),
        f"{prefix}_compactness_bbox": float(compact_bbox),
    }

def first_order(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = arr[mask]
    if vals.size == 0:
        return {f"{prefix}_n": 0, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_min": np.nan, f"{prefix}_p10": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_p90": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_n": int(vals.size),
            f"{prefix}_mean": float(vals.mean()),
            f"{prefix}_std": float(vals.std(ddof=1)) if vals.size>1 else 0.0,
            f"{prefix}_min": float(vals.min()),
            f"{prefix}_p10": float(np.percentile(vals,10)),
            f"{prefix}_median": float(np.median(vals)),
            f"{prefix}_p90": float(np.percentile(vals,90)),
            f"{prefix}_max": float(vals.max())}

def suv_peak_3x3x3(pet: np.ndarray, tumor_mask: np.ndarray) -> float:
    if tumor_mask.sum() == 0: return np.nan
    t = pet.copy(); t[~tumor_mask] = -np.inf
    x, y, z = np.unravel_index(int(np.argmax(t)), t.shape)
    xs, ys, zs = slice(max(0, x-1), min(pet.shape[0], x+2)), slice(max(0, y-1), min(pet.shape[1], y+2)), slice(max(0, z-1), min(pet.shape[2], z+2))
    vals = pet[xs, ys, zs][tumor_mask[xs, ys, zs]]
    return float(vals.mean()) if vals.size else np.nan

def mtv_tlg(pet: np.ndarray, tumor_mask: np.ndarray, spacing_zyx: Tuple[float,float,float], mode: str="41pct"):
    vals = pet[tumor_mask]
    if vals.size == 0: return 0.0, 0.0
    thr = float(vals.max() * 0.41) if mode == "41pct" else 2.5
    sub = tumor_mask & (pet >= thr)
    if sub.sum() == 0: return 0.0, 0.0
    vol_ml = sub.sum() * voxel_volume_mm3(spacing_zyx) / 1000.0
    tlg = float(pet[sub].mean() * vol_ml)
    return float(vol_ml), float(tlg)

# =============================================================================
# MODULE 4 : PYRADIOMICS (Textures Intelligentes)
# =============================================================================
def _make_extractor(bin_width: float, glcm_distances: List[int] = [1], enable_shape: bool = True, enable_log: bool = False, enable_wavelet: bool = False):
    """Instancie l'extracteur PyRadiomics avec la liste COMPLÈTE des features."""
    # IMPORTANT:
    #
    # Shape features are enabled only for true anatomical ROIs
    # such as tumors.
    #
    # They are disabled for artificial peritumoral rings because:
    #
    # - ring geometry is algorithmically generated
    # - strongly dependent on dilation/distance definition
    # - highly correlated with tumor size
    # - often biologically non-informative
    #
    # This improves feature robustness and reduces redundancy.
  
    try:
        from radiomics import featureextractor
    except ImportError as e:
        raise RuntimeError("PyRadiomics non installé. Installer avec 'pip install pyradiomics'.")

    settings = {
        "binWidth": float(bin_width),
    
        # ---------------------------------------------------------
        # Les volumes ont déjà été rééchantillonnés explicitement
        # en isotropique avant PyRadiomics.
        #
        # IMPORTANT :
        # Empêche PyRadiomics de refaire un resampling interne
        # silencieux pouvant modifier :
        #
        # - les textures
        # - les voisinages GLCM
        # - les matrices GLRLM/GLSZM
        # ---------------------------------------------------------
    
        "resampledPixelSpacing": None,
    
        "normalize": False,
        "interpolator": "sitkBSpline", # B-Spline est plus robuste que Linear pour l'analyse de texture
        "label": 1,
        "preCrop": True,
        "correctMask": True,
        "geometryTolerance": 1e-5,
    
        # Distances spatiales GLCM
        "distances": glcm_distances,
    }

    extr = featureextractor.RadiomicsFeatureExtractor(**settings)
    extr.enableImageTypes(Original={})
    
    extr.disableAllFeatures()

    for cls in [
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm"
    ]:
        extr.enableFeatureClassByName(cls)

    # ============================================================
    # Shape features are biologically meaningful for tumors,
    # but generally NOT for artificial peritumoral rings.
    #
    # Therefore they can be disabled for ring extraction.
    # ============================================================
    if enable_shape:
        extr.enableFeatureClassByName("shape")

    img_types = {"Original": {}}

    if enable_log:
        img_types["LoG"] = {"sigma": [1.0, 2.0]}
    
    if enable_wavelet:
        img_types["Wavelet"] = {}
    
    extr.enableImageTypes(**img_types)
      
    return extr

def _pyrad_ring_features(extr, img_sitk: sitk.Image, ring_np: np.ndarray, ref_sitk: sitk.Image, prefix: str) -> Dict[str, float]:
    """
    [RESTAURÉ] Wrapper DRY pour exécuter PyRadiomics proprement, avec gestion fine des NaN.
    Prend l'image originale SITK, le masque Numpy, et une image de référence pour copier la géométrie.
    """
    if ring_np.sum() == 0:
        return {f"{prefix}EMPTY": 1}
        
    # Reconversion du masque rogné en SimpleITK avec géométrie stricte
    msk_sitk = sitk.GetImageFromArray(ring_np.astype(np.uint8))
    msk_sitk.CopyInformation(ref_sitk)
    
    result = extr.execute(img_sitk, msk_sitk, label=1)
    out: Dict[str, float] = {}
    
    for k, v in result.items():
        if str(k).startswith("diagnostics"):
            continue
        try:
            out[f"{prefix}{k}"] = float(v)
        except Exception:
            try:
                out[f"{prefix}{k}"] = float(np.asarray(v).item())
            except Exception:
                pass
    return out

# =============================================================================
# MODULE 5 : ORCHESTRATION PAR PATIENT
# =============================================================================
def case_features(case_id: str,
                  pet_path: str,
                  ct_path: str,
                  breast_path: str,
                  tumor_path: str,
                  ring_mm_1: float = 5.0,
                  ring_mm_2: float = 10.0,
                  enable_pyradiomics: bool = True,
                  pet_binwidth_suv: float = 0.5,
                  ct_binwidth_hu: float = 50.0) -> Dict[str, float]:

    print(f" [EXTRACTION] Traitement du patient {case_id}...")
    
    # 1. Chargement SimpleITK
    pet_img = sitk.ReadImage(pet_path, sitk.sitkFloat32)
    ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
    breast_mask = sitk.ReadImage(breast_path, sitk.sitkUInt8)
    tumor_mask = sitk.ReadImage(tumor_path, sitk.sitkUInt8)
    
    # 2. Ré-échantillonnage et Alignement Strict (SITK = ordre XYZ)
    pet_sitk, ct_sitk, breast_sitk, tumor_sitk, spacing_xyz = isotropic_and_align_to_pet(
        pet_img, ct_img, breast_mask, tumor_mask, iso_spacing=ISO_SPACING
    )

    # [CORRECTION CRITIQUE] L'ordre des axes NumPy est (Z, Y, X). On inverse le spacing.
    spacing_zyx = spacing_xyz[::-1]

    # 3. Extraction Numpy
    pet_np = sitk.GetArrayFromImage(pet_sitk)
    ct_np = sitk.GetArrayFromImage(ct_sitk)
    # CRITIQUE : Convertir en booléen pour que Numpy comprenne qu'il s'agit d'un masque
    # et non d'une matrice de coordonnées 3D (évite l'explosion de la RAM en 5D).
    breast_np = sitk.GetArrayFromImage(breast_sitk).astype(bool)
    tumor_np = sitk.GetArrayFromImage(tumor_sitk).astype(bool)
    
    # Cropping sur la bounding box du sein pour économiser la RAM
    [pet_np, ct_np, breast_np, tumor_np], slicer = _crop_to_mask_bbox(
        [pet_np, ct_np, breast_np, tumor_np], breast_np, spacing_zyx, margin_mm=12.0
    )
    
    # [AJOUT SÉCURITÉ] Vérification masque tumoral vide après cropping/resampling
    tumor_np = tumor_np & breast_np
    if tumor_np.sum() == 0:
        print(f"   [ALERTE] Tumeur vide ou hors du sein pour {case_id}.")
        return {"case_id": case_id, "empty_tumor_mask": 1}

    # Stabilisation CT
    ct_np = np.clip(ct_np, -200.0, 300.0).astype(np.float32)
    
    # Topologie
    ipsi, contra = split_breasts(breast_np, tumor_np)
    ring05 = _ring_by_distance(
        tumor_np,
        breast_np,
        spacing_zyx,
        0.0,
        ring_mm_1
    )
    
    ring10 = _ring_by_distance(
        tumor_np,
        breast_np,
        spacing_zyx,
        ring_mm_1,
        ring_mm_2
    )
    contra_bg = contra & (~tumor_np)
    
    # --- DEBUT DU DICTIONNAIRE DE FEATURES ---
    feats: Dict[str, float] = {"case_id": case_id}
    
    # --- Forme de la tumeur (On utilise bien spacing_zyx ici) ---
    feats.update(shape_features(tumor_np, spacing_zyx, prefix="tumor_shape"))
    
    # --- Caractéristiques PET ---
    fo_pet_tumor = first_order(pet_np, tumor_np, "pet_tumor")
    keep_pet = ["pet_tumor_mean","pet_tumor_std","pet_tumor_min","pet_tumor_median","pet_tumor_max","pet_tumor_p10","pet_tumor_p90"]
    feats.update({k: fo_pet_tumor[k] for k in keep_pet})

    feats["pet_tumor_SUVpeak3x3x3"] = suv_peak_3x3x3(pet_np, tumor_np)
    
    mtv41, tlg41 = mtv_tlg(pet_np, tumor_np, spacing_zyx, "41pct")
    feats["pet_tumor_MTV41_ml"], feats["pet_tumor_TLG41"] = mtv41, tlg41
    mtv25, tlg25 = mtv_tlg(pet_np, tumor_np, spacing_zyx, "2.5")
    feats["pet_tumor_MTV2p5_ml"], feats["pet_tumor_TLG2p5"] = mtv25, tlg25
    
    # [CORRIGÉ] TBR (Tumor to Background Ratio) robuste aux NaN
    cmean = float(pet_np[contra_bg].mean()) if contra_bg.any() else np.nan
    tmean = feats.get("pet_tumor_mean", np.nan)
    if cmean is not None and not np.isnan(cmean) and cmean != 0:
        feats["pet_TBR_tumorMean_over_contraMean"] = (tmean / cmean)
    else:
        feats["pet_TBR_tumorMean_over_contraMean"] = np.nan
    
    # --- Caractéristiques CT ---
    fo_ct_tumor = first_order(ct_np, tumor_np, "ct_tumor_HU")
    keep_ct = ["ct_tumor_HU_mean","ct_tumor_HU_std","ct_tumor_HU_min","ct_tumor_HU_median","ct_tumor_HU_max","ct_tumor_HU_p10","ct_tumor_HU_p90"]
    feats.update({k: fo_ct_tumor[k] for k in keep_ct})
    
    # --- PyRadiomics (Textures Avancées) ---
    if enable_pyradiomics:
        try:
            # Pour PyRadiomics, on doit fournir des images SITK de la MÊME TAILLE que le masque rogné.
            # On recrée donc des objets SITK à partir de nos arrays Numpy rognés (qui sont petits et rapides).
            pet_sitk_cropped = sitk.GetImageFromArray(pet_np)
            ct_sitk_cropped = sitk.GetImageFromArray(ct_np)
            
            # On leur donne l'espacement original (SITK veut l'ordre XYZ)
            pet_sitk_cropped.SetSpacing(spacing_xyz)
            ct_sitk_cropped.SetSpacing(spacing_xyz)

            pet_sitk_cropped.SetDirection(pet_sitk.GetDirection())
            ct_sitk_cropped.SetDirection(ct_sitk.GetDirection())
            
            # Recalcul mathématique parfait de l'origine de l'image rognée
            # CRITIQUE : Conversion explicite en int() natif Python, car SimpleITK 
            # rejette les types numpy.int64 pour cette fonction spatiale.
            new_origin = pet_sitk.TransformIndexToPhysicalPoint(
                (int(slicer[2].start), int(slicer[1].start), int(slicer[0].start))
            )     
          
            pet_sitk_cropped.SetOrigin(new_origin)
            ct_sitk_cropped.SetOrigin(new_origin)

            pet_tumor_extr = _make_extractor(
                pet_binwidth_suv,
                glcm_distances=[1],
                enable_shape=True
            )
            
            ct_tumor_extr = _make_extractor(
                ct_binwidth_hu,
                glcm_distances=[1],
                enable_shape=True
            )
            
            pet_ring_extr = _make_extractor(
                pet_binwidth_suv,
                glcm_distances=[1],
                enable_shape=False
            )
            
            ct_ring_extr = _make_extractor(
                ct_binwidth_hu,
                glcm_distances=[1],
                enable_shape=False
            )

            # ============================================================
            # PYRADIOMICS COMPLET (PET + CT)
            # ============================================================
            # On extrait :
            # - les textures intratumorales
            # - les textures péritumorales 0-5 mm
            # - les textures péritumorales 5-10 mm
            #
            # Ces features capturent :
            # - hétérogénéité tumorale
            # - infiltration locale
            # - organisation spatiale
            # - micro-architecture
            # ============================================================
            
            # ---------- PET ----------
            feats.update(_pyrad_ring_features(
                pet_tumor_extr,
                pet_sitk_cropped,
                tumor_np,
                pet_sitk_cropped,
                "pyrad_pet_tumor__"
            ))
            
            feats.update(_pyrad_ring_features(
                pet_ring_extr,
                pet_sitk_cropped,
                ring05,
                pet_sitk_cropped,
                "pyrad_pet_ring0to5__"
            ))
            
            feats.update(_pyrad_ring_features(
                pet_ring_extr,
                pet_sitk_cropped,
                ring10,
                pet_sitk_cropped,
                "pyrad_pet_ring5to10__"
            ))
            
            # ---------- CT ----------
            feats.update(_pyrad_ring_features(
                ct_tumor_extr,
                ct_sitk_cropped,
                tumor_np,
                ct_sitk_cropped,
                "pyrad_ct_tumor__"
            ))
            
            feats.update(_pyrad_ring_features(
                ct_ring_extr,
                ct_sitk_cropped,
                ring05,
                ct_sitk_cropped,
                "pyrad_ct_ring0to5__"
            ))
            
            feats.update(_pyrad_ring_features(
                ct_ring_extr,
                ct_sitk_cropped,
                ring10,
                ct_sitk_cropped,
                "pyrad_ct_ring5to10__"
            ))
            
        except Exception as e:
            feats["pyradiomics_error"] = 1
            feats["pyradiomics_error_msg"] = str(e) # [RESTAURÉ] Tracabilité des erreurs
            print(f"   [Erreur PyRadiomics] {e}")

    
    # Features background
    breast_bg = breast_np & (~tumor_np)
    ipsi_bg = ipsi & (~tumor_np)
    contra_bg = contra & (~tumor_np)

    # Métadonnées pour traçabilité du modèle
    feats["ring_inner_mm"] = float(ring_mm_1)
    feats["ring_outer_mm"] = float(ring_mm_2)
    feats["isotropic_spacing_mm"] = float(spacing_xyz[0]) # En isométrique, x, y ou z c'est pareil
    feats["pyradiomics_enabled"] = int(enable_pyradiomics)

    # Ajout asymétries pet
    feats.update(first_order(pet_np, breast_bg, "pet_breast_bg"))
    feats.update(first_order(pet_np, ipsi_bg, "pet_ipsi_bg"))
    feats.update(first_order(pet_np, contra_bg, "pet_contra_bg"))
    
    feats["pet_asym_ipsi_over_contra_mean"] = asymmetry_metric(
        pet_np[ipsi_bg],
        pet_np[contra_bg]
    )

    # Ajout aymsétrie ct
    feats.update(first_order(ct_np, breast_bg, "ct_breast_bg_HU"))
    feats.update(first_order(ct_np, ipsi_bg, "ct_ipsi_bg_HU"))
    feats.update(first_order(ct_np, contra_bg, "ct_contra_bg_HU"))
    
    feats["ct_asym_ipsi_over_contra_meanHU"] = asymmetry_metric(
        ct_np[ipsi_bg],
        ct_np[contra_bg]
    )

    # Ajout features ring pet et ct
    feats.update(first_order(pet_np, ring05, "pet_ring_0to5mm"))
    feats.update(first_order(pet_np, ring10, "pet_ring_5to10mm"))
    feats.update(first_order(ct_np, ring05, "ct_ring_0to5mm_HU"))
    feats.update(first_order(ct_np, ring10, "ct_ring_5to10mm_HU"))

    return feats

# =============================================================================
# MODULE 6 : DÉCOUVERTE ET LANCEMENT MOTEUR (MAIN)
# =============================================================================
def discover_subjects_from_nnunet(nnunet_raw_dir: str, breast_masks_dir: str) -> List[Dict[str, str]]:
    labels_dir = os.path.join(nnunet_raw_dir, "labelsTr")
    images_dir = os.path.join(nnunet_raw_dir, "imagesTr")

    if not os.path.exists(labels_dir):
        print(f"[ERREUR FATALE] Dossier nnU-Net introuvable : {labels_dir}")
        return []
        
    tumor_files = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
    cases = []

    for tumor_path in tumor_files:
        subj_id = os.path.basename(tumor_path).replace(".nii.gz", "")
        pet_path = os.path.join(images_dir, f"{subj_id}_0000.nii.gz")
        ct_path = os.path.join(images_dir, f"{subj_id}_0001.nii.gz")
        breast_path = os.path.join(breast_masks_dir, f"{subj_id}_breast_mask.nii.gz")
        
        if not os.path.exists(pet_path) or not os.path.exists(ct_path):
            print(f" [SKIP] {subj_id} : Images PET/CT manquantes.")
            continue
            
        if not os.path.exists(breast_path):
            print(f" [SKIP] {subj_id} : Masque sein introuvable.")
            continue
            
        cases.append({
            "case_id": subj_id,
            "pet": pet_path,
            "ct": ct_path,
            "breast": breast_path,
            "tumor": tumor_path
        })
        
    return cases

def save_dataset(rows: List[Dict[str, float]], out_csv: str) -> pd.DataFrame:
    """RESTAURÉ : Enregistre le dataset final en .csv ET .xlsx."""
    df = pd.DataFrame(rows)
    if "case_id" in df.columns:
        cols = ["case_id"] + [c for c in df.columns if c != "case_id"]
        df = df[cols]
        
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCÈS] CSV sauvegardé -> {out_csv}")
    
    # Export Excel (très utile pour l'exploration clinique)
    out_xlsx = out_csv.replace(".csv", ".xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="radiomics")
        print(f"[SUCCÈS] Excel sauvegardé -> {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Sauvegarde Excel impossible (vérifiez si 'xlsxwriter' est installé) : {e}")
        
    return df

def main():
    parser = argparse.ArgumentParser(description="Extracteur Radiomique pour Modèles de Prédiction (Basé sur nnU-Net).")
    parser.add_argument("--nnunet_dir", type=str, default="./nnunet_data/nnUNet_raw/Dataset002_BreastPETCT")
    parser.add_argument("--breast_dir", type=str, default="./Base_PETCT_BreastMasks")
    parser.add_argument("--output_csv", type=str, default="./radiomics_features_petct.csv")
    parser.add_argument("--no_pyrad", action="store_true", help="Désactive PyRadiomics pour n'avoir que les features Numpy rapides.")

    args = parser.parse_args()
    
    print("=== DÉMARRAGE PIPELINE RADIOMIQUES ULTIME ===")
    cases = discover_subjects_from_nnunet(args.nnunet_dir, args.breast_dir)

    if not cases:
        print("Aucun patient trouvé. Vérifiez les chemins.")
        return
        
    print(f"-> {len(cases)} patients complets découverts.")

    all_features = []
    for c in cases:
        try:
            feats = case_features(
                case_id=c["case_id"],
                pet_path=c["pet"],
                ct_path=c["ct"],
                breast_path=c["breast"],
                tumor_path=c["tumor"],
                enable_pyradiomics=not args.no_pyrad
            )
            all_features.append(feats)
        except Exception as e:
            print(f" [ERREUR CRITIQUE] Échec extraction pour {c['case_id']} : {e}")
            
    if all_features:
        save_dataset(all_features, args.output_csv)
        
    print("=== FIN DU PIPELINE ===")

if __name__ == "__main__":
    main()

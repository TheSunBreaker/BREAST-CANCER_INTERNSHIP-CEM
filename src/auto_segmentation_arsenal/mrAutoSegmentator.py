#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Automate de Segmentation Tumorale PET/CT (Baseline Strict)
===============================================================================
Rôle :
  Orchestre la segmentation automatique des lésions mammaires sans Deep Learning.
  Croise les informations métaboliques (PET) et anatomiques (CT) en se 
  restreignant au masque des seins généré par TotalSegmentator.

Prérequis :
  - Le pipeline Ingesteur V6 doit être terminé.
  - Le convertisseur SUV doit être passé.
  - TotalSegmentator doit avoir généré les masques mammaires.

Structure Attendue :
  Base_PETCT/
    ├── DUKE_001/imgs/ 
    │   ├── DUKE_001_TEP_Baseline_XXX_SUV.nii.gz (ou RAW)
    │   └── DUKE_001_TDM_XXX.nii.gz
  Base_PETCT_BreastMasks/
    ├── DUKE_001_breast_mask.nii.gz

Structure Produite :
  Base_PETCT_AutoMasks/
    ├── DUKE_001_auto_tumor.nii.gz
===============================================================================
"""

import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from utils.spatial_standardizer import resample_to_reference

# ============================================================
# UTILITAIRES SPATIAUX
# ============================================================

def voxel_volume_ml(img: sitk.Image) -> float:
    """
    Calcule le volume d'un voxel en millilitres (mL) à partir de son spacing (mm).
    """
    sx, sy, sz = img.GetSpacing()
    return (sx * sy * sz) / 1000.0

# ============================================================
# FILTRES ANATOMIQUES (CT-BASED)
# ============================================================

def create_soft_tissue_mask(ct_img: sitk.Image) -> sitk.Image:
    """
    Crée un masque des tissus mous en utilisant les Unités Hounsfield (HU).
    Exclut l'air (< -200) et les os purs (> 300).
    """
    ct_np = sitk.GetArrayFromImage(ct_img)
    mask = (ct_np > -200) & (ct_np < 300)
    
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(ct_img)
    return out

def remove_high_density_regions(ct_img: sitk.Image) -> sitk.Image:
    """
    Identifie les structures très denses (Os, fortes calcifications).
    Elles seront soustraites de la segmentation PET pour éviter les faux positifs.
    """
    ct_np = sitk.GetArrayFromImage(ct_img)
    bone_mask = ct_np > 300
    
    out = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
    out.CopyInformation(ct_img)
    return out

# ============================================================
# SEGMENTATION MÉTABOLIQUE (PET-BASED)
# ============================================================

def pet_local_peak_segmentation(pet_img: sitk.Image, roi_mask: sitk.Image, rel_thr=0.45, seed_frac=0.3) -> sitk.Image:
    """
    Algorithme de segmentation par seuillage adaptatif local (Local Peak).
    1. Trouve les zones "chaudes" globales (seed_frac = ex: 30% du max de l'image).
    2. Isole ces zones en composants connectés.
    3. Pour chaque zone isolée, calcule son max local, et segmente à X% (rel_thr) de CE max local.
    """
    pet_np = sitk.GetArrayFromImage(pet_img)
    roi_np = sitk.GetArrayFromImage(roi_mask)

    # Applique le masque du sein (ROI) pour ignorer le cœur et le foie
    pet_roi = np.where(roi_np > 0, pet_np, 0.0)

    positive = pet_roi[pet_roi > 0]
    if positive.size == 0:
        return sitk.Image(pet_img.GetSize(), sitk.sitkUInt8)

    global_max = float(positive.max())

    # Étape 1: Seeding (seuil très permissif pour attraper toutes les lésions)
    seed_thr = seed_frac * global_max
    seed = (pet_roi >= seed_thr).astype(np.uint8)

    seed_img = sitk.GetImageFromArray(seed)
    seed_img.CopyInformation(pet_img)

    # Étape 2: Composants connectés
    cc = sitk.ConnectedComponent(seed_img)
    cc_np = sitk.GetArrayFromImage(cc)

    labels = np.unique(cc_np)
    labels = labels[labels != 0]

    final = np.zeros_like(cc_np, dtype=np.uint8)

    # Étape 3: Affinement par pic local
    for lab in labels:
        region = (cc_np == lab)
        local_peak = float(pet_roi[region].max())
        local_thr = rel_thr * local_peak
        
        refined = (pet_roi >= local_thr) & region
        final |= refined.astype(np.uint8)

    out = sitk.GetImageFromArray(final)
    out.CopyInformation(pet_img)
    return out

# ============================================================
# POST-TRAITEMENT ET CONTRAINTES
# ============================================================

def remove_small_components(mask_img: sitk.Image, min_volume_ml: float) -> sitk.Image:
    """Nettoie le bruit en supprimant les détections de volume inférieur au seuil."""
    cc = sitk.ConnectedComponent(mask_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    vx = voxel_volume_ml(mask_img)
    cc_np = sitk.GetArrayFromImage(cc)
    keep = np.zeros_like(cc_np, dtype=np.uint8)

    for lab in stats.GetLabels():
        vol = stats.GetNumberOfPixels(lab) * vx
        if vol >= min_volume_ml:
            keep[cc_np == lab] = 1

    out = sitk.GetImageFromArray(keep)
    out.CopyInformation(mask_img)
    return out

def apply_ct_constraints(tumor_mask: sitk.Image, soft_mask: sitk.Image, bone_mask: sitk.Image) -> sitk.Image:
    """Croise la matrice PET avec la vérité anatomique CT."""
    tumor_np = sitk.GetArrayFromImage(tumor_mask)
    soft_np = sitk.GetArrayFromImage(soft_mask)
    bone_np = sitk.GetArrayFromImage(bone_mask)

    tumor_np[bone_np > 0] = 0           # Retire les os
    tumor_np = tumor_np & (soft_np > 0) # Garde uniquement les tissus mous

    out = sitk.GetImageFromArray(tumor_np.astype(np.uint8))
    out.CopyInformation(tumor_mask)
    return out

# ============================================================
# PIPELINE MÉTIER
# ============================================================

# ============================================================
# PIPELINE MÉTIER
# ============================================================

def advanced_pet_ct_pipeline(pet_img: sitk.Image, ct_img: sitk.Image, breast_mask: sitk.Image,
                             rel_thr=0.45, seed_frac=0.3, min_volume_ml=0.5, use_ct=False) -> sitk.Image:
    """Combine toutes les étapes de manière orchestrée."""
    
    # 1. Recalage du masque du sein sur le PET
    breast_pet = resample_to_reference(moving_img=breast_mask, ref_img=pet_img, is_mask=True, pad_value=0.0)

    # 2. Segmentation PET primaire dans le sein
    tumor_mask = pet_local_peak_segmentation(pet_img, breast_pet, rel_thr=rel_thr, seed_frac=seed_frac)

    # 3. Application conditionnelle des contraintes CT
    if use_ct and ct_img is not None:
        # Recalage du CT sur le PET
        ct_pet = resample_to_reference(moving_img=ct_img, ref_img=pet_img, is_mask=False, pad_value=-1000.0)

        # Création des masques dérivés du CT
        soft_mask = create_soft_tissue_mask(ct_pet)
        bone_mask = remove_high_density_regions(ct_pet)

        # Restriction aux tissus mous à l'intérieur du sein
        soft_np = sitk.GetArrayFromImage(soft_mask)
        breast_np = sitk.GetArrayFromImage(breast_pet)
        soft_np = soft_np & (breast_np > 0)
        
        soft_mask = sitk.GetImageFromArray(soft_np.astype(np.uint8))
        soft_mask.CopyInformation(ct_pet)

        # Application des contraintes
        tumor_mask = apply_ct_constraints(tumor_mask, soft_mask, bone_mask)

    # 4. Nettoyage final
    tumor_mask = remove_small_components(tumor_mask, min_volume_ml)

    return tumor_mask

# ============================================================
# ORCHESTRATEUR PRINCIPAL (MAIN)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Segmentation tumorale automatique PET/CT.")
    
    # Chemins avec valeurs par défaut alignées sur l'architecture V6
    parser.add_argument("--petct_root", type=Path, default=Path("./Base_PETCT"), help="Racine des données PET/CT")
    parser.add_argument("--breast_masks", type=Path, default=Path("./Base_PETCT_BreastMasks_Expanded"), help="Dossier des masques mammaires")
    parser.add_argument("--output_root", type=Path, default=Path("./Base_PETCT_AutoMasks"), help="Dossier de sortie des tumeurs")
    
    # Paramètres algorithmiques
    parser.add_argument("--rel-thr", type=float, default=0.45, help="Seuil local du pic PET (Défaut: 45%)")
    parser.add_argument("--seed-frac", type=float, default=0.30, help="Fraction du max global pour le seeding (Défaut: 30%)")
    parser.add_argument("--min-vol", type=float, default=0.5, help="Volume minimal en mL (Défaut: 0.5 mL)")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les résultats existants")
    parser.add_argument("--use-ct", action="store_true", help="Active les contraintes anatomiques CT (Désactivé par défaut)")
    
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    
    patients = [p for p in args.petct_root.iterdir() if p.is_dir()]
    print(f"\n=== LANCEMENT PIPELINE AUTO-SEGMENTATION ({len(patients)} patients potentiels) ===")

    for patient_dir in patients:
        patient_id = patient_dir.name
        
        # 1. Règle Longitudinale : On cible uniquement 'imgs' (Baseline)
        imgs_dir = patient_dir / "imgs"
        if not imgs_dir.exists():
            continue
            
        out_mask_path = args.output_root / f"{patient_id}_auto_tumor.nii.gz"
        if out_mask_path.exists() and not args.overwrite:
            print(f"[SKIP] {patient_id} : Auto-mask déjà généré.")
            continue

        # 2. Recherche du Masque Sein (TotalSegmentator)
        breast_mask_path = args.breast_masks / f"{patient_id}_breast*.nii.gz"
        if not breast_mask_path.exists():
            print(f"[WARN] {patient_id} : Masque sein introuvable, patient ignoré.")
            continue

        # 3. Recherche du CT (Conditionnelle)
        ct_path = None
        ct_img = None
        if args.use_ct:
            ct_files = list(imgs_dir.glob("*_TDM_*.nii.gz"))
            if not ct_files:
                print(f"[WARN] {patient_id} : Scanner CT introuvable dans Baseline. Ignoré.")
                continue
            ct_path = ct_files[0]
            ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)

        # 4. Recherche du PET (On privilégie le SUV, sinon on prend le RAW en fallback)
        pet_files = list(imgs_dir.glob("*_TEP_*_SUV.nii.gz"))
        if not pet_files:
            pet_files = list(imgs_dir.glob("*_TEP_*_RAW.nii.gz"))
            if not pet_files:
                print(f"[WARN] {patient_id} : PET introuvable dans Baseline.")
                continue
            print(f"  [INFO] Impossible de trouver de PET SUV pour le patient : {patient_id}. On utilisera alors du PET RAW à la place.")
        else:
            print(f"  [INFO] Utilisation du PET SUV pour {patient_id}")
            
        pet_path = pet_files[0]

        # --------------------------------------------------------
        # EXECUTION
        # --------------------------------------------------------
        print(f"[RUN ] {patient_id} en cours de segmentation...")
        try:
            pet_img = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)
            breast_mask = sitk.ReadImage(str(breast_mask_path), sitk.sitkUInt8)
            
            final_tumor = advanced_pet_ct_pipeline(
                pet_img, ct_img, breast_mask,
                rel_thr=args.rel_thr,
                seed_frac=args.seed_frac,
                min_volume_ml=args.min_vol,
                use_ct=args.use_ct
            )
            
            sitk.WriteImage(final_tumor, str(out_mask_path))
            print(f"  -> [SUCCÈS] {out_mask_path.name}")
            
        except Exception as e:
            print(f"  -> [ÉCHEC ] {patient_id} : {e}")

    print("\n=== PIPELINE TERMINÉ ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Extracteur de Radiomiques IRM DCE Multi-Phases & Delta-Cinétiques (V2 - Clinique)
===============================================================================

DESCRIPTION :
    Ce script extrait les descripteurs radiomiques (statiques et dynamiques) à partir 
    d'images IRM DCE triées et normalisées au format nnU-Net V2.
    Il extrait les caractéristiques sur la tumeur ainsi que sur un anneau péritumoral.
    Il applique ensuite un pivotement (Flattening) pour condenser toutes les phases 
    d'un patient sur une SEULE ligne, puis calcule automatiquement les Delta-Radiomiques.

LOGIQUE DES DELTAS :
    Pour N phases, le script génère (N-1) blocs de deltas absolus et relatifs.
    Chaque phase post-contraste (0001, 0002, ...) est comparée à la phase native 
    pré-contraste (0000).
    - Delta Absolu  = Phase_X - Phase_0000
    - Delta Relatif = (Phase_X - Phase_0000) / (Phase_0000 + Epsilon)

SÉCURITÉS ANATOMIQUES ET TECHNIQUES :
    - Filtre Classe Tumeur : Isole strictement la valeur 1 (Tumeur). Exclut les 
      ganglions (Classe 2) pour ne pas contaminer les descripteurs texturaux.
    - Isotropisation : Force un rééchantillonnage à [1, 1, 1] mm pour garantir 
      la comparabilité des matrices de texture (GLCM, GLRLM, etc.).
    - Préservation Normalisation : `normalize=False` pour respecter le Z-Score global.
    - Multiprocessing : Parallélisation native par coeur CPU.
===============================================================================
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from scipy.ndimage import distance_transform_edt

# PyRadiomics est configuré en mode discret pour ne pas saturer la console de logs d'avertissement.
from radiomics import featureextractor, logger as pyr_logger
pyr_logger.setLevel("WARNING") 


# =====================================================================
# 0. Utilitaires
# =====================================================================

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

# =====================================================================
# 1. OUTILS DE RECHERCHE ET CONFIGURATION DES TÂCHES
# =====================================================================

def find_multiphase_tasks_nnunet(imagesTr_dir: Path, labelsTr_dir: Path) -> List[Dict]:
    """
    Scanne les dossiers nnU-Net pour apparier le masque binaire unique du patient
    avec l'ensemble de ses phases temporelles IRM DCE (0000, 0001, etc.).
    """
    tasks = []
    
    # Le point d'ancrage est le dossier des masques (un seul fichier .nii.gz par patient)
    for mask_path in labelsTr_dir.glob("*.nii.gz"):
        subject_id = mask_path.name.replace(".nii.gz", "")
        
        # Collecte de toutes les phases correspondantes (ex: PatientX_0000.nii.gz, PatientX_0001.nii.gz)
        img_files = sorted(list(imagesTr_dir.glob(f"{subject_id}_*.nii.gz")))
        
        if not img_files:
            print(f"[PRE-FLIGHT WARN] Patient {subject_id} ignoré : Masque trouvé mais aucune image correspondante.")
            continue
            
        for img_path in img_files:
            # Extraction propre du numéro de canal nnU-Net à 4 chiffres (ex: 0000, 0001)
            phase_id = img_path.name.replace(".nii.gz", "").split("_")[-1]
            
            tasks.append({
                "subject_id": subject_id,
                "phase_id": phase_id,
                "img_path": img_path,
                "mask_path": mask_path
            })
            
    return tasks

# =====================================================================
# 2. ALGORITHMES DE CONTOURING PÉRITUMORAL (SIMPLEITK)
# =====================================================================

def _ring_by_distance(
    tumor_mask: np.ndarray,
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

    if tumor_mask.sum() == 0:
        return np.zeros_like(tumor_mask, dtype=bool)

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
        (~tumor_mask) &
        (dist_mm > inner_mm) &
        (dist_mm <= outer_mm)
    )

    return ring

# =====================================================================
# 3. MOTEUR DE CONFIGURATION PYRADIOMICS
# =====================================================================

def make_extractor(binWidth: float = 25.0) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Instancie l'extracteur mathématique avec des paramètres reproductibles et stables.
    CRITIQUE : `normalize=False` car la normalisation est déléguée au script MAMA-MIA/nnU-Net.
    `resampledPixelSpacing=[1,1,1]` force l'isotropisation spatiale des voxels avant calcul.
    """
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=binWidth,
        normalize=False,  # OBLIGATOIREMENT FALSE (Respect de l'ingénierie amont)
        resampledPixelSpacing=[1, 1, 1],  # Isotropisation physique
        interpolator='sitkBSpline',       # Spline cubique pour l'intensité d'image
        padDistance=5,
        label=1
    )
    # Désactivation globale pour éviter l'extraction de métadonnées inutiles
    ext.disableAllFeatures()
    
    # Activation sélective des familles de caractéristiques validées en oncologie quantitative
    ext.enableFeaturesByName(
        firstorder=[],  # Statistiques d'intensité du premier ordre (histogramme)
        glcm=[],        # Gray Level Co-occurrence Matrix (textures locales)
        glrlm=[],       # Gray Level Run Length Matrix (alignements)
        glszm=[],       # Gray Level Size Zone Matrix (zones homogènes)
        ngtdm=[]        # Neighborhood Gray Tone Difference Matrix (contraste/coarseness)
    )
    return ext

def execute_extract(extractor, img: sitk.Image, mask: sitk.Image, prefix: str) -> Dict[str, float]:
    """Exécute le calcul sur la ROI et nettoie les en-têtes retournés par PyRadiomics."""
    mask.CopyInformation(img)
    feats = extractor.execute(img, mask)

    out = {}
    for k, v in feats.items():
        if k.startswith("diagnostics"):
            continue
        # Standardisation cosmétique : original_glcm_Contrast -> tumor_glcm_Contrast
        clean_name = f"{prefix}_{re.sub(r'^original_', '', k)}"
        try:
            out[clean_name] = float(v)
        except (ValueError, TypeError):
            continue
            
    return out

# =====================================================================
# 4. LE MULTIPROCESSING WORKER
# =====================================================================

def _process_one_phase_worker(args) -> Tuple[str, str, Dict[str, float], Optional[str]]:
    """
    Unité de calcul autonome (Worker). Traite un couple (Image_Phase, Masque) pour 
    un patient donné. Protégé contre les crashs individuels.
    """
    (task, peri_distances, save_peri_dir, extractor_params) = args
    peri_inner_mm, peri_outer_mm = peri_distances
    subject_id = task["subject_id"]
    phase_id = task["phase_id"]
    img_path = task["img_path"]
    mask_path = task["mask_path"]
    
    try:
        # 1. Chargement et seuillage strict de la tumeur (Classe 1 uniquement)
        img = sitk.ReadImage(str(img_path))
        mask_raw = sitk.ReadImage(str(mask_path))

        # --- SÉCURITÉ DE RECALAGE GÉOMÉTRIQUE INDISPENSABLE ---
        if (img.GetSize() != mask_raw.GetSize() or img.GetSpacing() != mask_raw.GetSpacing()
            or img.GetOrigin() != mask_raw.GetOrigin() or img.GetDirection() != mask_raw.GetDirection()):
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Plus proche voisin strict (Format Masque)
            resampler.SetTransform(sitk.Transform())
            mask_resampled = resampler.Execute(mask_raw)
        else:
            mask_resampled = mask_raw

        # Tout ce qui est au compris entre 1 et 1 est mis comme 1, le reste 0 (uniquement tumeurs donc)
        mask_tumor_sitk = sitk.BinaryThreshold(mask_resampled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)

        # 2. Passage sous NumPy et extraction des caractéristiques géométriques
        spacing_xyz = img.GetSpacing()
        spacing_zyx = spacing_xyz[::-1]
        
        img_np = sitk.GetArrayFromImage(img)
        tumor_np = sitk.GetArrayFromImage(mask_tumor_sitk).astype(bool)
        
        # 3. APPLICATION DU CROPPING (Gain RAM immédiat)
        [img_np, tumor_np], slicer = _crop_to_mask_bbox([img_np, tumor_np], tumor_np, spacing_zyx, margin_mm=12.0)
        
        # 4. Génération de la double couronne par Distance Euclidienne
        ring05 = _ring_by_distance(tumor_np, spacing_zyx, 0.0, peri_inner_mm)
        ring10 = _ring_by_distance(tumor_np, spacing_zyx, peri_inner_mm, peri_outer_mm)
        
        # 5. Re-création des objets SimpleITK rognés pour PyRadiomics
        img_cropped = sitk.GetImageFromArray(img_np)
        img_cropped.SetSpacing(spacing_xyz)
        img_cropped.SetDirection(img.GetDirection())
        
        # RECUPERATION CRITIQUE DE LA NOUVELLE ORIGINE PHYSIQUE
        new_origin = img.TransformIndexToPhysicalPoint((int(slicer[2].start), int(slicer[1].start), int(slicer[0].start)))
        img_cropped.SetOrigin(new_origin)
        
        # Préparation des masques pour PyRadiomics
        def prepare_sitk_mask(np_arr):
            m = sitk.GetImageFromArray(np_arr.astype(np.uint8))
            m.CopyInformation(img_cropped)
            return m
        
        tumor_sitk_crop = prepare_sitk_mask(tumor_np)
        ring05_sitk_crop = prepare_sitk_mask(ring05)
        ring10_sitk_crop = prepare_sitk_mask(ring10)
        
        # 6. Extraction PyRadiomics triple zone
        extractor = make_extractor(binWidth=extractor_params["binWidth"])
        feats_tumor = execute_extract(extractor, img_cropped, tumor_sitk_crop, "tumor")
        feats_ring05 = execute_extract(extractor, img_cropped, ring05_sitk_crop, "peri_0to5mm")
        feats_ring10 = execute_extract(extractor, img_cropped, ring10_sitk_crop, "peri_5to10mm")
        
        # Assemblage final de la ligne de résultats
        merged_features = {"subject_id": subject_id, "phase_id": phase_id}
        merged_features.update(feats_tumor)
        merged_features.update(feats_ring05)
        merged_features.update(feats_ring10)

        return subject_id, phase_id, merged_features, None

    except Exception as e:
        # En cas d'anomalie sur un fichier, on renvoie l'erreur au manager sans bloquer le pool complet
        return subject_id, phase_id, {}, f"{type(e).__name__}: {e}"

# =====================================================================
# 5. ORCHESTRATEUR CENTRAL & CALCULATEUR DE DELTAS DYNAMIQUES
# =====================================================================

def extract_mri_features_and_flatten(
    imagesTr_dir: Path,
    labelsTr_dir: Path,
    out_dir: Path,
    peri_inner_mm: float = 5.0,   # Remplacé peri_mm par inner
    peri_outer_mm: float = 10.0,  # Ajouté outer
    save_peri_masks: bool = False,
    bin_width: float = 25.0,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fonction maîtresse : Gère le pool parallélisé, exécute le pivotement (Flattening) 
    des phases et calcule la matrice complète des Delta-Radiomiques Cinétiques.
    """
    extractor_params = {"binWidth": bin_width}
    
    # Cartographie initiale du dataset
    raw_tasks = find_multiphase_tasks_nnunet(imagesTr_dir, labelsTr_dir)
    
    formatted_tasks = []
    for task in raw_tasks:
        formatted_tasks.append((
            task, 
            (peri_inner_mm, peri_outer_mm), # On passe le tuple des 2 distances au worker
            (out_dir / "peritumoral_masks") if save_peri_masks else None, 
            extractor_params
        ))

    if not formatted_tasks:
        raise RuntimeError("[FATAL] Aucune paire Image/Masque valide détectée. Vérifiez vos arborescences nnU-Net.")

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    results, errors = [], []

    # --- PIPELINE DE CALCUL PARALLÈLE ---
    print(f"[INFO] Initialisation du pool multiprocessing sur {n_jobs} coeurs...")
    with Pool(processes=n_jobs) as pool:
        for subject_id, phase_id, feats, err in tqdm(pool.imap_unordered(_process_one_phase_worker, formatted_tasks),
                                                     total=len(formatted_tasks), desc="Extraction quantitative"):
            if err is None:
                results.append(feats)
            else:
                errors.append({"subject_id": subject_id, "phase_id": phase_id, "error": err})

    # --- ARCHITECTURE DU SCRIPT ET MATRIX FLATTENING ---
    df_results = pd.DataFrame()
    if results:
        # Tri initial pour garantir un ordre d'intégration propre
        df_long = pd.DataFrame(results).sort_values(by=["subject_id", "phase_id"])
        
        print("\n[INFO] Phase de Pivotement (Flattening) : Une seule ligne par patient...")
        # Pivot multidimensionnel Pandas
        df_wide = df_long.pivot(index="subject_id", columns="phase_id")
        
        # Écrasement du MultiIndex des colonnes en en-têtes plats : phase0000_tumor_glcm_Contrast
        flat_columns = []
        for feature_name, phase in df_wide.columns:
            flat_columns.append(f"phase{phase}_{feature_name}")
        df_wide.columns = flat_columns
        
        # --- CALCUL AUTOMATIQUE ET DYNAMIQUE DES DELTA-RADIOMIQUES ---
        feature_names_raw = set()
        detected_phases = set()
        
        # Détection automatique de la morphologie du dataset (nombre de phases extraites)
        for col in df_wide.columns:
            match = re.match(r"phase(\d+)_(.+)", col)
            if match:
                detected_phases.add(match.group(1))
                feature_names_raw.add(match.group(2))
                
        sorted_phases = sorted(list(detected_phases))
        
        # La condition sine qua non pour calculer des deltas est d'avoir au moins la phase de référence 0000
        if "0000" in sorted_phases and len(sorted_phases) > 1:
            print(f"[DYNAMIC DELTA] {len(sorted_phases)} phases detectees. Calcul des deltas cinetiques vs phase0000...")
            
            for phase in sorted_phases:
                if phase == "0000":
                    continue  # On ne calcule pas le delta de la baseline contre elle-même
                    
                for feat in sorted_phases_features := sorted(list(feature_names_raw)):
                    col_baseline = f"phase0000_{feat}"
                    col_current  = f"phase{phase}_{feat}"
                    
                    if col_baseline in df_wide.columns and col_current in df_wide.columns:
                        # 1. Delta Absolu : Capturation de la captation brute de signal (Perfusion pure)
                        df_wide[f"delta_abs_p{phase}_vs_p0000_{feat}"] = df_wide[col_current] - df_wide[col_baseline]
                        
                        # 2. Delta Relatif : Ratio cinétique / Évolution normalisée (Modélisation Wash-in/Wash-out)
                        # Utilisation d'un epsilon à 1e-8 pour immuniser le code contre les divisions par zéro
                        df_wide[f"delta_rel_p{phase}_vs_p0000_{feat}"] = (df_wide[col_current] - df_wide[col_baseline]) / (df_wide[col_baseline] + 1e-8)
        else:
            print("[WARN] Impossible de generer les Delta-Radiomiques : Phase 0000 absente ou volume uniquement statique.")

        # Réintégration de la clé subject_id comme colonne standard
        df_results = df_wide.reset_index()
        
    df_errors = pd.DataFrame(errors).sort_values(["subject_id", "phase_id"]) if errors else pd.DataFrame()

    # --- PERSISTENCE ET SAUVEGARDE COMMANDE EN LIGNE ---
    csv_path  = out_dir / "radiomics_results_mri_FLATTENED.csv"
    xlsx_path = out_dir / "radiomics_results_mri_FLATTENED.xlsx"
    meta_path = out_dir / "run_metadata.json"

    df_results.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="radiomics_features", index=False)
        if not df_errors.empty:
            df_errors.to_excel(writer, sheet_name="extraction_errors", index=False)

    # Exportation du manifeste JSON pour tracer les paramètres d'exécution
    unique_patients_count = len(set([t["subject_id"] for t in raw_tasks])) if raw_tasks else 0
    metadata_manifest = {
        "timestamp_execution": pd.Timestamp.now().isoformat(),
        "n_patients_analysed_input": unique_patients_count,
        "n_rows_computed_long": len(results),
        "n_errors_encountered": len(errors),
        "peritumoral_inner_radius_mm": peri_inner_mm,
        "peritumoral_outer_radius_mm": peri_outer_mm,
        "pyradiomics_binWidth_used": bin_width
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_manifest, f, indent=4)

    print(f"\n[SUCCÈS] Pipeline d'extraction complet terminé avec succès.")
    print(f"Fichiers de données exportés dans : {out_dir.resolve()}")
    return df_results, df_errors

# =====================================================================
# 6. PARSER EN LIGNE DE COMMANDE (CLI PRODUCTION INTERFACE)
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI d'extraction de radiomiques IRM DCE multi-phases et delta-cinétiques pour nnU-Net v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Chemin vers le dossier imagesTr de nnU-Net contenant les IRM.")
    parser.add_argument("--labels_dir", type=str, required=True, help="Chemin vers le dossier labelsTr de nnU-Net contenant les Masques.")
    parser.add_argument("--output_dir", type=str, default="./results_radiomics_mri", help="Dossier de destination pour les fichiers CSV/XLSX.")
    parser.add_argument("--peri_inner_mm", type=float, default=5.0, help="Rayon interne du 1er anneau (mm).")
    parser.add_argument("--peri_outer_mm", type=float, default=10.0, help="Rayon externe du 2e anneau (mm).")
    parser.add_argument("--save_peri", action="store_true", help="Si activé, écrit les masques péritumoraux générés au format NIfTI.")
    parser.add_argument("--bin_width", type=float, default=25.0, help="Largeur de bin pour la discrétisation des niveaux de gris (PyRadiomics).")
    parser.add_argument("--cores", type=int, default=None, help="Nombre de cœurs CPU à allouer. Laisse None pour auto-détection.")
    
    args = parser.parse_args()

    # Déclenchement de l'orchestrateur
    extract_mri_features_and_flatten(
        imagesTr_dir=Path(args.images_dir),
        labelsTr_dir=Path(args.labels_dir),
        out_dir=Path(args.output_dir),
        peri_inner_mm=args.peri_inner_mm,  # Changé ici
        peri_outer_mm=args.peri_outer_mm,  # Changé ici
        save_peri_masks=args.save_peri,
        bin_width=args.bin_width,
        n_jobs=args.cores
    )

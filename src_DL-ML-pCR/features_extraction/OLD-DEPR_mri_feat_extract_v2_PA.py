#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracteur de radiomiques (Édition IRM nnU-Net + Flattening)
------------------------------------------------------------
Ce script a un objectif clair : lire des images IRM DCE multi-phases 
déjà normalisées (depuis l'arborescence nnU-Net), extraire les radiomiques 
de la tumeur et d'un anneau péritumoral, puis "aplatir" les résultats 
pour qu'un patient n'occupe qu'une seule ligne dans le CSV final.
"""

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# PyRadiomics est super bavard par défaut et spamme la console avec des infos inutiles.
# On force le niveau de log à WARNING pour n'afficher que les vrais problèmes.
from radiomics import featureextractor, logger as pyr_logger
pyr_logger.setLevel("WARNING") 

# =====================================================================
# 1. OUTILS DE DÉCOUVERTE (Scanner les dossiers)
# =====================================================================

def find_multiphase_tasks_nnunet(imagesTr_dir: Path, labelsTr_dir: Path) -> List[Dict]:
    """
    Parcourt les dossiers nnU-Net pour associer chaque phase d'une IRM à son masque.
    Retourne une liste de tâches, où 1 tâche = 1 patient + 1 phase spécifique.
    """
    tasks = []
    
    # On itère d'abord sur les masques. Pourquoi ? Parce qu'on est sûr qu'il n'y a 
    # qu'un seul masque par patient (ex: Patient01.nii.gz). C'est notre point d'ancrage.
    for mask_path in labelsTr_dir.glob("*.nii.gz"):
        
        # On extrait l'ID brut du patient en coupant l'extension.
        subject_id = mask_path.name.replace(".nii.gz", "")
        
        # On va chercher toutes les images associées à ce patient dans imagesTr.
        # En nnU-Net, elles s'appellent Patient01_0000, Patient01_0001, etc.
        img_files = sorted(list(imagesTr_dir.glob(f"{subject_id}_*.nii.gz")))
        
        # Sécurité : si on a un masque mais pas d'images, on prévient et on ignore.
        if not img_files:
            print(f"[WARN] Ignoré {subject_id}: masque trouvé mais aucune image dans {imagesTr_dir}")
            continue
            
        # On crée une tâche séparée pour CHAQUE phase trouvée pour ce patient.
        for img_path in img_files:
            # On extrait le numéro de la phase à la fin du nom de fichier.
            # split("_")[-1] prend le dernier bout (ex: "0000.nii.gz"), puis on vire l'extension.
            phase_id = img_path.name.replace(".nii.gz", "").split("_")[-1]
            
            # On empile cette tâche dans la liste. C'est ce dico qui sera envoyé au multiprocessing.
            tasks.append({
                "subject_id": subject_id,
                "phase_id": phase_id,
                "img_path": img_path,
                "mask_path": mask_path
            })
            
    return tasks

# =====================================================================
# 2. MANIPULATION DES MASQUES (SimpleITK)
# =====================================================================

def binary_peri_ring(mask_img: sitk.Image, dilation_mm: float, label: int = 1) -> sitk.Image:
    """
    Génère un masque "donut" autour de la tumeur pour étudier l'environnement proche (péritumoral).
    Logique : On gonfle la tumeur de X mm, puis on soustrait la tumeur d'origine.
    """
    # SimpleITK résonne en pixels (voxels), mais nous on veut dilater en millimètres.
    # On récupère donc l'espacement physique (la taille d'un pixel) pour faire la conversion.
    spacing = mask_img.GetSpacing()
    radius_vox = [int(math.ceil(dilation_mm / s)) for s in spacing]

    # Sécurité : On s'assure que le masque d'entrée est strictement binaire (que des 1 et des 0).
    bin_mask = sitk.BinaryThreshold(mask_img, lowerThreshold=label, upperThreshold=label,
                                    insideValue=1, outsideValue=0)

    # Configuration de l'outil de dilatation de SimpleITK.
    # On utilise une forme de "balle" (sitkBall) pour gonfler uniformément dans toutes les directions 3D.
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radius_vox)
    dilate.SetForegroundValue(1)
    
    # On exécute le gonflement.
    dilated = dilate.Execute(bin_mask)
    
    # Mathématiques basiques : Anneau = (Tumeur Gonflée) - (Tumeur Initiale)
    peri = sitk.Subtract(dilated, bin_mask)
    
    # Clamp est une ceinture de sécurité : on force les valeurs entre 0 et 1 au cas où
    # la soustraction aurait généré des valeurs bizarres (comme des -1).
    peri = sitk.Clamp(peri, lowerBound=0, upperBound=1)  
    return peri

# =====================================================================
# 3. CONFIGURATION DE PYRADIOMICS
# =====================================================================

def make_extractor(binWidth: float = 25.0, normalize: bool = False, label: int = 1) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Prépare le moteur de calcul PyRadiomics.
    ATTENTION : normalize DOIT être False ici, car on utilise des images IRM déjà 
    normalisées globalement (Z-score MAMA-MIA) en amont !
    """
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=binWidth,
        normalize=normalize,
        interpolator='sitkBSpline', # BSpline donne un meilleur rendu si redimensionnement il y a
        label=label
    )
    # Par défaut, on désactive tout pour garder le contrôle sur le temps de calcul.
    ext.disableAllFeatures()
    # On active les familles utiles : statistiques de base (firstorder) et textures (glcm, glrlm, etc.).
    ext.enableFeaturesByName(firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[])
    return ext

def execute_extract(extractor, img: sitk.Image, mask: sitk.Image, prefix: str) -> Dict[str, float]:
    """
    Lance l'extraction sur une image et son masque, et nettoie les noms des colonnes générées.
    """
    # PyRadiomics est extrêmement pointilleux : l'image et le masque doivent être 
    # géométriquement parfaits. CopyInformation force le masque à adopter la grille de l'image.
    mask.CopyInformation(img)
    
    # Le calcul lourd se passe ici.
    feats = extractor.execute(img, mask)

    out = {}
    for k, v in feats.items():
        # PyRadiomics génère plein de variables "diagnostics" (versions du logiciel, etc.). On les jette.
        if k.startswith("diagnostics"):
            continue
            
        # On renomme les variables. "original_glcm_Contrast" devient par exemple "tumor_glcm_Contrast".
        name = f"{prefix}_{re.sub(r'^original_', '', k)}"
        
        # On force la conversion en nombre flottant classique (float) pour que Pandas 
        # n'ait aucun problème à écrire ça dans un CSV plus tard.
        try:
            out[name] = float(v)
        except Exception:
            continue
            
    return out

# =====================================================================
# 4. LE WORKER (La fonction qui tourne en parallèle)
# =====================================================================

def _process_one(args) -> Tuple[str, str, Dict[str, float], Optional[str]]:
    """
    Notre worker. Il prend UNE tâche (ex: Patient01, Phase 2) et fait le boulot.
    """
    # Déballage des arguments envoyés par le multiprocessing
    (task, peri_mm, save_peri_dir, extractor_params) = args
    subject_id = task["subject_id"]
    phase_id = task["phase_id"]
    img_path = task["img_path"]
    mask_path = task["mask_path"]
    
    try:
        # Chargement en mémoire vive de l'image Nifti et du masque
        img = sitk.ReadImage(str(img_path))
        mask_raw = sitk.ReadImage(str(mask_path))

        # --- Étape de Recalage de sécurité ---
        # Si pour une raison obscure l'image et le masque n'ont pas exactement 
        # le même espacement ou origine, on force l'alignement.
        if (img.GetSize() != mask_raw.GetSize()
            or img.GetSpacing() != mask_raw.GetSpacing()
            or img.GetOrigin()  != mask_raw.GetOrigin()
            or img.GetDirection()!= mask_raw.GetDirection()):
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            # IMPORTANT : on utilise NearestNeighbor (le plus proche voisin) car c'est un masque ! 
            # On ne veut pas de pixels à 0.5, on veut des 0 ou des 1 purs.
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(sitk.Transform())
            mask = resampler.Execute(mask_raw)
        else:
            mask = mask_raw

        # Nettoyage brutal du masque pour être sûr que la tumeur = 1.
        mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=65535,
                                    insideValue=1, outsideValue=0)

        # On crée le fameux anneau autour de la tumeur
        peri = binary_peri_ring(mask, dilation_mm=peri_mm, label=1)

        # Astuce I/O : L'anneau péritumoral est le même peu importe la phase 
        # (car la tumeur ne bouge pas). On ne l'enregistre sur le disque QUE lors du 
        # traitement de la phase 0000 pour éviter d'écrire 4 fois le même fichier.
        if save_peri_dir is not None and phase_id == "0000": 
            save_peri_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(peri, str(save_peri_dir / f"{subject_id}_peri_mask.nii.gz"))

        # Instanciation de l'extracteur avec nos paramètres (normalize=False !)
        extractor = make_extractor(**extractor_params)
        
        # On lance l'extraction deux fois : une pour la tumeur, une pour le pourtour
        feats_tumor = execute_extract(extractor, img, mask, "tumor")
        feats_peri  = execute_extract(extractor, img, peri, "peri")

        # On initialise notre ligne de résultat avec l'identifiant du patient et de la phase
        feats = {
            "subject_id": subject_id,
            "phase_id": phase_id
        }
        
        # On fusionne les dictionnaires (les colonnes tumeur et péritumeur s'ajoutent à la ligne)
        feats.update(feats_tumor)
        feats.update(feats_peri)

        # On renvoie le succès (None signifie "pas d'erreur")
        return subject_id, phase_id, feats, None

    except Exception as e:
        # Si ce patient plante, on capture l'erreur et la renvoie sans faire crasher tout le script
        return subject_id, phase_id, {}, f"{type(e).__name__}: {e}"

# =====================================================================
# 5. L'ORCHESTRATEUR (Gestionnaire global et Flattening)
# =====================================================================

def extract_from_nnunet(
    imagesTr_dir: Path,
    labelsTr_dir: Path,
    out_dir: Path,
    peri_mm: float = 5.0,
    save_peri_masks: bool = False,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Règle d'or absolue pour ce script : on ne normalise PAS ici, ça a été fait avant.
    extractor_params = {
        "binWidth": 25.0, 
        "normalize": False, 
        "label": 1,
        "resampledPixelSpacing": [1, 1, 1], # Forcer des cubes de 1x1x1 mm
        "interpolator": "sitkBSpline"       # Interpolation douce pour l'image
    }
    
    # Création de la liste des tâches
    raw_tasks = find_multiphase_tasks_nnunet(imagesTr_dir, labelsTr_dir)
    tasks = []
    for task in raw_tasks:
        tasks.append((task, peri_mm, (out_dir / "peritumoral_masks") if save_peri_masks else None, extractor_params))

    if not tasks:
        raise RuntimeError("Aucune tâche générée. Vérifiez les chemins imagesTr et labelsTr.")

    # Si l'utilisateur ne dit rien, on prends tous les cœurs du processeur moins 1 pour éviter de geler l'ordinateur.
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    errors: List[Dict] = []

    # --- LANCEMENT DU MULTIPROCESSING ---
    with Pool(processes=n_jobs) as pool:
        # imap_unordered gère les tâches au fur et à mesure sans se soucier de l'ordre.
        # C'est plus rapide, et on triera le tableau final nous-mêmes avec Pandas.
        for subject_id, phase_id, feats, err in tqdm(pool.imap_unordered(_process_one, tasks),
                                                     total=len(tasks), desc="Extraction Radiomique"):
            if err is None:
                results.append(feats)
            else:
                errors.append({"subject_id": subject_id, "phase_id": phase_id, "error": err})

    # =================================================================
    # --- LA MAGIE DE L'APLATISSEMENT (FLATTENING / PIVOT) ---
    # =================================================================
    
    if results:
        # À ce stade, on a un tableau "long" : Patient01 possède 4 lignes (une par phase).
        df_long = pd.DataFrame(results).sort_values(by=["subject_id", "phase_id"])
        
        print("\n[INFO] Aplatissement (Flattening) des données IRM multi-phases...")
        
        # 1. Le Pivot : on demande à Pandas de mettre subject_id en index de gauche, 
        # et de créer de nouvelles colonnes en fonction de la phase_id.
        # Le Patient01 n'a plus qu'une seule ligne !
        df_wide = df_long.pivot(index="subject_id", columns="phase_id")
        
        # 2. Le Renommage : Pandas a créé un "MultiIndex" moche. Par exemple, le nom de la 
        # colonne est devenu un tuple : ('tumor_glcm_Contrast', '0000').
        # On boucle sur ces tuples pour les écraser en un nom plat : 'phase0000_tumor_glcm_Contrast'.
        flat_columns = []
        for feature_name, phase in df_wide.columns:
            flat_columns.append(f"phase{phase}_{feature_name}")
        
        df_wide.columns = flat_columns
        
        # 3. La finalisation : subject_id était coincé comme "Index" de la table.
        # On le réintègre comme une colonne normale, indispensable pour le script de label Tagg après !
        df_results = df_wide.reset_index()
    else:
        df_results = pd.DataFrame()
        
    df_errors  = pd.DataFrame(errors).sort_values(["subject_id", "phase_id"]) if errors else pd.DataFrame()

    # --- SAUVEGARDE SUR LE DISQUE ---
    # On ajoute bien "_FLATTENED" dans le nom du fichier pour indiquer que le travail d'aplatissement a été fait.
    csv_path  = out_dir / "radiomics_results_mri_FLATTENED.csv"
    xlsx_path = out_dir / "radiomics_results_mri_FLATTENED.xlsx"
    meta_path = out_dir / "run_metadata.json"

    df_results.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="features", index=False)
        if not df_errors.empty:
            df_errors.to_excel(writer, sheet_name="errors", index=False)

    # Sauvegarde des métadonnées pour se souvenir comment le script a été paramétré lors de ce lancement
    unique_patients = len(set([t["subject_id"] for t in raw_tasks]))
    meta = {
        "n_subjects_input": unique_patients,
        "n_phases_total_processed": len(results),
        "n_errors": len(errors),
        "peri_ring_mm": peri_mm,
        "extractor_params": extractor_params
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n--- Traitement terminé ---")
    print(f"Fichiers sauvegardés dans : {out_dir}")

    return df_results, df_errors

if __name__ == "__main__":
    # --- CONFIGURATION UTILISATEUR ---
    # Remplacement de ces chemins par les vrais dossiers contenant les données normalisées nécessaire.
    IMAGES_TR = Path("/chemin/vers/nnunet_raw/Dataset001_DCE/imagesTr")
    LABELS_TR = Path("/chemin/vers/nnunet_raw/Dataset001_DCE/labelsTr")
    OUTPUT_DIR = Path("./results_radiomics")
    
    extract_from_nnunet(
        imagesTr_dir=IMAGES_TR,
        labelsTr_dir=LABELS_TR,
        out_dir=OUTPUT_DIR,
        peri_mm=5.0,
        save_peri_masks=True,
        n_jobs=4
    )

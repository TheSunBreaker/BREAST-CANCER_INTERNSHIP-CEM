#!/usr/bin/env python3
r"""
Extracteur de Masques DICOM SEG (V2 - Support Longitudinal).
Lit les segmentations isolées par l'ingesteur V4 et les convertit en NIfTI via pydicom-seg.
S'adapte dynamiquement à la nouvelle structure temporelle (Baseline, Follow-ups, Orphelins).
"""

import os
import pydicom
import pydicom_seg
import SimpleITK as sitk
import numpy as np
import argparse

def convert_dicom_seg_to_nifti(dcm_path: str) -> sitk.Image:
    """
    Lit un DICOM SEG et extrait la matrice de pixels 3D en objet SimpleITK.
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        
        # Sécurité : vérifier que c'est bien une segmentation
        if getattr(ds, "Modality", "") != "SEG":
            return None
            
        reader = pydicom_seg.MultiClassReader()
        result = reader.read(ds)
        img = result.image
        
        # Binarisation stricte pour nnU-Net : Les classes multiples deviennent 1 (Lésion), fond = 0.
        # Utile si un radiologue a par erreur mis une valeur de 2 pour la même lésion.
        arr = sitk.GetArrayFromImage(img)
        arr = (arr > 0).astype(np.uint8)
        clean_img = sitk.GetImageFromArray(arr)
        clean_img.CopyInformation(img)
        
        return clean_img
        
    except Exception as e:
        print(f"   [ERREUR] Impossible de lire le SEG {os.path.basename(dcm_path)}: {e}")
        return None

def process_patient_masks(project_root: str, mask_prefix: str):
    """
    Parcourt la base de données, repère dynamiquement tous les dossiers de masques 
    (Baseline et suivis longitudinaux), convertit les DICOM SEG en NIfTI et les range
    dans le bon dossier 'mask' temporel correspondant.
    """
    if not os.path.exists(project_root):
        return

    print(f"\n=== EXTRACTION DES MASQUES DANS {project_root} (Préfixe: {mask_prefix}) ===")
    
    patients = [p for p in os.listdir(project_root) if os.path.isdir(os.path.join(project_root, p))]
    masques_generes = 0
    
    for patient_id in patients:
        patient_dir = os.path.join(project_root, patient_id)
        
        # 1. DÉTECTION DYNAMIQUE DES DOSSIERS DE MASQUES
        # On cherche tous les dossiers commençant par "dicom_mask_rm", "dicom_mask_pet", ou "dicom_mask_orphelins"
        # Cela permet d'attraper la Baseline ET les suivis (ex: dicom_mask_rm_20230514_1430)
        mask_dirs = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d)) and d.startswith(mask_prefix)]
        
        for source_mask_folder in mask_dirs:
            source_mask_dir = os.path.join(patient_dir, source_mask_folder)
            
            # 2. ROUTAGE VERS LE BON DOSSIER NIFTI CIBLE
            # Si c'est le dossier de base (dicom_mask_rm), la cible est "mask".
            # Si c'est un suivi (dicom_mask_rm_YYYYMMDD), la cible est "mask_YYYYMMDD".
            if source_mask_folder == mask_prefix:
                dest_mask_name = "mask"
            else:
                # On extrait la partie temporelle (le suffixe après le préfixe)
                suffixe_temporel = source_mask_folder.replace(mask_prefix, "")
                dest_mask_name = f"mask{suffixe_temporel}"
            
            dest_mask_dir = os.path.join(patient_dir, dest_mask_name)
            os.makedirs(dest_mask_dir, exist_ok=True)
            
            # 3. CONVERSION DE CHAQUE SÉRIE
            for sub_dir in os.listdir(source_mask_dir):
                series_path = os.path.join(source_mask_dir, sub_dir)
                if not os.path.isdir(series_path):
                    continue
                    
                # sub_dir ressemble maintenant à "DUKE_001_A1B2C". 
                # On extrait juste le UID pour le nom de fichier final.
                uid_suffix = sub_dir.split("_")[-1] if "_" in sub_dir else sub_dir
                
                for file_name in os.listdir(series_path):
                    dcm_path = os.path.join(series_path, file_name)
                    
                    # On lance la conversion
                    mask_img = convert_dicom_seg_to_nifti(dcm_path)
                    
                    if mask_img is not None:
                        # Le nom final sera propre : ex: DUKE_001_mask_A1B2C.nii.gz
                        dest_name = f"{patient_id}_mask_{uid_suffix}.nii.gz"
                        dest_path = os.path.join(dest_mask_dir, dest_name)
                        
                        sitk.WriteImage(mask_img, dest_path)
                        print(f" -> Masque ({source_mask_folder}) converti : {dest_path}")
                        masques_generes += 1

    print(f"=== {masques_generes} MASQUES (SEG) GÉNÉRÉS AU TOTAL ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertisseur de Masques DICOM SEG vers NIfTI (Longitudinal).")
    parser.add_argument("--mri_root", default="./Base_IRM", help="Racine des données IRM")
    parser.add_argument("--petct_root", default="./Base_PETCT", help="Racine des données PET/CT")
    args = parser.parse_args()
    
    # Lancement sur les masques IRM normaux
    process_patient_masks(args.mri_root, "dicom_mask_rm")
    # Lancement sur les masques PET/CT normaux
    process_patient_masks(args.petct_root, "dicom_mask_pet")
    
    # Lancement optionnel sur les masques Orphelins (s'il y en a eu de générés par l'ingesteur V4)
    # Ils seront placés dans des dossiers "mask_orphelins_YYYYMMDD"
    process_patient_masks(args.mri_root, "dicom_mask_orphelins")
    process_patient_masks(args.petct_root, "dicom_mask_orphelins")

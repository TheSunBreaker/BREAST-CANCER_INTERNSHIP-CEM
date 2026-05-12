#!/usr/bin/env python3
r"""
Extracteur de Masques DICOM SEG (Niveau 1.8).
Lit les segmentations isolées par l'ingesteur et les convertit en NIfTI via pydicom-seg.
Garde chaque masque indépendant (pour l'analyse longitudinale ou multi-lésions).
"""

import os
import pydicom
import pydicom_seg
import SimpleITK as sitk
import numpy as np
import argparse

def convert_dicom_seg_to_nifti(dcm_path: str) -> sitk.Image:
    """
    Lit un DICOM SEG et extrait la grille de pixels 3D en SimpleITK.
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        
        if getattr(ds, "Modality", "") != "SEG":
            return None
            
        reader = pydicom_seg.MultiClassReader()
        result = reader.read(ds)
        img = result.image
        
        # Formatage strict pour nnU-Net (0 et 1)
        arr = sitk.GetArrayFromImage(img)
        arr = (arr > 0).astype(np.uint8)
        clean_img = sitk.GetImageFromArray(arr)
        clean_img.CopyInformation(img)
        
        return clean_img
        
    except Exception as e:
        print(f"   [ERREUR] Impossible de lire le SEG {os.path.basename(dcm_path)}: {e}")
        return None

def process_patient_masks(project_root: str, mask_source_dir_name: str):
    """
    Parcourt la base de données, convertit les masques et les range individuellement.
    """
    if not os.path.exists(project_root):
        return

    print(f"\n=== EXTRACTION DES MASQUES DANS {project_root} ===")
    
    patients = [p for p in os.listdir(project_root) if os.path.isdir(os.path.join(project_root, p))]
    masques_generes = 0
    
    for patient_id in patients:
        patient_dir = os.path.join(project_root, patient_id)
        source_mask_dir = os.path.join(patient_dir, mask_source_dir_name)
        
        if not os.path.exists(source_mask_dir):
            continue
            
        dest_mask_dir = os.path.join(patient_dir, "mask")
        os.makedirs(dest_mask_dir, exist_ok=True)
        
        # On parcourt chaque dossier de série de masque (les series_uid[-5:])
        for sub_dir in os.listdir(source_mask_dir):
            series_path = os.path.join(source_mask_dir, sub_dir)
            if not os.path.isdir(series_path):
                continue
                
            for file_name in os.listdir(series_path):
                dcm_path = os.path.join(series_path, file_name)
                
                # Conversion d'UN seul fichier
                mask_img = convert_dicom_seg_to_nifti(dcm_path)
                
                if mask_img is not None:
                    # Le nom inclut le sous-dossier (series_uid) pour éviter les écrasements
                    dest_name = f"{patient_id}_mask_{sub_dir}.nii.gz"
                    dest_path = os.path.join(dest_mask_dir, dest_name)
                    
                    sitk.WriteImage(mask_img, dest_path)
                    print(f" -> Masque généré : {dest_path}")
                    masques_generes += 1

    print(f"=== {masques_generes} MASQUES (SEG) GÉNÉRÉS AU TOTAL ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertisseur de Masques DICOM SEG vers NIfTI.")
    parser.add_argument("--mri_root", default="./Base_IRM", help="Racine des données IRM")
    parser.add_argument("--petct_root", default="./Base_PETCT", help="Racine des données PET/CT")
    args = parser.parse_args()
    
    process_patient_masks(args.petct_root, "dicom_mask_pet")
    process_patient_masks(args.mri_root, "dicom_mask_rm")

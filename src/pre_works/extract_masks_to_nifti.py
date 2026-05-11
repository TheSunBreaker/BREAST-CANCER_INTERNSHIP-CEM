#!/usr/bin/env python3
r"""
Extracteur de Masques DICOM SEG (Niveau 1.8).
Lit les segmentations isolées par l'ingesteur, les convertit en NIfTI via pydicom-seg,
et les fusionne si un patient possède plusieurs lésions pour garantir une "Single Source of Truth".
"""

import os
import glob
import pydicom
import pydicom_seg
import SimpleITK as sitk
import numpy as np
import argparse

def merge_sitk_masks(mask_list: list) -> sitk.Image:
    """
    Fusionne une liste de masques SimpleITK en un seul (Union logique OR).
    Gère automatiquement les différences de taille, d'espacement et d'origine.
    """
    if not mask_list:
        return None
    
    # On prend le premier masque comme référence géométrique absolue
    ref_img = mask_list[0]
    merged_arr = sitk.GetArrayFromImage(ref_img)
    
    # Préparation du "Resampler" (Notre standardiseur spatial)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # IMPORTANT: Pas de lissage sur des masques !
    resampler.SetDefaultPixelValue(0) # Ce qui déborde de la grille devient du fond (0)
    
    # On additionne les autres
    for i in range(1, len(mask_list)):
        moving_img = mask_list[i]
        
        # 1. On force le masque courant à épouser EXACTEMENT la grille de référence
        # Même si sa taille d'origine était [100, 100, 50] et la ref [512, 512, 100]
        aligned_img = resampler.Execute(moving_img)
        
        # 2. Maintenant, on est mathématiquement certain que les arrays ont la même 'shape'
        arr = sitk.GetArrayFromImage(aligned_img)
        
        # 3. Union logique (Si pixel > 0 dans A ou B -> devient 1)
        merged_arr = np.logical_or(merged_arr, arr).astype(np.uint8)
        
    # On reconstruit l'image SimpleITK
    result_img = sitk.GetImageFromArray(merged_arr)
    result_img.CopyInformation(ref_img)
    return result_img

def convert_dicom_seg_to_nifti(dicom_paths: list) -> sitk.Image:
    """
    Lit un (ou plusieurs) DICOM SEG et extrait la grille de pixels 3D en SimpleITK.
    """
    extracted_masks = []
    
    # pydicom-seg possède un lecteur spécialisé pour ce type de DICOM
    reader = pydicom_seg.MultiClassReader()
    
    for dcm_path in dicom_paths:
        try:
            ds = pydicom.dcmread(dcm_path)
            
            # On vérifie que c'est bien un SEG
            if getattr(ds, "Modality", "") != "SEG":
                continue
                
            # Extraction de la segmentation
            result = reader.read(ds)
            
            # L'objet result contient l'image SimpleITK directement !
            img = result.image
            
            # On s'assure que c'est un format UInt8 (Règle d'or pour nnU-Net)
            img = sitk.Cast(img, sitk.sitkUInt8)
            extracted_masks.append(img)
            
        except Exception as e:
            print(f"   [ERREUR] Impossible de lire le DICOM SEG {os.path.basename(dcm_path)}: {e}")
            
    return merge_sitk_masks(extracted_masks)

def process_patient_masks(project_root: str, mask_source_dir_name: str):
    """
    Parcourt la base de données, trouve les masques bruts, les convertit et les range.
    """
    if not os.path.exists(project_root):
        return

    print(f"\n=== ANALYSE DES MASQUES DANS {project_root} ===")
    
    patients = [p for p in os.listdir(project_root) if os.path.isdir(os.path.join(project_root, p))]
    patients_traites = 0
    
    for patient_id in patients:
        patient_dir = os.path.join(project_root, patient_id)
        source_mask_dir = os.path.join(patient_dir, mask_source_dir_name)
        
        if not os.path.exists(source_mask_dir):
            continue
            
        # Création du dossier de destination standardisé ("mask")
        dest_mask_dir = os.path.join(patient_dir, "mask")
        os.makedirs(dest_mask_dir, exist_ok=True)
        dest_mask_path = os.path.join(dest_mask_dir, f"{patient_id}_mask.nii.gz")
        
        # On cherche tous les fichiers DICOM dans les sous-dossiers du masque
        dicom_files = []
        for root, _, files in os.walk(source_mask_dir):
            for f in files:
                dicom_files.append(os.path.join(root, f))
                
        if not dicom_files:
            continue
            
        print(f" -> Extraction pour {patient_id} ({len(dicom_files)} fichier(s) trouvés)...")
        
        # Conversion
        final_mask = convert_dicom_seg_to_nifti(dicom_files)
        
        if final_mask is not None:
            # Sécurité géométrique : Forcer les valeurs à 0 et 1 (NearestNeighbor implicite)
            arr = sitk.GetArrayFromImage(final_mask)
            arr = (arr > 0).astype(np.uint8)
            final_mask_clean = sitk.GetImageFromArray(arr)
            final_mask_clean.CopyInformation(final_mask)
            
            sitk.WriteImage(final_mask_clean, dest_mask_path)
            print(f"    [SUCCÈS] Masque NIfTI généré : {dest_mask_path}")
            patients_traites += 1
        else:
            # Si pydicom-seg n'a rien trouvé, c'est que ce sont peut-être des RTSTRUCT
            print(f"    [ATTENTION] Aucun pixel extrait. S'agit-il de RTSTRUCT au lieu de SEG ?")

    print(f"=== {patients_traites} PATIENTS TRAITÉS AVEC SUCCÈS ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertisseur de Masques DICOM SEG vers NIfTI.")
    parser.add_argument("--mri_root", default="./Base_IRM", help="Racine des données IRM")
    parser.add_argument("--petct_root", default="./Base_PETCT", help="Racine des données PET/CT")
    args = parser.parse_args()
    
    # On lance l'extraction pour les deux bases
    process_patient_masks(args.petct_root, "dicom_mask_pet")
    process_patient_masks(args.mri_root, "dicom_mask_rm")

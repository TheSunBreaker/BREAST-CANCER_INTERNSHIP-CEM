#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
ORCHESTRATEUR DE TENSEURS POUR CNN MULTIMODAL (ResNet / DenseNet)
=============================================================================
Rôle : 
Extrait des cubes 3D stricts (ex: 96x96x96) centrés sur le centre de masse 
global de la charge tumorale de la patiente. 
Maintient l'intégrité géométrique (Origine, Direction) pour validation visuelle.

Modalités gérées :
- IRM (DCE) : Forcé en isotropique 1.0 x 1.0 x 1.0 mm
- PET/CT    : Forcé en isotropique 2.0 x 2.0 x 2.0 mm (Standard pour éviter l'interpolation excessive du PET)
=============================================================================
"""

import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def make_isotropic_crop_pad(
    image_path: str,
    mask_path: str,
    base_out_img_path: str,
    base_out_mask_path: str,
    target_spacing: tuple,
    target_shape: tuple,
    pad_value: float
) -> bool:
    """
    Ré-échantillonne l'image et le masque à l'espacement cible, calcule le centre
    de gravité global de toutes les tumeurs de la patiente, puis génère le tenseur cible.
    """
    try:
        # 1. CHARGEMENT
        img = sitk.ReadImage(image_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        
        # =====================================================================
        # 2. RÉ-ÉCHANTILLONNAGE ISOTROPIQUE 
        # =====================================================================
        orig_size = img.GetSize()       
        orig_spacing = img.GetSpacing() 
        
        # Calcul de la nouvelle taille matricielle pour conserver le volume physique exact
        new_size = [
            int(round(orig_size[0] * (orig_spacing[0] / target_spacing[0]))),
            int(round(orig_size[1] * (orig_spacing[1] / target_spacing[1]))),
            int(round(orig_size[2] * (orig_spacing[2] / target_spacing[2])))
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetDefaultPixelValue(pad_value)
        
        # Image : B-Spline pour préserver la douceur des tissus (Interpolation cubique)
        resampler.SetInterpolator(sitk.sitkBSpline)
        img_iso = resampler.Execute(img)
        
        # Masque : NearestNeighbor STRICT pour garder le binaire pur (0 ou 1)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0.0)
        mask_iso = resampler.Execute(mask)
        
        # =====================================================================
        # 3. CALCUL DU CENTRE DE MASSE GLOBAL (Approche "Patiente")
        # =====================================================================
        mask_np = sitk.GetArrayFromImage(mask_iso) # Converti en Numpy (Z, Y, X)
        img_np = sitk.GetArrayFromImage(img_iso)
        
        # np.where récupère les coordonnées de TOUS les voxels tumoraux (même s'il y a 3 foyers distincts)
        coords = np.where(mask_np > 0)
        
        if len(coords[0]) == 0:
            print(f"   [ERREUR] Masque vide pour {os.path.basename(image_path)}")
            return False
            
        # Moyenne des coordonnées = Centre de gravité global de la maladie
        center_z = int(np.mean(coords[0]))
        center_y = int(np.mean(coords[1]))
        center_x = int(np.mean(coords[2]))
        
        # =====================================================================
        # 4. CRÉATION DES TENSEURS (TOILES)
        # =====================================================================
        # On unpack target_shape (X, Y, Z) vers le repère numpy (Z, Y, X)
        tz, ty, tx = target_shape[2], target_shape[1], target_shape[0]
        
        crop_img_np = np.full((tz, ty, tx), pad_value, dtype=np.float32)
        crop_mask_np = np.zeros((tz, ty, tx), dtype=np.uint8)
        
        # =====================================================================
        # 5. CROP ET PADDING DYNAMIQUE
        # =====================================================================
        # Calcul des limites de lecture sur l'image source
        # max/min agissent comme des pare-chocs : si la tumeur est contre le bord du sein, on ne plante pas.
        z_min, z_max = max(0, center_z - tz//2), min(img_np.shape[0], center_z + tz//2 + (tz%2))
        y_min, y_max = max(0, center_y - ty//2), min(img_np.shape[1], center_y + ty//2 + (ty%2))
        x_min, x_max = max(0, center_x - tx//2), min(img_np.shape[2], center_x + tx//2 + (tx%2))
        
        # Calcul des limites d'écriture sur la toile PyTorch
        # Si on a buté contre un bord source, ces coordonnées ajustent le dessin pour le centrer
        cz_min = (tz//2) - (center_z - z_min)
        cz_max = cz_min + (z_max - z_min)
        cy_min = (ty//2) - (center_y - y_min)
        cy_max = cy_min + (y_max - y_min)
        cx_min = (tx//2) - (center_x - x_min)
        cx_max = cx_min + (x_max - x_min)
        
        # Transfert des matrices
        crop_img_np[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max] = img_np[z_min:z_max, y_min:y_max, x_min:x_max]
        # Tous les foyers tumoraux entrant dans le champ de vision de 96^3 sont étiquetés à 1
        crop_mask_np[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max] = (mask_np[z_min:z_max, y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
        
        # =====================================================================
        # 6. RESTAURATION GÉOMÉTRIQUE & SAUVEGARDE
        # =====================================================================
        final_img = sitk.GetImageFromArray(crop_img_np)
        final_mask = sitk.GetImageFromArray(crop_mask_np)
        
        # On restaure l'espacement
        final_img.SetSpacing(target_spacing)
        final_mask.SetSpacing(target_spacing)
        
        # FIX CRITIQUE : Calcul de la nouvelle Origine spatiale du coin supérieur gauche du cube.
        # Permet d'ouvrir le fichier .nii.gz dans ITK-SNAP et de voir qu'il s'aligne parfaitement avec l'image globale.
        # Attention, TransformIndexToPhysicalPoint exige (X, Y, Z).
        new_origin = img_iso.TransformIndexToPhysicalPoint((int(x_min), int(y_min), int(z_min)))
        final_img.SetOrigin(new_origin)
        final_mask.SetOrigin(new_origin)
        
        final_img.SetDirection(img_iso.GetDirection())
        final_mask.SetDirection(img_iso.GetDirection())
        
        os.makedirs(os.path.dirname(base_out_img_path), exist_ok=True)
        
        # Sauvegarde NIfTI (Validation Humaine)
        sitk.WriteImage(final_img, base_out_img_path)
        sitk.WriteImage(final_mask, base_out_mask_path)
        
        # Sauvegarde NumPy Array binaire (Lecture ultra-rapide pour le PyTorch DataLoader)
        np.save(base_out_img_path.replace('.nii.gz', '.npy'), crop_img_np)
        np.save(base_out_mask_path.replace('.nii.gz', '.npy'), crop_mask_np)
        
        return True
        
    except Exception as e:
        print(f"   [ERREUR FATALE] {os.path.basename(image_path)} : {e}")
        return False

def orchestrate_custom_cnn_tensors_from_nnunet(
    imagesTr_dir: str, 
    labelsTr_dir: str,
    out_dir: str, 
    modality: str = "MRI", 
    target_shape: tuple = (96, 96, 96) # Option 1 sélectionnée
):
    print(f"\n--- Génération Tenseurs CNN ({modality}) depuis nnU-Net | Cible : {target_shape} ---")
    
    # La liste stricte des patients est déduite des masques disponibles dans labelsTr
    patients = [f.replace('.nii.gz', '') for f in os.listdir(labelsTr_dir) if f.endswith('.nii.gz')]
    valid_count = 0
    
    for subj in tqdm(patients, desc="Traitement Patients"):
        mask_path = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
        out_subj_dir = os.path.join(out_dir, subj)
        
        # ==========================================================
        # PIPELINE IRM (DCE Multi-phases)
        # ==========================================================
        if modality == "MRI":
            target_spacing = (1.0, 1.0, 1.0) # Résolution fine à 1mm
            
            img_files = sorted(glob.glob(os.path.join(imagesTr_dir, f"{subj}_*.nii.gz")))
            for idx, img_path in enumerate(img_files):
                out_img = os.path.join(out_subj_dir, f"{subj}_MRI_phase{idx}.nii.gz")
                out_mask = os.path.join(out_subj_dir, f"{subj}_MRI_mask.nii.gz")
                
                success = make_isotropic_crop_pad(
                    image_path=img_path, mask_path=mask_path,
                    base_out_img_path=out_img, base_out_mask_path=out_mask,
                    target_spacing=target_spacing, target_shape=target_shape, pad_value=0.0
                )
                if success: valid_count += 1

        # ==========================================================
        # PIPELINE PET/CT (Multimodal Aligné)
        # ==========================================================
        elif modality == "PETCT":
            # On maintient 2mm pour le PET/CT, ce qui donnera avec un tenseur 96^3 un FOV physique énorme de 19.2 cm !
            target_spacing = (2.0, 2.0, 2.0) 
            
            pet_path = os.path.join(imagesTr_dir, f"{subj}_0000.nii.gz")
            ct_path = os.path.join(imagesTr_dir, f"{subj}_0001.nii.gz")
            
            if not os.path.exists(pet_path) or not os.path.exists(ct_path):
                continue
                
            out_mask = os.path.join(out_subj_dir, f"{subj}_PETCT_mask.nii.gz")
            
            # Extraction CT avec padding d'air (-1000 HU)
            out_ct = os.path.join(out_subj_dir, f"{subj}_CT.nii.gz")
            success_ct = make_isotropic_crop_pad(
                image_path=ct_path, mask_path=mask_path,
                base_out_img_path=out_ct, base_out_mask_path=out_mask,
                target_spacing=target_spacing, target_shape=target_shape, pad_value=-1000.0
            )
            
            # Extraction PET avec padding vide (0.0 SUV)
            out_pet = os.path.join(out_subj_dir, f"{subj}_PET.nii.gz")
            success_pet = make_isotropic_crop_pad(
                image_path=pet_path, mask_path=mask_path,
                base_out_img_path=out_pet, base_out_mask_path=out_mask,
                target_spacing=target_spacing, target_shape=target_shape, pad_value=0.0
            )
            
            if success_ct and success_pet: 
                valid_count += 1
                
    print(f"\n[Terminé] {valid_count} Tenseurs générés avec succès et prêts pour PyTorch !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # On pointe désormais directement sur la structure finale nnU-Net !
    parser.add_argument("--src", required=True, help="Dossier racine nnUNet_raw/DatasetXXX")
    parser.add_argument("--out", required=True, help="Dossier de destination des Tenseurs (.npy et .nii.gz)")
    parser.add_argument("--modality", choices=["MRI", "PETCT"], default="MRI")
    args = parser.parse_args()
    
    # Déduction automatique des sous-dossiers standards
    img_dir = os.path.join(args.src, "imagesTr")
    lbl_dir = os.path.join(args.src, "labelsTr")
    
    # On force la taille de 96x96x96
    orchestrate_custom_cnn_tensors_from_nnunet(img_dir, lbl_dir, args.out, args.modality, target_shape=(96, 96, 96))
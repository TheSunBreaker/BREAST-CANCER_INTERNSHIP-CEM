#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrateur pour Modèle Deep Learning "Fait Main" (Custom CNN / PyTorch).
-------------------------------------------------------------------------
Objectif : Prendre les images standardisées (Niveau 2 (données standardiées, alignées, normalisées, etc., et prêtes pour le deep learning), avec leur FOV (zone d'intérrêt, dans notre cas la zone seins) complet),
forcer un espacement isotropique parfait (ex: cubes de 1x1x1 mm ou 2x2x2 mm),
et générer des Tenseurs 3D de taille fixe (ex: 64x64x64) centrés sur CHAQUE tumeur.
Gère nativement les patientes ayant plusieurs tumeurs isolées (Cancers Multifocaux).
"""

import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import label # Pour détecter les tumeurs séparées

def make_isotropic_crop_pad(
    image_path: str,
    mask_path: str,
    base_out_img_path: str,
    base_out_mask_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    target_shape: tuple = (64, 64, 64),
    pad_value: float = 0.0
) -> int:
    """
    Ré-échantillonne en isotropique, détecte le nombre de tumeurs distinctes, 
    et crop/pad un tenseur cible `target_shape` pour CHAQUE tumeur.
    Retourne le nombre de tenseurs (lésions) générés avec succès.
    """
    try:
        # Chargement (Image en Float32 continu, Masque en UInt8 binaire)
        img = sitk.ReadImage(image_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        
        # =====================================================================
        # 1. RÉ-ÉCHANTILLONNAGE ISOTROPIQUE 
        # =====================================================================
        orig_size = img.GetSize()       
        orig_spacing = img.GetSpacing() 
        
        # Règle de 3 mathématique pour préserver le volume physique lors du changement d'espacement
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
        
        # Image : BSpline (Ordre 3) pour lisser les gradients
        resampler.SetInterpolator(sitk.sitkBSpline)
        img_iso = resampler.Execute(img)
        
        # Masque : NearestNeighbor STRICT pour éviter de créer de la fausse tumeur à 0.5
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0.0)
        mask_iso = resampler.Execute(mask)
        
        # =====================================================================
        # 2. DÉTECTION DES TUMEURS MULTIPLES (Composantes Connexes)
        # =====================================================================
        mask_np = sitk.GetArrayFromImage(mask_iso) # (Z, Y, X)
        img_np = sitk.GetArrayFromImage(img_iso)
        
        # La fonction label regroupe les pixels adjacents. 
        # Si y'a 2 tumeurs séparées par du tissu sain, num_lesions vaudra 2.
        # labeled_mask contiendra des 1 pour la lésion 1, et des 2 pour la lésion 2.
        labeled_mask, num_lesions = label(mask_np > 0)
        
        if num_lesions == 0:
            print(f"   [ERREUR] Masque vide (aucune tumeur trouvée) pour {os.path.basename(image_path)}")
            return 0
            
        lesions_generated = 0
        tz, ty, tx = target_shape[2], target_shape[1], target_shape[0] # PyTorch veut du Z, Y, X
        
        # =====================================================================
        # 3. BOUCLE SUR CHAQUE LÉSION (Multifocale)
        # =====================================================================
        for lesion_idx in range(1, num_lesions + 1):
            
            # On cherche les coordonnées (Z,Y,X) appartenant UNIQUEMENT à cette lésion
            coords = np.where(labeled_mask == lesion_idx)
            
            # Centre de masse de CETTE lésion
            center_z = int(np.mean(coords[0]))
            center_y = int(np.mean(coords[1]))
            center_x = int(np.mean(coords[2]))
            
            # Création des "Toiles" vierges (Tenseurs PyTorch-ready)
            crop_img_np = np.full((tz, ty, tx), pad_value, dtype=np.float32)
            crop_mask_np = np.zeros((tz, ty, tx), dtype=np.uint8)
            
            # =====================================================================
            # 4. CROP & PADDING (Gestion dynamique des bords de l'image)
            # =====================================================================
            # Limites sur l'image source (max et min empêchent le plantage si la tumeur touche le bord de l'image)
            z_min, z_max = max(0, center_z - tz//2), min(img_np.shape[0], center_z + tz//2 + (tz%2))
            y_min, y_max = max(0, center_y - ty//2), min(img_np.shape[1], center_y + ty//2 + (ty%2))
            x_min, x_max = max(0, center_x - tx//2), min(img_np.shape[2], center_x + tx//2 + (tx%2))
            
            # Limites correspondantes sur la toile cible (décale le crop si on a tapé un bord source)
            cz_min = (tz//2) - (center_z - z_min)
            cz_max = cz_min + (z_max - z_min)
            cy_min = (ty//2) - (center_y - y_min)
            cy_max = cy_min + (y_max - y_min)
            cx_min = (tx//2) - (center_x - x_min)
            cx_max = cx_min + (x_max - x_min)
            
            # Transfert des matrices
            crop_img_np[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max] = img_np[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # CRITIQUE : Pour le tenseur final, on veut que TOUTE lésion présente dans 
            # ce champ de vision 64x64 soit à 1 (pas seulement la lésion courante). 
            # On reprend donc le mask_np global (booléen).
            crop_mask_np[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max] = (mask_np[z_min:z_max, y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
            
            # =====================================================================
            # 5. GÉNÉRATION DES NOMS DE FICHIERS DYNAMIQUES
            # =====================================================================
            # Si le nom de base était "Patient01_MRI_phase0.nii.gz", 
            # ça devient "Patient01_MRI_phase0_lesion1.nii.gz"
            lesion_suffix = f"_lesion{lesion_idx}"
            out_img = base_out_img_path.replace(".nii.gz", f"{lesion_suffix}.nii.gz")
            out_mask = base_out_mask_path.replace(".nii.gz", f"{lesion_suffix}.nii.gz")
            
            # =====================================================================
            # 6. SAUVEGARDE NIFTI (Pour validation visuelle) & NUMPY (Pour PyTorch)
            # =====================================================================
            final_img = sitk.GetImageFromArray(crop_img_np)
            final_mask = sitk.GetImageFromArray(crop_mask_np)
            final_img.SetSpacing(target_spacing)
            final_mask.SetSpacing(target_spacing)
            
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            os.makedirs(os.path.dirname(out_mask), exist_ok=True)
            
            sitk.WriteImage(final_img, out_img)
            sitk.WriteImage(final_mask, out_mask)
            
            # Sauvegarde en binaire rapide pour le DataLoader
            np.save(out_img.replace('.nii.gz', '.npy'), crop_img_np)
            np.save(out_mask.replace('.nii.gz', '.npy'), crop_mask_np)
            
            lesions_generated += 1
            
        return lesions_generated
        
    except Exception as e:
        print(f"   [ERREUR FATALE] {os.path.basename(image_path)} : {e}")
        return 0

def orchestrate_custom_cnn_tensors_from_nnunet(
    imagesTr_dir: str, 
    labelsTr_dir: str,
    out_dir: str, 
    modality: str = "MRI", # "MRI" ou "PETCT"
    target_shape: tuple = (64, 64, 64)
):
    print(f"\n--- Génération Tenseurs CNN ({modality}) depuis nnU-Net | Cible : {target_shape} ---")
    
    # Dans nnU-Net, le dossier labelsTr contient exactement un fichier par patient (ex: Patient01.nii.gz)
    # C'est la méthode la plus sûre pour lister nos patients valides.
    patients = [f.replace('.nii.gz', '') for f in os.listdir(labelsTr_dir) if f.endswith('.nii.gz')]
    valid_count = 0
    
    for subj in tqdm(patients, desc="Traitement Patients"):
        mask_path = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
        out_subj_dir = os.path.join(out_dir, subj)
        
        # ==========================================================
        # BRANCHE IRM (DCE Multi-phases)
        # ==========================================================
        if modality == "MRI":
            target_spacing = (1.0, 1.0, 1.0)
            # Dans nnU-Net, les phases s'appellent Patient01_0000.nii.gz, Patient01_0001.nii.gz, etc.
            img_files = sorted(glob.glob(os.path.join(imagesTr_dir, f"{subj}_*.nii.gz")))
            
            for idx, img_path in enumerate(img_files):
                out_img = os.path.join(out_subj_dir, f"{subj}_MRI_phase{idx}.nii.gz")
                out_mask = os.path.join(out_subj_dir, f"{subj}_MRI_mask.nii.gz")
                
                success = make_isotropic_crop_pad(
                    image_path=img_path, mask_path=mask_path,
                    out_img_path=out_img, out_mask_path=out_mask,
                    target_spacing=target_spacing, target_shape=target_shape, pad_value=0.0
                )
                if success: valid_count += 1

        # ==========================================================
        # BRANCHE PET/CT (Double extraction multimodale)
        # ==========================================================
        elif modality == "PETCT":
            target_spacing = (2.0, 2.0, 2.0)
            
            # Dans nnU-Net, le canal 0000 est le PET, le canal 0001 est le CT
            pet_path = os.path.join(imagesTr_dir, f"{subj}_0000.nii.gz")
            ct_path = os.path.join(imagesTr_dir, f"{subj}_0001.nii.gz")
            
            if not os.path.exists(pet_path) or not os.path.exists(ct_path):
                continue
                
            out_mask = os.path.join(out_subj_dir, f"{subj}_PETCT_mask.nii.gz")
            
            # Traitement CT (Pad Air = -1000)
            out_ct = os.path.join(out_subj_dir, f"{subj}_CT.nii.gz")
            success_ct = make_isotropic_crop_pad(
                image_path=ct_path, mask_path=mask_path,
                out_img_path=out_ct, out_mask_path=out_mask,
                target_spacing=target_spacing, target_shape=target_shape, pad_value=-1000.0
            )
            
            # Traitement PET (Pad Vide = 0)
            out_pet = os.path.join(out_subj_dir, f"{subj}_PET.nii.gz")
            success_pet = make_isotropic_crop_pad(
                image_path=pet_path, mask_path=mask_path,
                out_img_path=out_pet, out_mask_path=out_mask,
                target_spacing=target_spacing, target_shape=target_shape, pad_value=0.0
            )
            
            if success_ct and success_pet: 
                valid_count += 1
                
    print(f"\n[Terminé] {valid_count} Tenseurs générés avec succès et prêts pour PyTorch !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Dossier des patients standardisés")
    parser.add_argument("--out", required=True, help="Dossier de destination")
    parser.add_argument("--modality", choices=["MRI", "PETCT"], default="MRI")
    args = parser.parse_args()
    
    orchestrate_custom_cnn_tensors(args.src, args.out, args.modality)

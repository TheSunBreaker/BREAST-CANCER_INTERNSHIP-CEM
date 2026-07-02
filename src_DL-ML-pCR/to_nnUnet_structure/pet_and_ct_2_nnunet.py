#!/usr/bin/env python3
"""
Orchestrateur PET/CT vers nnU-Net V2.
Copie le PET (référence absolue) et aligne strictement le CT et le Masque sur sa géométrie
via le module de standardisation spatiale idempotent.
"""

import os
import glob
import json
import shutil
import argparse

# Importation de notre nouveau couteau suisse géométrique
from utils.spatial_standardizer import enforce_strict_alignment

def prepare_pet_ct_for_nnunet(
    subjects_dir: str,
    nnunet_root: str,
    dataset_id: int = 2,
    dataset_name_suffix: str = "BreastPETCT",
    is_inference: bool = False
):
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_suffix}"
    nnunet_raw = os.path.join(nnunet_root, "nnUNet_raw", dataset_name)
    
    # MODIFICATION : imagesTs si inférence, sinon imagesTr
    target_images_dir = os.path.join(nnunet_raw, "imagesTs" if is_inference else "imagesTr")
    os.makedirs(target_images_dir, exist_ok=True)

    # On ne crée le dossier de labels que si on est en entraînement
    if not is_inference:
        labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")
        os.makedirs(labelsTr_dir, exist_ok=True)

    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f"\n--- Début du formatage PET/CT : {len(subjects)} patients trouvés ---")

    valid_subjects = 0

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        imgs_dir = os.path.join(subj_path, "imgs")
        mask_dir = os.path.join(subj_path, "mask")

        if not os.path.exists(imgs_dir):
            continue
        if not is_inference and not os.path.exists(mask_dir):
            continue
            
        # 1. On cherche STRICTEMENT le fichier converti en SUV
        pet_files = glob.glob(os.path.join(imgs_dir, "*_SUV.nii.gz"))
        
        # 2. Sécurité : Si le script de conversion SUV n'a pas tourné ou a échoué
        if not pet_files:
            print(f" [ALERTE CRITIQUE] Aucun fichier SUV trouvé pour {subj} ! Utilisation du RAW par défaut.")
            pet_files = glob.glob(os.path.join(imgs_dir, "*_RAW.nii.gz"))
        ct_files = glob.glob(os.path.join(imgs_dir, "*TDM*.nii.gz"))
        
        # Le glob des masques devient conditionnel
        mask_files = [] if is_inference else glob.glob(os.path.join(mask_dir, "*.nii.gz"))

        # Vérification finale conditionnelle
        if not pet_files or not ct_files:
            print(f" [SKIP] {subj} : Images manquantes.")
            continue
        if not is_inference and not mask_files:
            print(f" [SKIP] {subj} : Masque manquant pour l'entraînement.")
            continue
            
        print(f" Traitement de : {subj}")

        pet_dst = os.path.join(target_images_dir, f"{subj}_0000.nii.gz")
        ct_dst = os.path.join(target_images_dir, f"{subj}_0001.nii.gz")
        
        if not is_inference:
            mask_dst = os.path.join(labelsTr_dir, f"{subj}.nii.gz")

        try:
            # 1. Le PET est copié tel quel (c'est notre ancrage dans le monde réel)
            shutil.copy(pet_files[0], pet_dst)

            # 2. Alignement strict du CT sur le PET (pad_value = -1000 HU pour l'air)
            enforce_strict_alignment(
                ref_path=pet_dst, 
                moving_path=ct_files[0], 
                out_path=ct_dst, 
                is_mask=False, 
                pad_value=-1000.0
            )

            # 3. Alignement du masque SEULEMENT si on n'est pas en inférence
            if not is_inference:
                enforce_strict_alignment(
                    ref_path=pet_dst, 
                    moving_path=mask_files[0], 
                    out_path=mask_dst, 
                    is_mask=True
                )

            valid_subjects += 1
            
        except Exception as e:
            print(f" [ERREUR] Impossible de traiter {subj} : {e}")
            if os.path.exists(pet_dst):
                os.remove(pet_dst)

    # Génération du dataset.json
    # On ne génère le dataset.json que si on prépare le Training Set !
    # Sinon on risque d'écraser la config d'entraînement avec les stats du test set.
    if not is_inference:
        dataset_json = {
            "channel_names": {"0": "PET", "1": "CT"},
            "labels": {"background": 0, "tumor": 1},
            "numTraining": valid_subjects,
            "file_ending": ".nii.gz"
        }

        with open(os.path.join(nnunet_raw, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=4)
            
    print("\n" + "="*50)
    print(f" FORMATAGE TERMINÉ ! Patients valides : {valid_subjects}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./Base_PETCT", help="Dossier source des patients")
    parser.add_argument("--nnunet", default="./nnunet_data", help="Racine nnU-Net")
    parser.add_argument("--inference", action="store_true", help="Prépare les données pour la prédiction (ignore les masques, va dans imagesTs)")
    args = parser.parse_args()
    
    prepare_pet_ct_for_nnunet(args.src, args.nnunet, is_inference=args.inference)

#!/usr/bin/env python3
"""
Orchestrateur IRM DCE vers nnU-Net V2.
Vérifie et force l'alignement de toutes les phases sur la phase 0 (T1 pré-contraste),
applique la normalisation globale, et aligne le masque (si entraînement).
"""

import os
import glob
import json
import argparse
from utils.normalize_mris_phases import normalize_dce_patient
from utils.spatial_standardizer import enforce_strict_alignment

def extract_dce_to_nnunet_flat(
    subjects_dir: str,
    nnunet_root: str,
    dataset_id: int = 1,
    num_channels: int = 4,
    channel_prefix: str = "DCE",
    is_inference: bool = False # NOUVEAU : Flag d'inférence
):
    dataset_name = f"Dataset{dataset_id:03d}_{channel_prefix}"
    nnunet_raw = os.path.join(nnunet_root, "nnUNet_raw", dataset_name)
    
    # MODIFICATION : imagesTs si inférence, sinon imagesTr
    target_images_dir = os.path.join(nnunet_raw, "imagesTs" if is_inference else "imagesTr")
    os.makedirs(target_images_dir, exist_ok=True)

    # On ne crée le dossier de labels que si on est en entraînement
    if not is_inference:
        labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")
        os.makedirs(labelsTr_dir, exist_ok=True)

    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f" {len(subjects)} patients trouvés dans le dossier source.\n")

    valid_subjects = 0

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        imgs_dir = os.path.join(subj_path, "imgs")
        mask_dir = os.path.join(subj_path, "mask")

        if not os.path.exists(imgs_dir):
            continue

        # MODIFICATION : Le masque n'est requis qu'en entraînement
        if not is_inference and not os.path.exists(mask_dir):
            continue

        fichiers_images = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))
        if len(fichiers_images) < num_channels:
            print(f" [SKIP] {subj}: Pas assez de canaux (trouvé {len(fichiers_images)}, attendu {num_channels})")
            continue

        # MODIFICATION : Le glob des masques devient conditionnel
        fichiers_masques = [] if is_inference else sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        if not is_inference and not fichiers_masques:
            continue

        print(f"[INFO] Traitement patient : {subj}")

        # --- ÉTAPE 1 : Alignement inter-phases ---
        # La phase 0 (généralement le T1 natif) est notre référence
        ref_phase_path = fichiers_images[0]
        aligned_phases_paths = [ref_phase_path] # Le premier ne bouge pas
        
        # On crée un dossier temporaire pour stocker les phases ré-échantillonnées si besoin
        tmp_dir = os.path.join(subj_path, "tmp_aligned")
        os.makedirs(tmp_dir, exist_ok=True)

        for i in range(1, num_channels):
            moving_phase = fichiers_images[i]
            tmp_out = os.path.join(tmp_dir, f"aligned_phase_{i}.nii.gz")
            
            # enforce_strict_alignment se charge de vérifier. Si c'est déjà aligné, il copie juste.
            enforce_strict_alignment(
                ref_path=ref_phase_path,
                moving_path=moving_phase,
                out_path=tmp_out,
                is_mask=False
            )
            aligned_phases_paths.append(tmp_out)

        # --- ÉTAPE 2 : Normalisation Globale (MAMA-MIA) ---
        chemins_sorties = [os.path.join(target_images_dir, f"{subj}_{idx:04d}.nii.gz") for idx in range(num_channels)]
        
        try:
            # On nourrit l'algorithme de normalisation avec nos phases garanties alignées
            normalize_dce_patient(aligned_phases_paths, chemins_sorties)
        except Exception as e:
            print(f"   [ERREUR] Normalisation échouée pour {subj} : {e}")
            continue

        # --- ÉTAPE 3 : Alignement du Masque (SEULEMENT EN ENTRAÎNEMENT) ---
        if not is_inference:
            dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
            enforce_strict_alignment(
                ref_path=ref_phase_path,
                moving_path=fichiers_masques[0],
                out_path=dst_mask,
                is_mask=True
            )

        valid_subjects += 1

    # MODIFICATION : On ne génère le dataset.json qu'en entraînement
    if not is_inference:
        channel_names = {str(i): f"{channel_prefix}_{i}" for i in range(num_channels)}
        dataset_json = {
            "channel_names": channel_names,
            "labels": {"background": 0, "lesion": 1},
            "numTraining": valid_subjects,
            "file_ending": ".nii.gz"
        }

        with open(os.path.join(nnunet_raw, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=4)

    print("\n" + "="*40)
    print(f" CONVERSION TERMINÉE ! Patients valides : {valid_subjects}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./Base_IRM", help="Dossier source des patients IRM")
    parser.add_argument("--nnunet", default="./nnunet_data", help="Racine nnU-Net")
    parser.add_argument("--inference", action="store_true", help="Prépare les données pour la prédiction (ignore masques, cible imagesTs)")
    args = parser.parse_args()
    
    extract_dce_to_nnunet_flat(args.src, args.nnunet, is_inference=args.inference)

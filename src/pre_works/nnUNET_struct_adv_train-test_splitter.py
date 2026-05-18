#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script de Séparation Train / Test pour nnU-Net V2
===============================================================================
Rôle :
  Déplace un pourcentage des données d'entraînement (imagesTr / labelsTr)
  vers un dossier de test (imagesTs / labelsTs) pour évaluation post-entraînement.

Fonctionnalités :
  - Détection automatique des patients via le dossier labelsTr.
  - Gestion des préférences : Tente d'inclure des patients spécifiques dans le 
    jeu de test (soit par leur ID exact, soit par leur indice numérique).
  - Complétion aléatoire : Si les préférences ne suffisent pas à atteindre le 
    ratio demandé, sélectionne aléatoirement le reste des patients.
  - Mise à jour automatique du fichier 'dataset.json' (champ numTraining).

Utilisation :
  python split_train_test.py --dataset_dir ./nnUNet_raw/Dataset002_BreastPETCT \
                             --ratio 0.15 \
                             --pref QIN-BREAST-01-0005 QIN-BREAST-01-0012 0 1
===============================================================================
"""

import os
import glob
import json
import random
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Sépare un dataset nnU-Net V2 en Train et Test avec gestion de préférences."
    )
    
    # --- Arguments ---
    parser.add_argument(
        "--dataset_dir", 
        type=Path, 
        required=True,
        help="Chemin vers la racine du dataset (ex: ./nnUNet_raw/Dataset002_BreastPETCT)"
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.20, 
        help="Ratio de données à déplacer vers le test (ex: 0.20 pour 20%%). (Défaut: 0.20)"
    )
    parser.add_argument(
        "--pref", 
        nargs="*", 
        default=[], 
        help="Liste des patients préférés pour le test. Peut être l'ID exact (ex: QIN_001) "
             "ou l'indice numérique dans la liste triée (ex: 0 4 12)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Graine aléatoire pour la reproductibilité du tirage au sort. (Défaut: 42)"
    )

    args = parser.parse_args()

    # 1. Vérification des chemins
    dataset_dir = args.dataset_dir
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTs = dataset_dir / "labelsTs"

    if not imagesTr.exists() or not labelsTr.exists():
        raise FileNotFoundError(f"Les dossiers imagesTr ou labelsTr sont introuvables dans {dataset_dir}")

    imagesTs.mkdir(parents=True, exist_ok=True)
    labelsTs.mkdir(parents=True, exist_ok=True)

    # 2. Identification du VRAI total (Train + Test existant)
    train_patients = sorted([f.name.replace(".nii.gz", "") for f in labelsTr.glob("*.nii.gz")])
    test_patients_existants = sorted([f.name.replace(".nii.gz", "") for f in labelsTs.glob("*.nii.gz")])
    
    all_patients_ever = train_patients + test_patients_existants
    total_patients = len(all_patients_ever)
    
    if total_patients == 0:
        print("Aucun patient trouvé dans le dataset.")
        return

    # 3. Calcul de la cible et SÉCURITÉ
    safe_ratio = args.ratio if args.ratio <= 1.0 else args.ratio / 100.0
    target_test_count = int(round(total_patients * safe_ratio))
    
    if safe_ratio > 0 and target_test_count == 0:
        target_test_count = 1

    print(f"\nAnalyse du dataset : {total_patients} patients au total (Train: {len(train_patients)} | Test: {len(test_patients_existants)}).")
    print(f"Objectif global pour le Test ({safe_ratio*100}%) : {target_test_count} patients.")

    # --- LE BOUCLIER DE SÉCURITÉ ---
    if len(test_patients_existants) >= target_test_count:
        print(f"\n🛡️ SÉCURITÉ ACTIVÉE : Le dossier de test contient déjà {len(test_patients_existants)} patients.")
        print("L'objectif est déjà atteint ou dépassé. Le script s'arrête pour éviter de vider l'entraînement.")
        return
        
    # S'il en manque, on calcule combien on doit encore en déplacer
    needed_count = target_test_count - len(test_patients_existants)
    print(f"Ajustement : Il manque {needed_count} patient(s) à déplacer pour atteindre le ratio.")

    # 4. Traitement des Préférences (On cherche uniquement parmi ceux qui sont encore dans Train)
    test_patients_a_deplacer = []
    
    for p in args.pref:
        patient_to_add = None
        if p.isdigit():
            idx = int(p)
            # Attention: l'indice s'applique sur la liste globale triée pour être constant
            if 0 <= idx < total_patients:
                patient_to_add = sorted(all_patients_ever)[idx]
            else:
                print(f"  [Avertissement] Indice préféré '{p}' hors limites. Ignoré.")
        else:
            patient_to_add = p
                
        # On vérifie s'il est valide, et SURTOUT s'il n'est pas DÉJÀ dans le test
        if patient_to_add in test_patients_existants:
            print(f"  [Info] La préférence {patient_to_add} est déjà dans le dossier de test.")
        elif patient_to_add in train_patients and patient_to_add not in test_patients_a_deplacer:
            if len(test_patients_a_deplacer) < needed_count:
                test_patients_a_deplacer.append(patient_to_add)
                print(f"Préférence honorée : {patient_to_add} sélectionné pour le déplacement.")
            else:
                print(f"  [Avertissement] Quota manquant atteint, préférence {patient_to_add} ignorée.")

    # 5. Complétion aléatoire si le quota manquant n'est pas atteint
    remaining_in_train = [p for p in train_patients if p not in test_patients_a_deplacer]
    still_needed = needed_count - len(test_patients_a_deplacer)
    
    if still_needed > 0:
        random.seed(args.seed)
        random_selection = random.sample(remaining_in_train, still_needed)
        test_patients_a_deplacer.extend(random_selection)
        print(f"  Complétion aléatoire : {still_needed} patient(s) supplémentaire(s) sélectionné(s).")

    test_patients_a_deplacer.sort()
    
    # 6. DÉPLACEMENT PHYSIQUE DES FICHIERS
    print("\nDéplacement des fichiers en cours...")
    
    moved_count = 0
    for patient_id in test_patients:
        # 6.1 Déplacement du Label (La Vérité Terrain)
        lbl_src = labelsTr / f"{patient_id}.nii.gz"
        lbl_dst = labelsTs / f"{patient_id}.nii.gz"
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(lbl_dst))
        
        # 6.2 Déplacement des Images (Toutes les modalités associées)
        # On utilise glob pour attraper dynamiquement ID_0000.nii.gz, ID_0001.nii.gz...
        patient_images = imagesTr.glob(f"{patient_id}_*.nii.gz")
        for img_src in patient_images:
            img_dst = imagesTs / img_src.name
            shutil.move(str(img_src), str(img_dst))
            
        moved_count += 1

    # 7. Mise à jour de dataset.json (Crucial pour nnU-Net)
    json_path = dataset_dir / "dataset.json"
    new_train_count = total_patients - moved_count
    
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)
            
        old_count = dataset_meta.get("numTraining", "Inconnu")
        dataset_meta["numTraining"] = new_train_count
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset_meta, f, indent=4)
            
        print(f"\ndataset.json mis à jour : numTraining passe de {old_count} à {new_train_count}.")
    else:
        print(f"\nAttention : dataset.json introuvable à la racine {dataset_dir}.")

    # 8. Bilan Final
    print("\n" + "="*50)
    print("                BILAN DU SPLIT              ")
    print("="*50)
    print(f"  Patients en Entraînement (Train) : {new_train_count}")
    print(f"  Patients en Test (Test)          : {moved_count}")
    print("="*50)
    print("  Liste des patients isolés pour le test :")
    for p in test_patients:
        print(f"    - {p}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

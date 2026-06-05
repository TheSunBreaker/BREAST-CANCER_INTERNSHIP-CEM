#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script de Segmentation des Seins via TotalSegmentator (Baseline Strict) - V3
===============================================================================
Rôle :
  Parcourt la base de données PET/CT générée par l'Ingesteur V6, cible 
  UNIQUEMENT les examens de Baseline, et utilise TotalSegmentator (modèle breasts)
  pour générer un masque binaire GLOBAL des seins, mais aussi des zones à éviter qui sont autre chose que des seins.

Nouveautés V3 :
  - Support Colab natif (API et CLI).
  - Mode hybride (API/CLI) restauré avec le sous-modèle 'breasts'.
  - Fusion intelligente : Combine le sein gauche et droit en un seul NIfTI.
  - Extrait toutes les segmentations de corps afin de pouvoir construire la zone non sein
  - Cohérence Spatiale : Utilise STRICTEMENT SimpleITK.
===============================================================================

  ATTENTION, TOTALSEGMENTATOR V2 MINIMUM NECESSAIRE POUR SEGMENTER LE CARTILAGE INTERCOSTAL
"""

import argparse
import subprocess
from pathlib import Path
import shutil
import SimpleITK as sitk
import numpy as np

# Tentative d'importation de l'API Python de TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False



def run_totalseg_api(ct_file: Path, output_root: Path, patient_id: str, fast: bool, tmp_dir: Path):
    """Lance TotalSegmentator via l'API Python native vers le dossier temporaire."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
  
    totalsegmentator(
        input=str(ct_file),
        output=str(tmp_dir),
        task="breasts",
        fast=fast,
        ml=False
    )
    
    # Déplacement direct des masques individuels au lieu de les fusionner
    for mask_file in tmp_dir.glob("*.nii.gz"):
        output_path = output_root / f"{patient_id}_{mask_file.name}"
        shutil.move(str(mask_file), str(output_path))


def run_totalseg_cli(ct_file: Path, output_root: Path, device: str, patient_id: str, fast: bool, tmp_dir: Path):
    """Lance TotalSegmentator via ligne de commande vers le dossier temporaire."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(ct_file),
        "-o", str(tmp_dir),
        "-ta", "breasts",
        "--device", device
    ]

    if fast:
        cmd.insert(4, "--fast")

    subprocess.run(cmd, check=True,
                   #stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)
    
    # Déplacement direct des masques individuels au lieu de les fusionner
    for mask_file in tmp_dir.glob("*.nii.gz"):
        output_path = output_root / f"{patient_id}_{mask_file.name}"
        shutil.move(str(mask_file), str(output_path))

def extract_shield_organs(ct_file: Path, patient_organs_dir: Path, mode: str, device: str, muscles_seg: bool = False):
    """Lance TS (tâche totale) en mode FAST, et ne conserve que le bouclier."""
    tmp_total_dir = patient_organs_dir / "tmp_total"
    if tmp_total_dir.exists():
        shutil.rmtree(tmp_total_dir)
    tmp_total_dir.mkdir(parents=True, exist_ok=True)

    print(f"    -> Extraction des boucliers (cœur, sternum, poumon, côtes, vertèbres, clavicules, cartilage inter-costal, muscles pectoraux, ..., etc.) en mode FAST...")
    
    # On force le mode fast pour ne pas alourdir les calculs
    if mode == "api":
        # La tâche 'total' contient la majorité de nos organes boucliers voulus. Mais pour avoir les pectoraux, il nous faut lancer la tâche 'abdominal_muscles'
        print("-------------> Segmentation totale : Début...")
        totalsegmentator(input=str(ct_file), output=str(tmp_total_dir), task="total", fast=True, ml=False)
        print("-------------> Segmentation totale terminée.")
        # Si on veut la segmentation musculaire aussi. 
        # ATTENTION : La tâche musculaire ne supporte pas le mode fast
        if muscles_seg : 
          print("-------------> Segmentation muscles : Début...")
          totalsegmentator(input=str(ct_file), output=str(tmp_total_dir), task="abdominal_muscles", fast=False, ml=False)
          print("-------------> Segmentation muscles terminée.")
    else:
        cmd = ["TotalSegmentator", "-i", str(ct_file), "-o", str(tmp_total_dir), "-ta", "total", "--fast", "--device", device]
        print("-------------> Segmentation totale : Début...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("-------------> Segmentation totale terminée.")
        # Si on veut la segmentation musculaire aussi. 
        if muscles_seg :
          # ATTENTION : La tâche musculaire ne supporte pas le mode fast
          print("-------------> Segmentation muscles : Début...")
          cmd = ["TotalSegmentator", "-i", str(ct_file), "-o", str(tmp_total_dir), "-ta", "abdominal_muscles", "--device", device]
          subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
          print("-------------> Segmentation muscles terminée.")
            

    # Filtrage : On ne déplace QUE les organes qui nous intéressent vers le dossier final du patient
    fichiers_a_garder = [
        "heart.nii.gz", 
        "sternum.nii.gz", 
        "costal_cartilages.nii.gz",
        "clavicula_left.nii.gz",
        "clavicula_right.nii.gz",
        "pectoralis_major_left.nii.gz",
        "pectoralis_major_right.nii.gz"
    ]
    
    for fichier in fichiers_a_garder:
        src = tmp_total_dir / fichier
        if src.exists():
            shutil.move(str(src), str(patient_organs_dir / fichier))
            
    # On récupère aussi toutes les côtes (TotalSegmentator les sépare par défaut)
    for rib_file in tmp_total_dir.glob("*rib*.nii.gz"):
        shutil.move(str(rib_file), str(patient_organs_dir / rib_file.name))

    # On récupère également les poumons 
    for lung_file in tmp_total_dir.glob("*lung*.nii.gz"):
        shutil.move(str(lung_file), str(patient_organs_dir / lung_file.name))

    # Egalement les verèbres
    for vert_file in tmp_total_dir.glob("*vertebrae*.nii.gz"):
        shutil.move(str(vert_file), str(patient_organs_dir / vert_file.name))
  

    # On supprime les autres organes générés pour libérer l'espace disque
    shutil.rmtree(tmp_total_dir, ignore_errors=True)
    print(f"    -> Boucliers sauvegardés dans {patient_organs_dir.name}/")
# --- FIN DE L'AJOUT ZONE 2 ---


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation automatique des seins à partir des CT de Baseline."
    )

    parser.add_argument(
        "--input_root", 
        type=Path, 
        default=Path("./Base_PETCT"), 
        help="Dossier racine des données (Défaut: ./Base_PETCT)"
    )
    parser.add_argument(
        "--output_root", 
        type=Path, 
        default=Path("./Base_PETCT_BreastMasks"), 
        help="Dossier de sauvegarde des masques (Défaut: ./Base_PETCT_BreastMasks)"
    )

    parser.add_argument(
        "--output_organs_root", 
        type=Path, 
        default=Path("./Base_PETCT_Organs"), 
        help="Dossier de sauvegarde pour les boucliers (cœur, sternum, côtes)."
    )

    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default="cli", 
        help="Mode d'exécution : 'api' (librairie Python) ou 'cli' (ligne de commande terminal). Défaut: cli"
    )

    parser.add_argument("--device", default="gpu", help="Matériel: gpu, gpu:0... ou cpu (Défaut: gpu)")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (Recommandé si CPU)")
    parser.add_argument("--muscles_seg", action="store_true", help="Récupérer la segmentation des muscles également (attention, tâche lourde)")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les masques déjà existants")
    parser.add_argument("--ct-suffix", default="_TDM_", help="Marqueur du fichier CT (Défaut: _TDM_)")

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_organs_root.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "api" and not API_AVAILABLE:
        print("API TotalSegmentator non détectée.")
        print("   -> Bascule automatique en mode Ligne de Commande (CLI).")
        args.mode = "cli"

    patients = [p for p in args.input_root.iterdir() if p.is_dir()]
    
    if not patients:
        print(f"Aucun dossier patient trouvé dans {args.input_root}")
        return

    print(f"{len(patients)} dossier(s) patient analysé(s) pour la Baseline (Mode {args.mode.upper()}).\n")

    for patient_dir in patients:
        patient_id = patient_dir.name
        
        imgs_dir = patient_dir / "imgs"
        if not imgs_dir.exists():
            continue

        ct_files = list(imgs_dir.glob(f"*{args.ct_suffix}*.nii.gz"))
        if not ct_files:
            continue
            
        ct_file = ct_files[0]
        output_mask = args.output_root / f"{patient_id}_breast_mask.nii.gz"

        print(f"\n[{patient_id}] (Source: {ct_file.name}) -> {args.device.upper()}")

        # =========================================================
        # TÂCHE 1 : MASQUE MAMMAIRE (BASELINE)
        # =========================================================
        if not list(args.output_root.glob(f"{patient_id}_*breast*.nii.gz")) or args.overwrite:

            print(f" [RUN ] {patient_id} Segmentation mammaire en cours...")

            try:
                tmp_dir_breast = args.output_root / f"{patient_id}_tmp_breast"
                
                if args.mode == "api":
                    run_totalseg_api(ct_file, args.output_root, patient_id, args.fast, tmp_dir_breast)
                else:
                    run_totalseg_cli(ct_file, args.output_root, args.device, patient_id, args.fast, tmp_dir_breast)
                
                shutil.rmtree(tmp_dir_breast, ignore_errors=True)
                print(f"    -> [SUCCÈS] {patient_id} Masques mammaires individuels sauvegardés.")

            except subprocess.CalledProcessError as e:
                print(f"    -> [ÉCHEC] {patient_id} Crash de TotalSegmentator (Seins) : {e}.")
            except Exception as e:
                print(f"    -> [ÉCHEC] {patient_id} Erreur inattendue (Seins) : {e}")

        else:          
            print(f"[SKIP  ] {patient_id} : Masque mammaire déjà existant.")
        

        # =========================================================
        # TÂCHE 2 : BOUCLIERS ANATOMIQUES (CŒUR, STERNUM, CÔTES, etc.)
        # =========================================================
        patient_organs_dir = args.output_organs_root / patient_id
        patient_organs_dir.mkdir(parents=True, exist_ok=True)

        # On vérifie si les pièces maîtresses du bouclier sont déjà là
        # (Si le coeur et le sternum y sont, on considère que le dossier est complet)
        bouclier_complet = (patient_organs_dir / "heart.nii.gz").exists() and \
                           (patient_organs_dir / "sternum.nii.gz").exists() and \
                           (patient_organs_dir / "costal_cartilages.nii.gz").exists()

        # --- NOUVEAU : On exige les pectoraux si l'argument est actif ---
        if args.muscles_seg:
            bouclier_complet = bouclier_complet and (patient_organs_dir / "pectoralis_major_left.nii.gz").exists()
      
        if not bouclier_complet or args.overwrite:
            try:
                extract_shield_organs(ct_file, patient_organs_dir, args.mode, args.device, args.muscles_seg)
            except Exception as e:
                print(f"    -> [ÉCHEC] {patient_id} Erreur lors de l'extraction des boucliers : {e}")
        else:
            print(f"  [SKIP] {patient_id} Boucliers anatomiques déjà existants.")

    print("\n=== SEGMENTATION AUTOMATIQUE DES SEINS TERMINÉE ===")


if __name__ == "__main__":
    main()

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


def merge_breast_masks_sitk(tmp_dir: Path, output_mask: Path):
    """
    Lit tous les masques générés par TotalSegmentator (gauche/droit ou unifié),
    les binarise (valeurs > 0), les fusionne via un OU logique, 
    et sauvegarde le résultat avec SimpleITK.
    """
    breast_files = list(tmp_dir.glob("*breast*.nii.gz"))
    
    # Si le modèle n'a pas mis 'breast' dans le nom, on prend tout ce qui est .nii.gz 
    # (car le dossier temp est censé être exclusif à cette tâche)
    if not breast_files:
        breast_files = list(tmp_dir.glob("*.nii.gz"))

    if not breast_files:
        raise RuntimeError("Aucun masque généré par TotalSegmentator. (FOV trop petit ou erreur modèle).")

    merged_np = None
    ref_img = None

    for f in breast_files:
        img = sitk.ReadImage(str(f))
        # On binarise : tout ce qui est > 0 devient True (couvre le cas où TS sort des valeurs 1 et 2)
        arr = sitk.GetArrayFromImage(img) > 0
        
        if merged_np is None:
            merged_np = arr
            ref_img = img
        else:
            merged_np = np.logical_or(merged_np, arr)

    merged_img = sitk.GetImageFromArray(merged_np.astype(np.uint8))
    merged_img.CopyInformation(ref_img)

    sitk.WriteImage(merged_img, str(output_mask))
    print(f"    -> Fusion de {len(breast_files)} structure(s) réussie.")


def run_totalseg_api(ct_file: Path, output_mask: Path, fast: bool, tmp_dir: Path):
    """Lance TotalSegmentator via l'API Python native vers le dossier temporaire."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    totalsegmentator(
        input=str(ct_file),
        output=str(tmp_dir),
        task="breasts",
        fast=fast,
        ml=True
    )
    
    merge_breast_masks_sitk(tmp_dir, output_mask)


def run_totalseg_cli(ct_file: Path, output_mask: Path, device: str, fast: bool, tmp_dir: Path):
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
    
    merge_breast_masks_sitk(tmp_dir, output_mask)

def extract_shield_organs(ct_file: Path, patient_organs_dir: Path, mode: str, device: str):
    """Lance TS (tâche totale) en mode FAST, et ne conserve que le bouclier."""
    tmp_total_dir = patient_organs_dir / "tmp_total"
    if tmp_total_dir.exists():
        shutil.rmtree(tmp_total_dir)
    tmp_total_dir.mkdir(parents=True, exist_ok=True)

    print(f"    -> Extraction des boucliers (cœur, sternum, côtes) en mode FAST...")
    
    # On force le mode fast pour ne pas alourdir les calculs
    if mode == "api":
        totalsegmentator(input=str(ct_file), output=str(tmp_total_dir), task="total", fast=True, ml=True)
    else:
        cmd = ["TotalSegmentator", "-i", str(ct_file), "-o", str(tmp_total_dir), "-ta", "total", "--fast", "--device", device]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Filtrage : On ne déplace QUE les organes qui nous intéressent vers le dossier final du patient
    fichiers_a_garder = [
        "heart.nii.gz", 
        "sternum.nii.gz", 
        "costal_cartilages.nii.gz"  # <-- L'ajout indispensable ici
    ]
    
    for fichier in fichiers_a_garder:
        src = tmp_total_dir / fichier
        if src.exists():
            shutil.move(str(src), str(patient_organs_dir / fichier))
            
    # On récupère aussi toutes les côtes (TotalSegmentator les sépare par défaut)
    for rib_file in tmp_total_dir.glob("*rib*.nii.gz"):
        shutil.move(str(rib_file), str(patient_organs_dir / rib_file.name))

    # On supprime les 100 autres organes générés pour libérer l'espace disque
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
        if not output_mask.exists() or args.overwrite:

            print(f" [RUN ] {patient_id} Segmentation mammaire en cours...")

            try:
                tmp_dir_breast = args.output_root / f"{patient_id}_tmp_breast"
                
                if args.mode == "api":
                    run_totalseg_api(ct_file, output_mask, args.fast, tmp_dir_breast)
                else:
                    run_totalseg_cli(ct_file, output_mask, args.device, args.fast, tmp_dir_breast)
                
                shutil.rmtree(tmp_dir_breast, ignore_errors=True)
                print(f"    -> [SUCCÈS] {patient_id} Masque sauvegardé : {output_mask.name}")

            except subprocess.CalledProcessError as e:
                print(f"    -> [ÉCHEC] {patient_id} Crash de TotalSegmentator (Seins) : {e}.")
            except Exception as e:
                print(f"    -> [ÉCHEC] {patient_id} Erreur inattendue (Seins) : {e}")

        else:          
            print(f"[SKIP  ] {patient_id} : Masque mammaire déjà existant.")
        

        # =========================================================
        # TÂCHE 2 : BOUCLIERS ANATOMIQUES (CŒUR, STERNUM, CÔTES)
        # =========================================================
        patient_organs_dir = args.output_organs_root / patient_id
        patient_organs_dir.mkdir(parents=True, exist_ok=True)

        # On vérifie si les pièces maîtresses du bouclier sont déjà là
        # (Si le coeur et le sternum y sont, on considère que le dossier est complet)
        bouclier_complet = (patient_organs_dir / "heart.nii.gz").exists() and \
                           (patient_organs_dir / "sternum.nii.gz").exists() and \
                           (patient_organs_dir / "costal_cartilages.nii.gz").exists() # <-- Ajout ici
      
        if not bouclier_complet or args.overwrite:
            try:
                extract_shield_organs(ct_file, patient_organs_dir, args.mode, args.device)
            except Exception as e:
                print(f"    -> [ÉCHEC] {patient_id} Erreur lors de l'extraction des boucliers : {e}")
        else:
            print(f"  [SKIP] {patient_id} Boucliers anatomiques déjà existants.")

    print("\n=== SEGMENTATION AUTOMATIQUE DES SEINS TERMINÉE ===")


if __name__ == "__main__":
    main()

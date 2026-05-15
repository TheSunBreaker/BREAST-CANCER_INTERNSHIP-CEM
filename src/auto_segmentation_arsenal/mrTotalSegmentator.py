#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script de Segmentation des Seins via TotalSegmentator (Baseline Strict)
===============================================================================
Rôle :
  Parcourt la base de données PET/CT générée par l'Ingesteur V6, cible 
  UNIQUEMENT les examens de Baseline (dossier 'imgs' sans suffixe temporel),
  et utilise TotalSegmentator pour générer un masque binaire des seins 
  à partir du scanner CT.

Prérequis Absolus :
  - Installation locale obligatoire : `pip install TotalSegmentator`
  - La première exécution nécessitera internet pour télécharger les modèles IA.
  - Matériel : GPU fortement recommandé (arg: --device gpu:0), mais CPU 
    possible (arg: --device cpu) au prix d'un temps de calcul bien plus long.

Structure Attendue (Par défaut : ./Base_PETCT) :
  Base_PETCT/
    ├── DUKE_001/
    │   ├── imgs/                   <-- CIBLE STRICTE (Baseline)
    │   │   ├── DUKE_001_TEP_Baseline_A1B2C_RAW.nii.gz
    │   │   └── DUKE_001_TDM_A1B2C.nii.gz   <-- Fichier CT lu par ce script
    │   ├── imgs_20230514_1430/     <-- IGNORÉ (Suivi longitudinal)
    │   └── TEP/                    <-- IGNORÉ
    └── ...

Structure Produite (Par défaut : ./Base_PETCT_BreastMasks) :
  Base_PETCT_BreastMasks/
    ├── DUKE_001_breast_mask.nii.gz
    ├── DUKE_002_breast_mask.nii.gz
    └── ...
===============================================================================
"""

import argparse
import subprocess
from pathlib import Path
import shutil
import sys

# Tentative d'importation de l'API Python de TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


def run_totalseg_api(ct_file: Path, output_mask: Path):
    """Lance TotalSegmentator via l'API Python native (librairie locale)."""
    totalsegmentator(
        input=str(ct_file),
        output=str(output_mask),
        task="total", 
        roi_subset=["breast_female_left", "breast_female_right"], 
        ml=True 
    )


def run_totalseg_cli(ct_file: Path, output_mask: Path, device: str, fast: bool, tmp_dir: Path):
    """
    Lance TotalSegmentator via ligne de commande (Terminal local).
    Utilise un dossier temporaire pour isoler les fichiers produits.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(ct_file),
        "-o", str(tmp_dir),
        "-ta", "breasts", # Sous-modèle spécifique pour les seins
        "--device", device,
        "--statistics",
    ]

    # Mode "Fast" : utilise une résolution plus basse (3mm). Très utile sur CPU !
    if fast:
        cmd.insert(4, "--fast")

    # Lancement du processus
    subprocess.run(cmd, check=True)

    # Recherche du masque généré dans le dossier temporaire
    breast_files = list(tmp_dir.glob("*breast*.nii.gz"))

    if not breast_files:
        raise RuntimeError("Aucun fichier 'breast' trouvé en sortie de la CLI TotalSegmentator.")

    # Déplacement du fichier final vers sa destination officielle
    shutil.move(str(breast_files[0]), output_mask)


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation automatique des seins à partir des CT de Baseline."
    )

    # --- ARGUMENTS PRINCIPAUX AVEC VALEURS PAR DÉFAUT ---
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

    # --- MODE D'EXÉCUTION (Ligne de commande privilégiée par défaut) ---
    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default="cli", 
        help="Mode d'exécution : 'api' (librairie Python) ou 'cli' (ligne de commande terminal). Défaut: cli"
    )

    # --- OPTIONS MATÉRIELLES ET DE FLUX ---
    parser.add_argument("--device", default="gpu", help="Matériel: gpu:0, gpu:1... ou cpu (Défaut: gpu:0)")
    parser.add_argument("--fast", action="store_true", help="Mode rapide basse résolution (Très recommandé si CPU)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Passe les patients déjà segmentés (Activé par défaut)")
    
    # Suffixe pour identifier le CT dans le dossier imgs (issu de l'ingesteur V6)
    parser.add_argument("--ct-suffix", default="_TDM_", help="Marqueur du fichier CT (Défaut: _TDM_)")

    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    
    output_root.mkdir(parents=True, exist_ok=True)

    # Vérification des dépendances si mode API demandé
    if args.mode == "api" and not API_AVAILABLE:
        print("❌ API TotalSegmentator non détectée (Avez-vous fait 'pip install' ?).")
        print("   -> Bascule automatique en mode Ligne de Commande (CLI).")
        args.mode = "cli"

    # Récupération des dossiers patients
    patients = [p for p in input_root.iterdir() if p.is_dir()]
    
    if not patients:
        print(f"❌ Aucun dossier patient trouvé dans {input_root}")
        return

    print(f"🔍 {len(patients)} dossier(s) patient analysé(s) pour la Baseline (Mode {args.mode.upper()}).\n")

    for patient_dir in patients:
        patient_id = patient_dir.name
        
        # ---------------------------------------------------------------------
        # RÈGLE D'OR : Cible strictement la Baseline (dossier "imgs")
        # ---------------------------------------------------------------------
        imgs_dir = patient_dir / "imgs"
        
        if not imgs_dir.exists():
            print(f"[IGNORE] {patient_id} : Aucun dossier 'imgs' (Baseline) trouvé.")
            continue

        ct_files = list(imgs_dir.glob(f"*{args.ct_suffix}*.nii.gz"))
        
        if not ct_files:
            print(f"[IGNORE] {patient_id} : Aucun CT ({args.ct_suffix}) trouvé dans la Baseline.")
            continue
            
        # Règle demandée : on prend strictement le premier CT trouvé
        ct_file = ct_files[0]
        
        output_mask = output_root / f"{patient_id}_breast_mask.nii.gz"

        if args.skip_existing and output_mask.exists():
            print(f"[SKIP  ] {patient_id} : Masque déjà existant.")
            continue

        print(f"[RUN   ] {patient_id} (Source: {ct_file.name}) -> {args.device.upper()}")

        # ---------------------------------------------------------------------
        # EXÉCUTION
        # ---------------------------------------------------------------------
        try:
            if args.mode == "api":
                run_totalseg_api(ct_file, output_mask)

            elif args.mode == "cli":
                tmp_dir = output_root / f"{patient_id}_tmp"
                run_totalseg_cli(ct_file, output_mask, args.device, args.fast, tmp_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)

            print(f"[SUCCÈS] Masque sauvegardé : {output_mask.name}")

        except Exception as e:
            print(f"[ÉCHEC ] {patient_id} : Erreur lors de la segmentation : {e}")

    print("\n=== SEGMENTATION AUTOMATIQUE TERMINÉE ===")

if __name__ == "__main__":
    main()

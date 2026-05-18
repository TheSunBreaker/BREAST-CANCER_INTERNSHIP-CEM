#!/usr/bin/env python3
r"""
Orchestrateur IRM DCE vers nnU-Net V2 - V2 (Sélecteur Physiologique Multicentrique).

=============================================================================
NOUVEAUTÉS V2 :
- Time-Matching Absolu : Sélectionne les phases NIfTI non pas par leur index (0,1,2,3),
  mais par leur TEMPS BIOLOGIQUE absolu post-injection (ex: 0s, 90s, 180s, 360s).
- Modularité des Canaux : Supporte de 1 à 4 canaux via le flag `--num_channels`.
- Modes Datasets : 
    -> INGESTEUR : Lit le 'DCE_temporal_log.txt' généré par notre Ingesteur V6.
    -> MAMAMIA   : Parse le fichier CSV officiel de MAMA-MIA pour lire 'acquisition_times'.
    -> BLIND     : Mode survie si aucune métadonnée n'est disponible (heuristique).
=============================================================================

Pré-requis :
Les NIfTI doivent être triés par ordre chronologique alphabétique dans le dossier 'imgs/'.
"""

import os
import glob
import json
import argparse
import csv
import ast
import numpy as np

from utils.normalize_mris_phases import normalize_dce_patient
from utils.spatial_standardizer import enforce_strict_alignment

# ============================================================================
# MODULE DE SÉLECTION TEMPORELLE (TIME-MATCHING)
# ============================================================================

def get_target_times(num_channels: int) -> list:
    """
    Définit les cibles physiologiques idéales en secondes post-début d'acquisition.
    Ces temps sont standards pour la cinétique du cancer du sein (Wash-in, Plateau, Wash-out).
    """
    targets = [0.0] # Canal 0 est TOUJOURS la Baseline (T0 / Pré-contraste)
    if num_channels >= 2:
        targets.append(90.0)  # Canal 1 : Pic précoce / Wash-in (Arteriel)
    if num_channels >= 3:
        targets.append(180.0) # Canal 2 : Plateau / Début de lavage
    if num_channels >= 4:
        targets.append(360.0) # Canal 3 : Wash-out tardif
    return targets

def parse_ingesteur_log(log_path: str) -> list:
    """
    Reconstruit la frise chronologique (timeline) en lisant le fichier .txt de l'Ingesteur V6.
    """
    timeline = [0.0]
    if not os.path.exists(log_path):
        return []
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                # On cherche les lignes du type : " - Ecart 0->1 : 17.0 s"
                if ligne.startswith("- Ecart"):
                    parts = ligne.split(":")
                    if len(parts) == 2:
                        ecart_sec = float(parts[1].replace("s", "").strip())
                        # Le temps actuel est le temps précédent + l'écart
                        timeline.append(timeline[-1] + ecart_sec)
        return timeline
    except Exception as e:
        print(f"    [ATTENTION] Échec de la lecture du log Ingesteur : {e}")
        return []

def load_mamamia_database(csv_path: str) -> dict:
    """
    Parse le fichier CSV de MAMA-MIA pour extraire les listes de temps.
    Retourne un dict: {'QIN-BREAST-01-0014': [0.0, 165.0, 288.0, 411.0], ...}
    """
    db = {}
    if not csv_path or not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier CSV MAMA-MIA introuvable : {csv_path}")
        return db
        
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id", "").strip()
                times_str = row.get("acquisition_times", "").strip()
                
                if pid and times_str:
                    try:
                        # Convertit la string "[0, 165, ...]" en vraie liste Python
                        time_list = ast.literal_eval(times_str)
                        db[pid] = [float(t) for t in time_list]
                    except Exception:
                        pass
    except Exception as e:
        print(f"[ERREUR] Échec du parsing du CSV MAMA-MIA : {e}")
        
    print(f" -> Base MAMA-MIA chargée en mémoire ({len(db)} patients avec timeline).")
    return db

def select_best_dce_phases(timeline: list, num_channels: int, nb_files_available: int) -> list:
    """
    Le cœur de l'intelligence artificielle du routage.
    Fait correspondre les images disponibles aux temps cibles idéaux.
    """
    # Si la timeline est vide ou cassée, on bascule en mode BLIND heuristique
    if not timeline or len(timeline) == 0:
        print("    [MODE BLIND ACTIVÉ] Aucune métadonnée temporelle. Sélection à l'aveugle.")
        # On prend le premier, le dernier, et on répartit au milieu
        if nb_files_available <= num_channels:
            return list(range(nb_files_available)) + [nb_files_available-1] * (num_channels - nb_files_available)
        else:
            return [int(i * (nb_files_available - 1) / (num_channels - 1)) for i in range(num_channels)]

    targets = get_target_times(num_channels)
    selected_indices = []
    
    # Sécurité : On limite la recherche aux index physiquement présents sur le disque
    max_idx = min(len(timeline), nb_files_available)
    valid_timeline = np.array(timeline[:max_idx])
    
    for t_cible in targets:
        # np.argmin trouve l'index où la différence absolue est minimale
        # Ex: si cible = 90s, et valid_timeline = [0, 85, 170], il choisira l'index 1 (85s)
        best_idx = np.argmin(np.abs(valid_timeline - t_cible))
        selected_indices.append(int(best_idx))
        
    return selected_indices

# ============================================================================
# L'ORCHESTRATEUR PRINCIPAL (INTACT ET AMÉLIORÉ)
# ============================================================================

def extract_dce_to_nnunet_flat(
    subjects_dir: str,
    nnunet_root: str,
    dataset_id: int = 1,
    num_channels: int = 4,
    channel_prefix: str = "DCE",
    is_inference: bool = False,
    mode: str = "INGESTEUR",
    csv_path: str = None
):
    dataset_name = f"Dataset{dataset_id:03d}_{channel_prefix}"
    nnunet_raw = os.path.join(nnunet_root, "nnUNet_raw", dataset_name)
    
    target_images_dir = os.path.join(nnunet_raw, "imagesTs" if is_inference else "imagesTr")
    os.makedirs(target_images_dir, exist_ok=True)

    if not is_inference:
        labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")
        os.makedirs(labelsTr_dir, exist_ok=True)

    # Chargement préalable de la base MAMA-MIA si demandé
    mamamia_db = {}
    if mode == "MAMAMIA":
        mamamia_db = load_mamamia_database(csv_path)

    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f"\n==================================================")
    print(f" DÉMARRAGE PIPELINE (Mode: {mode} | Canaux visés: {num_channels})")
    print(f" {len(subjects)} patients trouvés dans le dossier source.")
    print(f" Temps cibles physiologiques : {get_target_times(num_channels)} secondes")
    print(f"==================================================\n")

    valid_subjects = 0

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        imgs_dir = os.path.join(subj_path, "imgs") # On pointe toujours vers la Baseline par défaut !
        mask_dir = os.path.join(subj_path, "mask")

        if not os.path.exists(imgs_dir):
            continue

        if not is_inference and not os.path.exists(mask_dir):
            continue

        fichiers_images = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))
        nb_fichiers_dispos = len(fichiers_images)
        
        if nb_fichiers_dispos == 0:
            continue

        # --- NOUVEAUTÉ : GÉNÉRATION DE LA TIMELINE ---
        print(f"[INFO] Traitement patient : {subj} (Fichiers trouvés: {nb_fichiers_dispos})")
        timeline = []
        
        if mode == "INGESTEUR":
            log_path = os.path.join(imgs_dir, "DCE_temporal_log.txt")
            timeline = parse_ingesteur_log(log_path)
            
        elif mode == "MAMAMIA":
            # On essaie de faire matcher le nom de dossier avec le patient_id du CSV
            timeline = mamamia_db.get(subj, [])
            if not timeline:
                print(f"    [ALERTE] Patient {subj} absent du CSV MAMA-MIA. Bascule en BLIND.")

        # --- SÉLECTION PHYSIOLOGIQUE DES PHASES ---
        selected_indices = select_best_dce_phases(timeline, num_channels, nb_fichiers_dispos)
        
        # Log visuel pour vérifier le comportement de l'IA de sélection
        temps_reels = [f"{timeline[i]}s" if i < len(timeline) else "N/A" for i in selected_indices]
        print(f"    -> Phases retenues (Index) : {selected_indices}")
        print(f"    -> Temps réels correspondants : {temps_reels}")
        
        # On extrait physiquement les chemins des fichiers choisis
        phases_selectionnees_paths = [fichiers_images[idx] for idx in selected_indices]

        # --- ÉTAPE 1 : Alignement inter-phases ---
        ref_phase_path = phases_selectionnees_paths[0] # La T0 est toujours la référence absolue
        aligned_phases_paths = [ref_phase_path] 
        
        tmp_dir = os.path.join(subj_path, "tmp_aligned")
        os.makedirs(tmp_dir, exist_ok=True)

        for i in range(1, num_channels):
            moving_phase = phases_selectionnees_paths[i]
            tmp_out = os.path.join(tmp_dir, f"aligned_phase_{i}.nii.gz")
            
            # Garantit que la grille de l'image Wash-out correspond exactement à la Baseline
            enforce_strict_alignment(
                ref_path=ref_phase_path,
                moving_path=moving_phase,
                out_path=tmp_out,
                is_mask=False
            )
            aligned_phases_paths.append(tmp_out)

        # --- ÉTAPE 2 : Normalisation Globale (Z-score MAMA-MIA) ---
        chemins_sorties = [os.path.join(target_images_dir, f"{subj}_{idx:04d}.nii.gz") for idx in range(num_channels)]
        
        try:
            normalize_dce_patient(aligned_phases_paths, chemins_sorties)
        except Exception as e:
            print(f"   [ERREUR] Normalisation échouée pour {subj} : {e}")
            continue

        # --- ÉTAPE 3 : Alignement du Masque (SEULEMENT EN ENTRAÎNEMENT) ---
        if not is_inference:
            fichiers_masques = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
            if not fichiers_masques:
                print(f"   [ERREUR] Masque introuvable au moment de l'alignement pour {subj}.")
                continue
                
            dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
            enforce_strict_alignment(
                ref_path=ref_phase_path,
                moving_path=fichiers_masques[0], # Le masque est aligné sur le T0
                out_path=dst_mask,
                is_mask=True
            )

        valid_subjects += 1
        print("    -> Normalisation et Alignement OK.")

    # --- ÉTAPE 4 : Génération du dataset.json de nnU-Net ---
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

    print("\n" + "="*50)
    print(f" STRUCTURATION TERMINÉE ! Patients correctement exportés : {valid_subjects}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur IRM DCE Physiologique vers nnU-Net.")
    
    # Configurations des chemins
    parser.add_argument("--src", default="./Base_IRM", help="Dossier source des patients IRM (Post-Ingesteur).")
    parser.add_argument("--nnunet", default="./nnunet_data", help="Racine du projet nnU-Net.")
    
    # Options comportementales
    parser.add_argument("--inference", action="store_true", help="Prépare les données pour la prédiction (imagesTs).")
    parser.add_argument("--num_channels", type=int, default=3, choices=[1, 2, 3, 4], 
                        help="Nombre de phases à injecter dans le réseau (1=T0, 2=T0+Pic, etc.).")
    
    # Configurations des métadonnées (Les fameux flags !)
    parser.add_argument("--mode", type=str, default="INGESTEUR", choices=["INGESTEUR", "MAMAMIA", "BLIND"],
                        help="Stratégie pour recréer la frise chronologique d'injection.")
    parser.add_argument("--csv", type=str, default=None, 
                        help="Chemin vers le CSV (Obligatoire si --mode MAMAMIA).")
    
    args = parser.parse_args()
    
    # Vérification de sécurité
    if args.mode == "MAMAMIA" and not args.csv:
        parser.error("Le flag --csv est obligatoire lorsque le --mode MAMAMIA est activé.")
        
    extract_dce_to_nnunet_flat(
        subjects_dir=args.src, 
        nnunet_root=args.nnunet, 
        num_channels=args.num_channels,
        is_inference=args.inference,
        mode=args.mode,
        csv_path=args.csv
    )

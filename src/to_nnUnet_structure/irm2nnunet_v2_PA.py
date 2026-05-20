#!/usr/bin/env python3
r"""
Orchestrateur IRM DCE vers nnU-Net V2 - V3 (Jump-Anchored Time-Matching).

=============================================================================
NOUVEAUTÉS V3 :
- Time-Matching Absolu : Sélectionne les phases NIfTI non pas par leur index (0,1,2,3),
  mais par leur TEMPS BIOLOGIQUE absolu post-injection (ex: 0s, 90s, 180s, 360s).
- Modularité des Canaux : Supporte de 1 à 4 canaux via le flag `--num_channels`.
- Modes Datasets : 
    -> INGESTEUR : Lit le 'DCE_temporal_log.txt' généré par notre Ingesteur V6.
    -> MAMAMIA   : Parse le fichier CSV officiel de MAMA-MIA pour lire 'acquisition_times'.
    -> BLIND     : Mode survie si aucune métadonnée n'est disponible (heuristique).
=============================================================================

=============================================================================
LA STRATÉGIE TEMPORELLE ABSOLUE :
1. Détecte le "Grand Saut" (Injection du produit de contraste).
2. T0 (Baseline) = L'image juste avant le saut.
3. T1 (Wash-in) = L'image juste après le saut (Ancrage Temporel T_inj).
4. T2, T3... = Images les plus proches de (T_inj + 90s), (T_inj + 180s), etc.
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

# ============================================================================
# MODULE DE SÉLECTION TEMPORELLE (JUMP-ANCHORED TIME-MATCHING)
# Ce module garantit que nnU-Net reçoit toujours les mêmes fenêtres biologiques,
# indépendamment de la machine IRM, en s'ancrant sur le moment de l'injection.
# ============================================================================

def parse_ingesteur_log(log_path: str) -> list:
    """
    Reconstruit la frise chronologique absolue (timeline) depuis le log de l'Ingesteur V6.
    
    Exemple de sortie : [0.0, 17.0, 34.0, 70.0, 87.0...]
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
                        # On additionne l'écart au temps précédent pour avoir le temps absolu
                        timeline.append(timeline[-1] + ecart_sec)
        return timeline
    except Exception as e:
        print(f"    [ATTENTION] Échec lecture du log temporel : {e}")
        return []


def load_mamamia_database(csv_path: str) -> dict:
    """
    Parse le CSV externe (ex: MAMA-MIA) pour extraire la colonne 'acquisition_times'.
    
    Retourne un dictionnaire liant le patient à sa timeline:
    {'QIN-01': [0.0, 584.0, 714.0], 'QIN-02': [0.0, 165.0, 288.0]}
    """
    db = {}
    if not csv_path or not os.path.exists(csv_path): 
        return db
        
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id", "").strip()
                times_str = row.get("acquisition_times", "").strip()
                
                if pid and times_str:
                    try: 
                        # ast.literal_eval transforme la string "[0, 165]" en vraie liste Python
                        db[pid] = [float(t) for t in ast.literal_eval(times_str)]
                    except: 
                        pass
    except Exception as e:
        print(f"    [ERREUR CSV] Impossible de lire {csv_path}: {e}")
        
    return db

def select_best_dce_phases_jump_anchored(timeline: list, num_channels: int, nb_files_available: int) -> list:
    """
    L'Algorithme Hybride (Le "Grand Saut" + Cohérence Temporelle Absolue).
    
    Logique biologique :
    1. Trouve le plus grand écart de temps (qui correspond au moment de l'injection).
    2. T0 (Baseline sans contraste) = L'image juste AVANT ce saut.
    3. T1 (Wash-in) = L'image juste APRÈS ce saut (Ancrage T_inj).
    4. T_X (Plateau/Wash-out) = Les images les plus proches de (T_inj + 90s), (T_inj + 180s).
    """
    # --- MODE BLIND (Répartition Heuristique) ---
    if not timeline or len(timeline) < 2:
        print("    [MODE BLIND ACTIVÉ] Aucune métadonnée. Répartition heuristique de l'examen.")

        # SÉCURITÉ : Empêche la division par zéro si on ne demande qu'une seule phase (T0)
        if num_channels == 1:
            return [0]
          
        if nb_files_available <= num_channels:
            # Si on a moins de fichiers que de canaux demandés, on complète avec le dernier fichier
            return list(range(nb_files_available)) + [nb_files_available-1] * (num_channels - nb_files_available)
        else:
            # Si on a beaucoup de fichiers, on répartit équitablement du début à la fin de l'examen
            return [int(i * (nb_files_available - 1) / (num_channels - 1)) for i in range(num_channels)]

    # Sécurité : On s'assure de ne pas chercher des index au-delà des fichiers réellement sur le disque
    max_idx = min(len(timeline), nb_files_available)
    valid_timeline = np.array(timeline[:max_idx])
    
    # --- 1. DÉTECTION DU SAUT (INJECTION) ---
    # np.diff calcule l'écart entre chaque élément consécutif
    diffs = np.diff(valid_timeline)
    # np.argmax trouve l'index du plus grand écart
    jump_idx_before = int(np.argmax(diffs))
    jump_idx_after = jump_idx_before + 1
    
    # L'heure exacte (en secondes depuis le début de l'examen) où le contraste arrive
    t_injection = valid_timeline[jump_idx_after]

    selected_indices = []
    
    # --- CANAL 0 : BASELINE (Image sans contraste) ---
    selected_indices.append(jump_idx_before)
    
    # --- CANAL 1 : WASH-IN IMMEDIAT (Si on a demandé au moins 2 canaux) ---
    if num_channels >= 2:
        selected_indices.append(jump_idx_after)
        
    # --- CANAUX SUIVANTS : CIBLES CINÉTIQUES POST-INJECTION ---
    # Cibles standards pour le cancer du sein : 90s (Plateau), 180s (Wash-out précoce)
    target_deltas_post_inj = [90.0, 180.0, 360.0] 
    
    for i in range(2, num_channels):
        # On calcule le temps absolu visé
        cible_absolue = t_injection + target_deltas_post_inj[i - 2]
        
        # On cherche l'image la plus proche de cette cible, MAIS uniquement parmi les images 
        # situées APRÈS l'injection (pour éviter que l'algo ne recule dans le temps)
        temps_futurs = valid_timeline[jump_idx_after:]
        best_future_idx = int(np.argmin(np.abs(temps_futurs - cible_absolue)))
        
        # On replace l'index relatif dans le repère global de la liste complète
        vrai_index = jump_idx_after + best_future_idx
        selected_indices.append(vrai_index)

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

    # === NOUVEAU : Chargement de la base externe si demandée ===
    mamamia_db = {}
    if mode == "MAMAMIA" and csv_path:
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

        print(f"[INFO] Traitement patient : {subj} (Fichiers dispos: {nb_fichiers_dispos})")

        # === NOUVEAU : GÉNÉRATION DE LA TIMELINE ===
        timeline = []
        if mode == "INGESTEUR":
            log_path = os.path.join(imgs_dir, "DCE_temporal_log.txt")
            timeline = parse_ingesteur_log(log_path)
        elif mode == "MAMAMIA":
            timeline = mamamia_db.get(subj, [])
            if not timeline:
                print(f"    [ALERTE] Patient {subj} absent du CSV. Bascule en mode Aveugle.")

        # Si mode == "BLIND", timeline reste volontairement vide [] pour déclencher la sécurité.

        # === NOUVEAU : SÉLECTION PHYSIOLOGIQUE DES PHASES ===
        selected_indices = select_best_dce_phases_jump_anchored(timeline, num_channels, nb_fichiers_dispos)
        
        # Affichage pour le suivi console
        temps_reels = [f"{timeline[i]}s" if i < len(timeline) else "N/A" for i in selected_indices]
        print(f"    -> Index choisis : {selected_indices}")
        print(f"    -> Temps réels   : {temps_reels}")
        
        # On remplace la liste totale par notre sélection stricte
        phases_selectionnees_paths = [fichiers_images[idx] for idx in selected_indices]

        # --- ÉTAPE 1 : Alignement inter-phases ---
        # ATTENTION : Remplacer `fichiers_images[0]` par `phases_selectionnees_paths[0]`
        ref_phase_path = phases_selectionnees_paths[0] 
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
    parser.add_argument("--src", default="./Base_IRM", help="Dossier source des patients IRM")
    parser.add_argument("--nnunet", default="./nnunet_data", help="Racine nnU-Net")
    parser.add_argument("--inference", action="store_true", help="Prépare les données pour la prédiction (imagesTs)")
    
    # --- NOUVEAUX FLAGS ---
    parser.add_argument("--num_channels", type=int, default=3, choices=[1, 2, 3, 4], 
                        help="Nombre de phases à extraire (ex: 3 = T0, Wash-in, +90s)")
    parser.add_argument("--mode", type=str, default="INGESTEUR", choices=["INGESTEUR", "MAMAMIA", "BLIND"],
                        help="Source de la chronologie (Log interne, CSV externe, ou Aveugle).")
    parser.add_argument("--csv", type=str, default=None, 
                        help="Chemin vers le metadata.csv (Requis si mode MAMAMIA).")
                        
    args = parser.parse_args()
    
    if args.mode == "MAMAMIA" and not args.csv:
        parser.error("Le flag --csv est obligatoire avec le mode MAMAMIA.")
        
    # N'oublie pas d'ajouter les nouveaux paramètres à l'appel de ta fonction !
    extract_dce_to_nnunet_flat(
        subjects_dir=args.src, 
        nnunet_root=args.nnunet, 
        num_channels=args.num_channels,
        is_inference=args.inference,
        mode=args.mode,
        csv_path=args.csv
    )

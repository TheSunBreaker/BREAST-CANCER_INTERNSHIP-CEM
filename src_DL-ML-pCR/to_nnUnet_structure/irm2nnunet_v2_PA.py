#!/usr/bin/env python3
r"""
Orchestrateur IRM DCE vers nnU-Net V2 - V3.1 (Jump-Anchored Time-Matching + Filtre Clinique).

=============================================================================
NOUVEAUTÉS V3.1 (Le Filtre Oncologique) :
- Support Natif Excel/CSV : Gère directement les fichiers .xlsx et .csv pour les métadonnées.
- Filtre Triple Négatif : Option `--triple_neg_only` pour exclure automatiquement
  les patients dont les récepteurs (ER, PR, HER2) ne sont pas strictement à 0.
- Structuration de la DB : mamamia_db contient désormais la timeline ET le statut TNBC.
=============================================================================
NOUVEAUTÉS V3 :
- Time-Matching Absolu : Sélectionne les phases NIfTI non pas par leur index (0,1,2,3),
  mais par leur TEMPS BIOLOGIQUE absolu post-injection (ex: 0s, 90s, 180s, 360s).
- Modularité des Canaux : Supporte de 1 à 4 canaux via le flag `--num_channels`.
- Modes Datasets : 
    -> INGESTEUR : Lit le 'DCE_temporal_log.txt' généré par notre Ingesteur V6.
    -> MAMAMIA   : Parse le fichier CSV officiel de MAMA-MIA pour lire 'acquisition_times'.
    -> BLIND     : Mode survie si aucune métadonnée n'est disponible (heuristique).
==
  ============================================================================
  LA STRATÉGIE TEMPORELLE ABSOLUE :
  1. Détecte le "Grand Saut" (Injection du produit de contraste).
  2. T_0 (Baseline) = L'image juste avant le saut.
  3. T_1 (Wash-in) = L'image juste après le saut (Ancrage Temporel T_inj).
  4. T_2 = Image la plus proche de (T_inj + 90s)  [Pic précoce / Plateau]
  5. T_3 = Image la plus proche de (T_inj + 180s) [Wash-out tardif]
# =============================================================================


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
    Retourne une description textuelle des cibles physiologiques pour les logs console.
    Reflète la stratégie Jump-Anchored (Ancrage sur l'injection).
    """
    targets = ["T_Baseline (Canal 0)"] 
    if num_channels >= 2:
        targets.append("T_inj / Wash-in (Canal 1)")
    if num_channels >= 3:
        targets.append("T_inj + 90s (Canal 2)")
    if num_channels >= 4:
        targets.append("T_inj + 180s (Canal 3)")
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


def load_mamamia_database(file_path: str) -> dict:
    """
    Parse le fichier clinique externe (CSV ou XLSX).
    Extrait la timeline ET le statut hormonal (ER, PR, HER2) pour le filtre TNBC.
    
    Retourne un dictionnaire imbriqué :
    {
        'QIN-01': {'timeline': [0.0, 584.0, 714.0], 'is_tnbc': True},
        'QIN-02': {'timeline': [0.0, 165.0, 288.0], 'is_tnbc': False}
    }
    """
    db = {}
    if not file_path or not os.path.exists(file_path): 
        return db
        
    try:
        rows = []
        # --- SUPPORT EXCEL (.xlsx) ET CSV ---
        if file_path.lower().endswith(('.xlsx', '.xls')):
            # Si c'est un Excel, on importe pandas à la volée
            import pandas as pd
            df = pd.read_excel(file_path)
            # Nettoyage des noms de colonnes (minuscule, sans espaces)
            df.columns = df.columns.str.strip().str.lower()
            # Transformation en liste de dictionnaires pour homogénéiser avec le csv.DictReader
            rows = df.to_dict('records')
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Nettoyage des clés en minuscule pour éviter les bugs (ex: "ER" vs "er")
                rows = [{k.strip().lower() if k else '': v for k, v in row.items()} for row in reader]

        # --- PARSING DES DONNÉES CLINIQUES ---
        for row in rows:
            pid = str(row.get("patient_id", "")).strip()
            times_str = str(row.get("acquisition_times", "")).strip()
            
            # Extraction du statut hormonal. 
            # On met "1" par défaut si vide pour éviter qu'une valeur manquante
            # ne soit comptée à tort comme "0" (Triple Négatif).
            er = str(row.get("er", "1")).strip()
            pr = str(row.get("pr", "1")).strip()
            her2 = str(row.get("her2", "1")).strip()
            
            # On ignore les lignes sans ID ou sans timeline (NaN)
            if pid and times_str and str(times_str).lower() != 'nan':
                try: 
                    timeline = [float(t) for t in ast.literal_eval(times_str)]
                    
                    # Un patient est Triple Négatif SI ET SEULEMENT SI les 3 marqueurs sont à 0.
                    # On autorise les strings "0" et "0.0" pour parer aux formats d'export de données.
                    is_tnbc = (er in ["0", "0.0", "0,0"]) and (pr in ["0", "0.0", "0,0"]) and (her2 in ["0", "0.0", "0,0"])
                    
                    db[pid] = {
                        "timeline": timeline,
                        "is_tnbc": is_tnbc
                    }
                except Exception:
                    pass
    except Exception as e:
        print(f"    [ERREUR BASE CLINIQUE] Impossible de lire {file_path}: {e}")
        
    return db

def select_best_dce_phases_jump_anchored(timeline: list, num_channels: int, nb_files_available: int) -> list:
    """
    MATRICE DE SECOURS MULTI-NIVEAUX (V4 Robustesse Clinique).
    
    Cette fonction détermine les index optimaux des volumes 3D à extraire pour nnU-Net.
    Elle implémente la détection du moment de l'injection (Ancrage) et projette les
    cibles physiologiques standards du cancer du sein (90s, 180s, 360s).
    
    LOGIQUE DE SÉCURITÉ EN 3 NIVEAUX :
    Niveau 1 : Détection et validation du saut temporel (Pause d'injection détectée).
    Niveau 2 : Secours pour acquisition continue (Uniforme) -> Ancrage forcé à l'index 1.
    Niveau 3 : Mode BLIND total -> Répartition heuristique de secours.
    """
    
    # ==========================================
    # NIVEAU 3 : MODE BLIND (Parachute de secours)
    # ==========================================
    if not timeline or len(timeline) < 2:
        print("    [ALERT - SÉCURITÉ NIVEAU 3] Timeline vide ou insuffisante. Bascule en mode BLIND.")
        if num_channels == 1:
            return [0]
        if nb_files_available <= num_channels:
            # Remplissage par duplication de la dernière phase si pas assez de fichiers
            return list(range(nb_files_available)) + [nb_files_available-1] * (num_channels - nb_files_available)
        else:
            # Échantillonnage géométrique régulier sur l'ensemble de l'examen
            return [int(i * (nb_files_available - 1) / (num_channels - 1)) for i in range(num_channels)]

    # Alignement défensif entre la timeline des métadonnées et les fichiers physiques réels
    max_idx = min(len(timeline), nb_files_available)
    valid_timeline = np.array(timeline[:max_idx])
    
    # Calcul des deltas de temps entre chaque phase consécutive
    diffs = np.diff(valid_timeline)
    
    if len(diffs) == 0:
        return [0] * num_channels

    max_gap = float(np.max(diffs))
    median_gap = float(np.median(diffs))
    
    # ==========================================
    # NIVEAU 1 & 2 : RECHERCHE DE L'ANCRAGE D'INJECTION
    # ==========================================
    # Seuil clinique de validation du saut : le plus grand écart doit être au moins 1.5 fois
    # plus grand que l'écart médian de la séquence. Sinon, l'acquisition est considérée comme continue.
    if max_gap > (1.5 * median_gap):
        # Niveau 1 : Le saut est validé mathématiquement
        jump_idx_before = int(np.argmax(diffs))
        jump_idx_after = jump_idx_before + 1
        print(f"    [SÉCURITÉ NIVEAU 1] Saut d'injection détecté : Phase {jump_idx_before} -> Phase {jump_idx_after} (Delta: {max_gap:.1f}s)")
    else:
        # Niveau 2 : Acquisition continue (Deltas uniformes, ex: 15s, 15s, 15s...)
        # Heuristique standard de l'état de l'art : On assume que la toute première phase (index 0) 
        # est le masque à blanc (Baseline) et que l'injection produit ses effets à l'index 1.
        jump_idx_before = 0
        jump_idx_after = 1
        print(f"    [SÉCURITÉ NIVEAU 2] Acquisition continue détectée (Pas de saut temporel). Ancrage forcé : Phase 0 -> Phase 1")

    # Définition du temps d'ancrage absolu (Le T = 0 biologique du produit de contraste)
    t_washin_initial = valid_timeline[jump_idx_after]
    selected_indices = []
    
    # --- CANAL 0 : BASELINE (Image anatomique pure sans contraste) ---
    selected_indices.append(jump_idx_before)
    
    # --- CANAL 1 : WASH-IN IMMÉDIAT (Arrivée initiale du produit) ---
    if num_channels >= 2:
        selected_indices.append(jump_idx_after)
        
    # --- CANAUX SUIVANTS : COHÉRENCE CINÉTIQUE PHYSIOLOGIQUE ---
    # Jalons BI-RADS standards : +90s (Pic précoce / Plateau), +180s (Wash-out / Phase tardive)
    target_deltas_post_inj = [90.0, 180.0] 
    
    for i in range(2, num_channels):
        cible_absolue = t_washin_initial + target_deltas_post_inj[i - 2]
        
        # Restriction causale : On ne cherche que parmi les phases situées APRÈS l'injection
        # pour empêcher le modèle de reculer accidentellement dans le temps en cas de bruit
        temps_futurs = valid_timeline[jump_idx_after:]
        best_future_idx = int(np.argmin(np.abs(temps_futurs - cible_absolue)))
        
        vrai_index = jump_idx_after + best_future_idx
      
        # Si l'index choisi est inférieur ou égal au canal précédent, on force 
        # l'avancement d'au moins 1 frame (si disponible sur le disque)
        if selected_indices and vrai_index <= selected_indices[-1]:
            vrai_index = min(selected_indices[-1] + 1, nb_files_available - 1)
            
        selected_indices.append(vrai_index)

    # --- AUDIT QUALITÉ ET LOGGING DES DUPLICATIONS ---
    # On vérifie si l'hétérogénéité du protocole a forcé l'algorithme à dupliquer des canaux
    if len(selected_indices) != len(set(selected_indices)):
        print(f"    [NOTE CLINIQUE] Profil basse résolution temporelle détecté. Canaux dupliqués pour la stabilité nnU-Net.")

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
    csv_path: str = None,
    triple_neg_only: bool = False #
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
    if triple_neg_only:
        print(" [!] FILTRE ACTIF : Uniquement les tumeurs Triple Négatives (TNBC) [!]")
    print(f" {len(subjects)} patients trouvés dans le dossier source.")
    print(f"==================================================\n")

    valid_subjects = 0

    # --- NOUVEAU : LISTES DE SUIVI DES ERREURS ---
    erreurs_normalisation = []
    erreurs_masque = []
    patients_exclus_tnbc = []
    exclus_hors_base = []

    # --- NOUVEAU : INITIALISATION DU LOG GLOBAL ---
    rapport_global = [
        "==================================================",
        f" RAPPORT D'EXTRACTION DCE -> nnU-Net",
        f" Mode: {mode} | Canaux visés: {num_channels}",
        "==================================================\n"
    ]

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
            # Si le fichier log de notre ingesteur est absent ou vide pour ce patient
            if not timeline:
                print(f"    [REJET STRICT] Log temporel introuvable pour {subj}. Exclusion du dataset.")
                rapport_global.append(f"[EXCLUSION] {subj} | Fichier DCE_temporal_log.txt manquant ou vide.")
                exclus_hors_base.append(subj)
                continue # Interruption immédiate du traitement pour ce sujet
              
        elif mode == "MAMAMIA":
            patient_info = mamamia_db.get(subj, {})
            timeline = patient_info.get("timeline", [])
            is_tnbc = patient_info.get("is_tnbc", False)
            
            # --- LE FILTRE TRIPLE NÉGATIF ---
            # Si le filtre est activé et que le patient n'est pas TNBC (ou info manquante),
            # on passe silencieusement au patient suivant.
            if triple_neg_only and not is_tnbc:
                print(f"    -> [FILTRE CLINIQUE] Patient ignoré (Statut Hormonal non Triple Négatif).")
                rapport_global.append(f"[FILTRÉ] {subj} | Exclu (Non Triple Négatif)")
                patients_exclus_tnbc.append(subj)
                continue
                
            if not timeline:
                print(f"    [REJET STRICT] Patient {subj} absent du CSV MAMA-MIA. Exclusion du dataset.")
                rapport_global.append(f"[EXCLUSION] {subj} | Absent des enregistrements du CSV de référence.")
                exclus_hors_base.append(subj)
                continue # Interruption immédiate du traitement pour ce sujet
              
        # Si mode == "BLIND", timeline reste volontairement vide [] pour déclencher la sécurité.

        # === SÉLECTION PHYSIOLOGIQUE DES PHASES ===
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
            rapport_global.append(f"!!!!!!!!!!!!![ERREUR] {subj} | Normalisation échouée : {e}!!!!!!!!!!!!!")
            erreurs_normalisation.append(subj)
            continue

        # --- ÉTAPE 3 : Alignement du Masque (SEULEMENT EN ENTRAÎNEMENT) ---
        if not is_inference:
            fichiers_masques = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
            if not fichiers_masques:
                print(f"   [ERREUR] Masque introuvable au moment de l'alignement pour {subj}.")
                rapport_global.append(f"!!!!!!!!!!!!![ERREUR] {subj} | Masque introuvable.!!!!!!!!!!!!!")
                erreurs_masque.append(subj)
                continue
                
            dst_mask = os.path.join(labelsTr_dir, f"{subj}.nii.gz")
            enforce_strict_alignment(
                ref_path=ref_phase_path,
                moving_path=fichiers_masques[0], # Le masque est aligné sur le T0
                out_path=dst_mask,
                is_mask=True
            )

        # --- NOUVEAU : AJOUT AU LOG DES BONS PATIENTS ---
        rapport_global.append(f"[SUCCÈS] {subj} | Index: {selected_indices} | Temps réels: {temps_reels}")

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

    # --- NOUVEAU : BILAN STATISTIQUE DES ERREURS DANS LE LOG ---
    rapport_global.append("\n==================================================")
    rapport_global.append("               BILAN DES ERREURS                  ")
    rapport_global.append("==================================================")
    
    # 1. Suivi des pannes techniques de traitement de l'image
    rapport_global.append(f"Erreurs de Normalisation : {len(erreurs_normalisation)}")
    if erreurs_normalisation:
        rapport_global.append(f"   -> Patients concernés : {', '.join(erreurs_normalisation)}")
        
    # 2. Suivi des pannes d'annotations (Vérité terrain manquante pour l'entraînement)
    rapport_global.append(f"Erreurs de Masques manquants : {len(erreurs_masque)}")
    if erreurs_masque:
        rapport_global.append(f"   -> Patients concernés : {', '.join(erreurs_masque)}")

    # 3. Suivi du contrôle qualité de cohérence de la base de données (Nouveau !)
    rapport_global.append(f"Patients exclus car absents de la base (Pas de tracking temporel) : {len(exclus_hors_base)}")
    if exclus_hors_base:
        rapport_global.append(f"   -> Identifiants des exclus : {', '.join(exclus_hors_base)}")

    # 4. Suivi du tri triple neg
    rapport_global.append(f"Patients exclus par le filtre TNBC : {len(patients_exclus_tnbc)}")
    if patients_exclus_tnbc:
        rapport_global.append(f"   -> Patients : {', '.join(patients_exclus_tnbc)}")

    # --- NOUVEAU : SAUVEGARDE DU RAPPORT TEXTUEL ---
    rapport_global.append("\n==================================================")
    rapport_global.append(f" STRUCTURATION TERMINÉE ! Patients valides : {valid_subjects}")
    rapport_global.append("==================================================")
    
    # On le sauvegarde à la racine du projet nnU-Net pour le retrouver facilement
    chemin_rapport = os.path.join(nnunet_root, f"rapport_extraction_{channel_prefix}_{mode}.txt")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write("\n".join(rapport_global))

    print(f"\n==================================================")
    print(f" PIPELINE TERMINÉ ! Patients valides : {valid_subjects}")
    print(f" -> Rapport d'extraction sauvegardé ici : {chemin_rapport}") # <-- INFO CONSOLE
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur IRM DCE Physiologique vers nnU-Net.")
    parser.add_argument("--src", default="./Base_IRM", help="Dossier source des patients IRM")
    parser.add_argument("--nnunet", default="./nnunet_data", help="Racine nnU-Net")
    parser.add_argument("--inference", action="store_true", help="Prépare les données pour la prédiction (imagesTs)")
    
    parser.add_argument("--num_channels", type=int, default=3, choices=[1, 2, 3, 4], 
                        help="Nombre de phases à extraire (ex: 3 = T0, Wash-in, +90s)")
    parser.add_argument("--mode", type=str, default="INGESTEUR", choices=["INGESTEUR", "MAMAMIA", "BLIND"],
                        help="Source de la chronologie (Log interne, CSV externe, ou Aveugle).")
    parser.add_argument("--csv", type=str, default=None, 
                        help="Chemin vers le metadata.csv (Requis si mode MAMAMIA).")

    # --- NOUVEAU FLAG ---
    parser.add_argument("--triple_neg_only", action="store_true",
                        help="Exclut les patients qui ne sont pas strictement Triple Négatifs (ER=0, PR=0, HER2=0).")
                        
    args = parser.parse_args()
    
    if args.mode == "MAMAMIA" and not args.csv:
        parser.error("Le flag --csv est obligatoire avec le mode MAMAMIA.")
        
    extract_dce_to_nnunet_flat(
        subjects_dir=args.src, 
        nnunet_root=args.nnunet, 
        num_channels=args.num_channels,
        is_inference=args.inference,
        mode=args.mode,
        csv_path=args.csv,
        triple_neg_only=args.triple_neg_only
    )

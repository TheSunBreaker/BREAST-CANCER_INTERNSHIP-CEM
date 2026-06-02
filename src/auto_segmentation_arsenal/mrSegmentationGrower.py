#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script d'Expansion des Masques Mammaires via Region Growing (SimpleITK) - V2
===============================================================================
Rôle :
  Prend les masques générés par TotalSegmentator (qui sous-estiment la zone)
  et les étend en utilisant l'intensité (HU) réelle du CT-scan de la patiente.
  L'algorithme s'arrête de lui-même lorsqu'il rencontre l'air ou l'os/muscle.

Nouveautés de cette version :
  - Génération automatique d'un rapport de log (TXT) à la fin de l'exécution
    détaillant les succès, les omissions et les erreurs.

Méthode :
  1. Chargement du CT et du masque initial.
  2. Léger lissage du CT pour éviter que le bruit du scanner ne bloque la croissance.
  3. Érosion du masque initial pour cibler le centre de la glande (graines/seeds).
  4. Croissance de région (ConfidenceConnected) basée sur les statistiques HU.
  5. Fusion (OU logique) du masque initial et de l'expansion pour ne rien perdre.
===============================================================================
"""

import argparse
from pathlib import Path
import random
from datetime import datetime
import SimpleITK as sitk
import numpy as np

def expand_breast_mask(ct_path: Path, mask_path: Path, output_path: Path, multiplier: float, iterations: int):
    """
    Exécute la croissance de région sur un scanner CT à partir d'un masque existant.
    Retourne un tuple : (Statut_booléen, Message_ou_Raison)
    """
    print(f"  [TRAITEMENT] Analyse de l'image en cours...")
    
    # 1. Chargement des images
    # Le CT contient les informations d'intensité (Unités Hounsfield - HU)
    ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32) # Float32 recommandé pour les calculs ITK
    mask_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    
    # 2. Lissage du CT (Smoothing)
    # On applique un léger flou pour réduire le bruit (granulométrie) du CT.
    # Cela permet à la "tache d'huile" de s'étendre plus fluidement.
    smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smoothing_filter.SetSigma(1.0) # Sigma de 1 mm (léger)
    ct_smoothed = smoothing_filter.Execute(ct_img)
    
    # 3. Extraction des "Graines" (Seeds)
    # L'algorithme a besoin de points de départ (x, y, z).
    # Pour s'assurer qu'on ne part pas d'une zone périphérique incertaine (comme la peau), 
    # on érode le masque de TotalSegmentator de 2 voxels.
    eroded_mask = sitk.BinaryErode(mask_img, [2, 2, 2])
    
    # Passage en NumPy pour manipuler les coordonnées facilement
    np_eroded = sitk.GetArrayFromImage(eroded_mask)
    
    # Récupération des indices où le masque érodé vaut 1 (présence de tissu)
    # Attention: NumPy utilise l'ordre (z, y, x)
    z_idx, y_idx, x_idx = np.where(np_eroded == 1)
    
    if len(z_idx) == 0:
        return False, "Masque initial trop petit après érosion (impossible de placer des graines)."

    # On sélectionne aléatoirement 50 points (graines) pour amorcer l'algorithme.
    # Trop de graines ralentirait inutilement le calcul sans améliorer le résultat.
    nb_seeds = min(50, len(z_idx))
    random_indices = random.sample(range(len(z_idx)), nb_seeds)
    
    seed_list = []
    for idx in random_indices:
        # SimpleITK utilise l'ordre (x, y, z), il faut donc inverser par rapport à NumPy !
        seed_list.append((int(x_idx[idx]), int(y_idx[idx]), int(z_idx[idx])))
        
    # 4. Configuration et exécution du Region Growing
    region_grow_filter = sitk.ConfidenceConnectedImageFilter()
    region_grow_filter.SetSeedList(seed_list)
    region_grow_filter.SetMultiplier(multiplier)          # Tolérance de la croissance (c dans la formule)
    region_grow_filter.SetNumberOfIterations(iterations)  # Nombre de fois où le seuil est recalculé
    region_grow_filter.SetInitialNeighborhoodRadius(2)    # Rayon autour de la graine pour le calcul statistique initial
    region_grow_filter.SetReplaceValue(1)                 # Valeur donnée à la nouvelle zone segmentée
    
    # Exécution sur le CT lissé
    expanded_mask = region_grow_filter.Execute(ct_smoothed)
    
    # 5. Fusion des masques
    # L'algorithme peut parfois "oublier" une zone très dense du masque d'origine.
    # On fait un OU logique pour s'assurer que le Masque Final = Masque Initial + Expansion.
    final_mask = sitk.Or(mask_img, expanded_mask)
    
    # Nettoyage : remplir les éventuels petits trous créés par l'expansion
    # Cela garantit un masque bien plein (crucial pour l'extraction de radiomiques)
    final_mask = sitk.BinaryFillhole(final_mask)
    
    # Sauvegarde
    sitk.WriteImage(final_mask, str(output_path))
    return True, f"Succès (Sauvegardé sous {output_path.name})"


def generate_log_report(output_dir: Path, stats: dict, total_masks: int):
    """
    Génère un fichier texte récapitulatif de l'exécution du script.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"rapport_expansion_{timestamp}.txt"
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("    RAPPORT D'EXPANSION DES MASQUES MAMMAIRES\n")
        f.write("====================================================\n")
        f.write(f"Date et heure de fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Total de masques analysés : {total_masks}\n\n")
        
        f.write("--- RÉSUMÉ ---\n")
        f.write(f"Traités avec succès : {len(stats['success'])}\n")
        f.write(f"Ignorés (Skip)      : {len(stats['skipped'])}\n")
        f.write(f"Échecs (Erreurs)    : {len(stats['failed'])}\n\n")
        
        if stats['skipped']:
            f.write("--- DÉTAIL DES PATIENTS IGNORÉS ---\n")
            for patient, raison in stats['skipped']:
                f.write(f" - {patient} : {raison}\n")
            f.write("\n")
            
        if stats['failed']:
            f.write("--- DÉTAIL DES ÉCHECS ---\n")
            for patient, erreur in stats['failed']:
                f.write(f" - {patient} : Erreur technique -> {erreur}\n")
            f.write("\n")
            
    print(f"\n📄 Un rapport détaillé a été généré : {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Extension des masques mammaires via Region Growing avec Log.")
    
    # Arguments d'arborescence
    parser.add_argument("--ct_dir", type=Path, default=Path("./Base_PETCT"), 
                        help="Dossier contenant les sous-dossiers patients avec les CT initiaux.")
    parser.add_argument("--mask_dir", type=Path, default=Path("./Base_PETCT_BreastMasks"), 
                        help="Dossier contenant les masques TotalSegmentator générés à l'étape précédente.")
    parser.add_argument("--output_dir", type=Path, default=Path("./Base_PETCT_BreastMasks_Expanded"), 
                        help="Dossier de sauvegarde des nouveaux masques (évite l'écrasement).")
    
    # Paramètres de l'algorithme
    parser.add_argument("--multiplier", type=float, default=2.5, 
                        help="Tolérance de l'algorithme. Plus c'est haut, plus ça s'étend. (Défaut: 2.5)")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Nombre d'itérations de mise à jour des statistiques. (Défaut: 3)")
    parser.add_argument("--ct-suffix", default="_TDM_", help="Marqueur du fichier CT (Défaut: _TDM_)")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionnaire pour le suivi du log
    execution_stats = {
        "success": [],
        "skipped": [],
        "failed": []
    }
    
    # On parcourt les masques existants
    masks = list(args.mask_dir.glob("*_breast_mask.nii.gz"))
    if not masks:
        print(f"❌ Aucun masque trouvé dans {args.mask_dir}")
        return
        
    print(f"🔍 {len(masks)} masque(s) trouvé(s). Début de l'expansion avec tolérance à {args.multiplier}...\n")
    
    for mask_path in masks:
        # On extrait l'ID patient en se basant sur le nommage standard
        patient_id = mask_path.name.replace("_breast_mask.nii.gz", "")
        print(f"[{patient_id}]")
        
        # On recherche le CT correspondant dans l'arborescence
        patient_dir = args.ct_dir / patient_id
        imgs_dir = patient_dir / "imgs"
        
        if not imgs_dir.exists():
            print(f"  [SKIP] Dossier images introuvable.")
            execution_stats["skipped"].append((patient_id, "Dossier 'imgs' introuvable dans le CT_dir"))
            continue
            
        ct_files = list(imgs_dir.glob(f"*{args.ct_suffix}*.nii.gz"))
        if not ct_files:
            print(f"  [SKIP] Aucun fichier CT (*{args.ct_suffix}*.nii.gz) trouvé.")
            execution_stats["skipped"].append((patient_id, "CT scan introuvable"))
            continue
            
        ct_file = ct_files[0]
        output_mask_path = args.output_dir / f"{patient_id}_breast_mask_expanded.nii.gz"
        
        try:
            # Lancement de la fonction métier
            success, message = expand_breast_mask(ct_file, mask_path, output_mask_path, args.multiplier, args.iterations)
            
            if success:
                print(f"  [OK] {message}")
                execution_stats["success"].append(patient_id)
            else:
                print(f"  [SKIP] {message}")
                execution_stats["skipped"].append((patient_id, message))
                
        except Exception as e:
            print(f"  [ÉCHEC] Erreur inattendue : {e}")
            execution_stats["failed"].append((patient_id, str(e)))

    print("\n=== EXPANSION DES MASQUES TERMINÉE ===")
    
    # Génération du fichier de log à la fin de la boucle
    generate_log_report(args.output_dir, execution_stats, len(masks))

if __name__ == "__main__":
    main()

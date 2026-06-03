#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script d'Expansion des Masques Mammaires via Region Growing (SimpleITK) - V6
===============================================================================
Rôle :
  Prend les masques générés par TotalSegmentator (qui sous-estiment la zone mammaire)
  et les étend en utilisant l'intensité (HU) réelle du CT-scan de la patiente.
  L'algorithme s'arrête de lui-même lorsqu'il rencontre l'air ou nos boucliers.

Nouveautés de la V6 (Le Bouclier IA Infaillible) :
  - BOUCLIER 1 (Anatomique DL) : Abandon du seuillage osseux (vulnérable aux fuites 
    3D par les cartilages, le cou et le diaphragme). Remplacement par une "Zone 
    Interdite" stricte combinant les masques TotalSegmentator (Cœur, Sternum, Côtes). 
    Verrouille définitivement le médiastin tout en permettant à la croissance d'aller 
    chercher les tumeurs profondes collées contre la paroi pré-pectorale.

Héritage des versions précédentes :
  - NORMALISATION LPS (V5) : Force l'alignement des images sur le repère anatomique 
    du patient. Le script est 100% insensible à la position (ventre ou dos).
  - BOUCLIER 2 (Géométrique - V4) : Mur virtuel arrière (avec marge de profondeur) 
    pour couper la fuite latérale vers la graisse sous-cutanée du dos.
  - Fermeture Morphologique (V3) : Comble les trous internes (hétérogénéité de la graisse).
  - Rapport de Log détaillé (V2) : Suivi des succès, skips et erreurs en fichier texte.
===============================================================================
"""

import argparse
from pathlib import Path
import random
from datetime import datetime
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi

def expand_breast_mask(ct_path: Path, mask_path: Path, organs_dir: Path, output_path: Path, multiplier: float, iterations: int):
  
    """
    Exécute la croissance de région sur un scanner CT à partir d'un masque existant.
    Retourne un tuple : (Statut_booléen, Message_ou_Raison)
    """
    print(f"  [TRAITEMENT] Analyse de l'image en cours...")
    
    # ---------------------------------------------------------
    # 1. Chargement des images
    # ---------------------------------------------------------
    ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    mask_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    
    # ---------------------------------------------------------
    # 2. NOUVEAUTÉ V5 : Standardisation de l'orientation en LPS
    # ---------------------------------------------------------
    # Cette étape lit le header DICOM/NIfTI et réorganise les pixels pour que
    # l'axe X = Droite->Gauche, l'axe Y = Ventre->Dos, et l'axe Z = Pieds->Tête.
    # C'est ce qui garantit l'immunité face aux positions Ventre/Dos.
    print("  -> Normalisation de l'orientation dans l'espace patient (LPS)...")
    ct_img = sitk.DICOMOrient(ct_img, 'LPS')
    mask_img = sitk.DICOMOrient(mask_img, 'LPS')
    
    # Extraction des matrices NumPy APRÈS alignement rigoureux
    ct_np = sitk.GetArrayFromImage(ct_img)
    mask_np = sitk.GetArrayFromImage(mask_img)

    # ---------------------------------------------------------
    # 3. BOUCLIER 1 : La Muraille Anatomique (Deep Learning)
    # ---------------------------------------------------------
    print("  -> Création de la muraille anatomique (via masques TotalSegmentator)...")
    
    forbidden_mask_np = np.zeros_like(ct_np, dtype=bool)
    
    # Chargement du coeur, du sternum et du cartilage intercostal
    for organe in ["heart.nii.gz", "sternum.nii.gz", "costal_cartilages.nii.gz"]:
        organe_path = organs_dir / organe
        if organe_path.exists():
            org_img = sitk.ReadImage(str(organe_path))
            org_img = sitk.DICOMOrient(org_img, 'LPS')
            org_np = sitk.GetArrayFromImage(org_img)
            forbidden_mask_np = np.logical_or(forbidden_mask_np, org_np > 0)
            
    # Chargement de toutes les côtes générées
    for rib_path in organs_dir.glob("*rib*.nii.gz"):
        rib_img = sitk.ReadImage(str(rib_path))
        rib_img = sitk.DICOMOrient(rib_img, 'LPS')
        rib_np = sitk.GetArrayFromImage(rib_img)
        forbidden_mask_np = np.logical_or(forbidden_mask_np, rib_np > 0)

    # Dilatation du bouclier (3 voxels) pour sceller parfaitement les espaces intercostaux
    forbidden_sitk = sitk.GetImageFromArray(forbidden_mask_np.astype(np.uint8))
    forbidden_sitk = sitk.BinaryDilate(forbidden_sitk, [3, 3, 3])
    forbidden_mask_np_dilated = sitk.GetArrayFromImage(forbidden_sitk)

    # Application de la Zone Interdite
    ct_np[forbidden_mask_np_dilated == 1] = -1000

    # ---------------------------------------------------------
    # 4. BOUCLIER 2 : Le Mur Virtuel (Anti-fuite dos - V4/V3)
    # ---------------------------------------------------------
    print("  -> Création du mur virtuel arrière (anti-fuite vers le dos)...")
    # Grâce à la normalisation LPS, y_idx représente de façon CERTAINE le dos quand il augmente.
    z_idx, y_idx, x_idx = np.where(mask_np > 0)
    
    if len(y_idx) == 0:
        return False, "Masque initial TotalSegmentator complètement vide."

    # On cherche la limite arrière du masque initial
    y_max = y_idx.max()
    
    # On laisse 30 voxels de liberté (3 à 5 cm) pour englober la poitrine profonde,
    # puis on dresse le mur virtuel de -1000 HU pour verrouiller la graisse du dos.
    marge_profondeur = 30
    limite_arriere = min(y_max + marge_profondeur, ct_np.shape[1] - 1)
    ct_np[:, limite_arriere:, :] = -1000 
    
    # Reconversion en image SimpleITK avec transfert des métadonnées propres
    ct_img_constrained = sitk.GetImageFromArray(ct_np)
    ct_img_constrained.CopyInformation(ct_img)

    # ---------------------------------------------------------
    # 5. Lissage du CT (Smoothing)
    # ---------------------------------------------------------
    # Un léger flou gaussien élimine le bruit du scanner pour fluidifier la croissance.
    smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smoothing_filter.SetSigma(1.0)
    ct_smoothed = smoothing_filter.Execute(ct_img_constrained)
    
    # ---------------------------------------------------------
    # 6. Extraction des "Graines" (Seeds)
    # ---------------------------------------------------------
    # Érosion de 2 voxels pour planter les graines loin de la peau et de l'air.
    eroded_mask = sitk.BinaryErode(mask_img, [2, 2, 2])
    np_eroded = sitk.GetArrayFromImage(eroded_mask)
    z_idx_seed, y_idx_seed, x_idx_seed = np.where(np_eroded == 1)
    
    if len(z_idx_seed) == 0:
        return False, "Masque initial trop petit après érosion (impossible de placer des graines)."

    # Sélection aléatoire de 50 points
    nb_seeds = min(50, len(z_idx_seed))
    random_indices = random.sample(range(len(z_idx_seed)), nb_seeds)
    
    seed_list = []
    for idx in random_indices:
        seed_list.append((int(x_idx_seed[idx]), int(y_idx_seed[idx]), int(z_idx_seed[idx])))
        
    # ---------------------------------------------------------
    # 7. Configuration et exécution du Region Growing
    # ---------------------------------------------------------
    print(f"  -> Croissance de région en cours (Tolérance: {multiplier})...")
    region_grow_filter = sitk.ConfidenceConnectedImageFilter()
    region_grow_filter.SetSeedList(seed_list)
    region_grow_filter.SetMultiplier(multiplier)          
    region_grow_filter.SetNumberOfIterations(iterations)  
    region_grow_filter.SetInitialNeighborhoodRadius(2)
    region_grow_filter.SetReplaceValue(1)
    
    expanded_mask = region_grow_filter.Execute(ct_smoothed)
    
    # ---------------------------------------------------------
    # 8. Fusion des masques et Traitement Morphologique
    # ---------------------------------------------------------
    final_mask = sitk.Or(mask_img, expanded_mask)
    
    print("  -> Traitement morphologique (comblement des trous internes)...")
    # Fermeture Morphologique (rayon 10 voxels) pour effacer les trous "de gruyère"
    # formés par l'hétérogénéité de la graisse sans modifier la frontière externe.
    final_mask = sitk.BinaryMorphologicalClosing(final_mask, [10, 10, 10])
    final_mask = sitk.BinaryFillhole(final_mask)
    
    # Sauvegarde du masque final corrigé et stabilisé
    sitk.WriteImage(final_mask, str(output_path))
    return True, f"Succès (Sauvegardé sous {output_path.name})"


def generate_log_report(output_dir: Path, stats: dict, total_masks: int):
    """Génère un fichier texte récapitulatif de l'exécution du script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"rapport_expansion_{timestamp}.txt"
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("    RAPPORT D'EXPANSION DES MASQUES MAMMAIRES (V6)\n")
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
            
    print(f"\nUn rapport détaillé a été généré : {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Extension des masques via Region Growing (LPS Indestructible V6).")
    
    # Arguments d'arborescence
    parser.add_argument("--ct_dir", type=Path, default=Path("./Base_PETCT"), 
                        help="Dossier contenant les sous-dossiers patients avec les CT initiaux.")
    parser.add_argument("--mask_dir", type=Path, default=Path("./Base_PETCT_BreastMasks"), 
                        help="Dossier contenant les masques TotalSegmentator générés à l'étape précédente.")

    parser.add_argument("--organs_dir", type=Path, default=Path("./Base_PETCT_Organs"), 
                        help="Dossier contenant les boucliers TS (coeur, sternum, côtes).")
  
    parser.add_argument("--output_dir", type=Path, default=Path("./Base_PETCT_BreastMasks_Expanded"), 
                        help="Dossier de sauvegarde des nouveaux masques.")
    
    # Paramètres de l'algorithme
    parser.add_argument("--multiplier", type=float, default=2.5, 
                        help="Tolérance de l'algorithme. Plus c'est haut, plus ça s'étend. (Défaut: 2.5)")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Nombre d'itérations de mise à jour des statistiques. (Défaut: 3)")
    parser.add_argument("--ct-suffix", default="_TDM_", help="Marqueur du fichier CT (Défaut: _TDM_)")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    execution_stats = {"success": [], "skipped": [], "failed": []}
    
    masks = list(args.mask_dir.glob("*_breast_mask.nii.gz"))
    if not masks:
        print(f"Aucun masque trouvé dans {args.mask_dir}")
        return
        
    print(f"{len(masks)} masque(s) trouvé(s). Début de l'expansion avec tolérance à {args.multiplier}...\n")
    
    for mask_path in masks:
        patient_id = mask_path.name.replace("_breast_mask.nii.gz", "")
        print(f"[{patient_id}]")
        
        patient_dir = args.ct_dir / patient_id
        imgs_dir = patient_dir / "imgs"
        
        if not imgs_dir.exists():
            print(f"  [SKIP] Dossier images introuvable.")
            execution_stats["skipped"].append((patient_id, "Dossier 'imgs' introuvable dans le ct_dir"))
            continue
            
        ct_files = list(imgs_dir.glob(f"*{args.ct_suffix}*.nii.gz"))
        if not ct_files:
            print(f"  [SKIP] Aucun fichier CT (*{args.ct_suffix}*.nii.gz) trouvé.")
            execution_stats["skipped"].append((patient_id, "CT scan introuvable"))
            continue
            
        ct_file = ct_files[0]
        output_mask_path = args.output_dir / f"{patient_id}_breast_mask_expanded.nii.gz"

        patient_organs_dir = args.organs_dir / patient_id
        if not patient_organs_dir.exists():
            print(f"  [SKIP] Dossier des boucliers (organes) introuvable.")
            execution_stats["skipped"].append((patient_id, "Dossier organes TS manquant"))
            continue
        
        try:
            success, message = expand_breast_mask(ct_file, mask_path, patient_organs_dir, output_mask_path, args.multiplier, args.iterations)
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
    generate_log_report(args.output_dir, execution_stats, len(masks))

if __name__ == "__main__":
    main()

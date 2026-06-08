#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Script de Création de ROI Mammaire (Soustraction Géométrique) - V6
===============================================================================
Rôle :
  Remplace le "Region Growing" instable par une approche déterministe.
  Dilate le masque initial des seins de manière contrôlée (vers la peau 
  et vers les côtes), puis soustrait un bouclier anatomique massif pour 
  garantir l'exclusion totale du thorax.

Nouveautés de la V6 :
  - Fin du Region Growing basé sur l'intensité.
  - Expansion Frontale Asymétrique : Force le masque à remplir les 40% de 
    vide externe jusqu'à toucher la peau (limite de l'air).
  - Soustraction de Bouclier IA : Utilise les pectoraux, côtes, cartilages 
    et poumons générés par TotalSegmentator.
  - Nettoyage par Composante Connexe : Élimine les îlots parasites.
===============================================================================
"""

import argparse
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi

def get_body_mask(ct_np: np.ndarray) -> np.ndarray:
    """
    Crée un masque plein du corps du patient (exclut l'air ambiant et la table du scanner).
    """
    # 1. Seuillage large (le corps humain est > -500 HU)
    # On abaisse à -700 HU pour être sûr d'attraper la peau fine du téton
    body_thresh = ct_np > -700
    
    # 2. Ignorer la table du scanner en ne gardant que le plus gros bloc connexe
    labeled_body, num_features = ndi.label(body_thresh)
    if num_features == 0:
        return body_thresh
        
    sizes = ndi.sum(body_thresh, labeled_body, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    body_mask = (labeled_body == largest_label)
    
    # 3. Remplissage des trous 2D pour avoir un corps complètement plein (poumons inclus)
    for z in range(body_mask.shape[0]):
        body_mask[z] = ndi.binary_fill_holes(body_mask[z])
        
    return body_mask

def expand_and_sculpt_breast(ct_path: Path, mask_path: Path, organs_dir: Path, output_path: Path):
    """
    Applique la logique de dilatation asymétrique et soustraction géométrique.
    Retourne un tuple : (Statut_booléen, Message_ou_Raison)
    """
    print(f"  [TRAITEMENT] Chargement des images et alignement LPS...")
    
    # ---------------------------------------------------------
    # 1. Chargement et Normalisation LPS (Crucial pour la géométrie)
    # ---------------------------------------------------------
    ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    breast_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    
    ct_img = sitk.DICOMOrient(ct_img, 'LPS')
    breast_img = sitk.DICOMOrient(breast_img, 'LPS')
    
    ct_np = sitk.GetArrayFromImage(ct_img)
    breast_np = sitk.GetArrayFromImage(breast_img).astype(bool)

    if not np.any(breast_np):
        return False, "Le masque initial des seins est complètement vide."

    # ---------------------------------------------------------
    # 2. Le Moule Corporel (Body Mask)
    # ---------------------------------------------------------
    print("  -> Génération de la limite cutanée (Body Mask)...")
    body_mask_np = get_body_mask(ct_np)

    # ---------------------------------------------------------
    # 3. L'Expansion Frontale (Vers le téton/la peau)
    # ---------------------------------------------------------
    print("  -> Expansion frontale asymétrique (remplissage externe)...")
    # En LPS, l'axe Y (index 1 de NumPy) va de l'avant (Ventre=0) vers l'arrière (Dos=Max).
    # On crée une structure asymétrique (40 voxels de long) orientée UNIQUEMENT vers l'avant.
    # Finalement, on passe le rayon à 300 voxels pour s'assurer d'atteindre le bout du sein
    struct_fwd = np.ones((5, 300, 5), dtype=bool)
    # On coupe la moitié arrière (150:) pour interdire la dilatation vers le dos
    struct_fwd[:, 150:, :] = False
    
    breast_fwd = ndi.binary_dilation(breast_np, structure=struct_fwd)
    
    # ---------------------------------------------------------
    # 4. L'Expansion Profonde (Vers le muscle pectoral/côtes)
    # ---------------------------------------------------------
    print("  -> Expansion profonde isotrope...")
    # On dilate le masque d'origine d'environ 15 voxels (1.5 cm) dans toutes les directions.
    # Cela permet d'aller chercher la tumeur collée au muscle.
    # Finalement, on passe à 20 iterations (environ 2 cm) pour être CERTAIN d'englober
    # la tumeur jusqu'au contact strict de la côte/du muscle.
    struct_iso = ndi.generate_binary_structure(3, 1)
    breast_deep = ndi.binary_dilation(breast_np, structure=struct_iso, iterations=20)
    
    # Fusion des deux expansions
    breast_expanded = breast_fwd | breast_deep
    
    # C'est ici qu'on coupe l'excédent : on intersecte avec le Body Mask.
    # Ainsi, l'expansion frontale s'écrase parfaitement contre la limite de la peau.
    breast_expanded = breast_expanded & body_mask_np

    # ---------------------------------------------------------
    # 5. Construction du Super-Bouclier (La Zone Interdite)
    # ---------------------------------------------------------
    print("  -> Construction du bouclier anatomique massif...")
    shield_np = np.zeros_like(ct_np, dtype=bool)
    
    # On liste les mots-clés des organes qui DOIVENT interdire le passage
    shield_keywords = [
        "heart", "sternum", "costal_cartilages", "rib", "vertebrae", 
        "lung", "pectoralis", "clavicula"
    ]
    
    # Chargement dynamique depuis le dossier des organes du patient
    if organs_dir.exists():
        for organ_file in organs_dir.glob("*.nii.gz"):
            if any(kw in organ_file.name for kw in shield_keywords):
                org_img = sitk.ReadImage(str(organ_file))
                org_img = sitk.DICOMOrient(org_img, 'LPS')
                org_np = sitk.GetArrayFromImage(org_img).astype(bool)
                shield_np = shield_np | org_np
    else:
        return False, f"Dossier des boucliers introuvable : {organs_dir.name}"

    # SOLUTION PERTE BOUT DE TUMEUR : On utilise un Closing au lieu d'un Dilation.
    # Le bouclier ne "gonfle" plus vers le sein, il se contente de lier 
    # fermement les côtes, cartilages et pectoraux entre eux.

    # 1. On bouche les fissures entre les côtes et les muscles
    shield_closed = ndi.binary_closing(shield_np, iterations=2)
    
    # 2. NOUVEAU : La Marge de Sécurité Anti-Fuite
    # On dilate le bouclier de 2 voxels (environ 2 mm) vers le sein.
    # Ainsi, la côte et le poumon deviennent "intouchables" avec une zone tampon.
    shield_final = ndi.binary_dilation(shield_closed, iterations=2)

    # ---------------------------------------------------------
    # 6. La Soustraction (Sculpture de la ROI)
    # ---------------------------------------------------------
    print("  -> Soustraction géométrique et nettoyage...")
    # ROI = Expansion Mammaire MINUS Le Bouclier
    final_breast_np = breast_expanded & ~shield_dilated

    # ---------------------------------------------------------
    # 7. Filtrage par Composante Connexe (Nettoyage des déchets)
    # ---------------------------------------------------------
    # La soustraction peut laisser de petits bouts de graisse flottants derrière les côtes.
    # On ne garde que les blocs qui touchent le masque TS d'origine.
    labeled_mask, num_features = ndi.label(final_breast_np)
    
    if num_features > 0:
        # Trouver les labels qui chevauchent le masque initial de TS
        overlapping_labels = np.unique(labeled_mask[breast_np])
        overlapping_labels = overlapping_labels[overlapping_labels != 0] # Exclure le fond (0)
        
        # Reconstruire le masque final uniquement avec ces labels autorisés
        final_cleaned_np = np.isin(labeled_mask, overlapping_labels)
    else:
        final_cleaned_np = final_breast_np

    # ---------------------------------------------------------
    # 8. Restauration et Sauvegarde
    # ---------------------------------------------------------
    final_img = sitk.GetImageFromArray(final_cleaned_np.astype(np.uint8))
    final_img.CopyInformation(breast_img) # Récupère métadonnées d'origine
    
    sitk.WriteImage(final_img, str(output_path))
    return True, f"Succès (ROI sauvegardée : {output_path.name})"


def generate_log_report(output_dir: Path, stats: dict, total_masks: int):
    """Génère un fichier texte récapitulatif de l'exécution du script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"rapport_roi_deterministe_{timestamp}.txt"
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("    RAPPORT DE CRÉATION ROI MAMMAIRE (DÉTERMINISTE V6)\n")
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
    parser = argparse.ArgumentParser(description="Création de ROI mammaire par soustraction géométrique (V6).")
    
    parser.add_argument("--ct_dir", type=Path, default=Path("./Base_PETCT"), 
                        help="Dossier contenant les sous-dossiers patients avec les CT initiaux.")
    parser.add_argument("--mask_dir", type=Path, default=Path("./Base_PETCT_BreastMasks"), 
                        help="Dossier contenant les masques TotalSegmentator générés à l'étape précédente.")
    parser.add_argument("--organs_dir", type=Path, default=Path("./Base_PETCT_Organs"), 
                        help="Dossier racine contenant les sous-dossiers patients avec les organes boucliers.")
    parser.add_argument("--output_dir", type=Path, default=Path("./Base_PETCT_BreastMasks_Expanded"), 
                        help="Dossier de sauvegarde des nouvelles ROI.")
    
    parser.add_argument("--ct-suffix", default="_TDM_", help="Marqueur du fichier CT (Défaut: _TDM_)")
    parser.add_argument("--overwrite", action="store_true", help="Écraser les masques existants")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    execution_stats = {"success": [], "skipped": [], "failed": []}
    
    # On se base sur le masque unifié généré par la tâche 'breasts' de TS
    masks = list(args.mask_dir.glob("*_breast*.nii.gz"))
    if not masks:
        print(f"Aucun masque trouvé dans {args.mask_dir}")
        return
        
    print(f"{len(masks)} masque(s) trouvé(s). Début du traitement déterministe...\n")
    
    for mask_path in masks:
        # Nettoyage pour récupérer l'ID (ex: PATIENT_01_breast.nii.gz -> PATIENT_01)
        # S'adapte au nom exact de sortie que TotalSegmentator a donné.
        patient_id = mask_path.name.split("_breast")[0]
        print(f"[{patient_id}]")
        
        patient_ct_dir = args.ct_dir / patient_id
        patient_organs_dir = args.organs_dir / patient_id
        imgs_dir = patient_ct_dir / "imgs"
        
        output_mask_path = args.output_dir / f"{patient_id}_breast_roi_V6.nii.gz"
        
        if output_mask_path.exists() and not args.overwrite:
            print(f"  [SKIP] ROI déjà existante.")
            execution_stats["skipped"].append((patient_id, "ROI existante"))
            continue
        
        if not imgs_dir.exists():
            print(f"  [SKIP] Dossier images introuvable.")
            execution_stats["skipped"].append((patient_id, "Dossier 'imgs' introuvable"))
            continue
            
        ct_files = list(imgs_dir.glob(f"*{args.ct_suffix}*.nii.gz"))
        if not ct_files:
            print(f"  [SKIP] CT scan introuvable.")
            execution_stats["skipped"].append((patient_id, "CT scan introuvable"))
            continue
            
        ct_file = ct_files[0]
        
        try:
            success, message = expand_and_sculpt_breast(ct_file, mask_path, patient_organs_dir, output_mask_path)
            if success:
                print(f"  [OK] {message}")
                execution_stats["success"].append(patient_id)
            else:
                print(f"  [SKIP] {message}")
                execution_stats["skipped"].append((patient_id, message))
        except Exception as e:
            print(f"  [ÉCHEC] Erreur inattendue : {e}")
            execution_stats["failed"].append((patient_id, str(e)))

    print("\n=== CRÉATION DES ROI TERMINÉE ===")
    generate_log_report(args.output_dir, execution_stats, len(masks))

if __name__ == "__main__":
    main()

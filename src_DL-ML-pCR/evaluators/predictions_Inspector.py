#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nom du Script : check_segmentations.py
Description   : Outil de contrôle qualité (QC) pour l'analyse des masques de segmentation 3D.
                Détecte la multifocalité (plusieurs tumeurs), les cas à 0 tumeur,
                et extrait des métriques oncologiques clés (volume, diamètre max de Feret).

Utilisation :
    python check_segmentations.py -i /chemin/vers/masques -o /chemin/vers/rapport.txt
"""

import os
import sys
import argparse
from datetime import datetime
import SimpleITK as sitk
import numpy as np


def analyze_segmentation(file_path):
    """
    Analyse un fichier de segmentation 3D avec SimpleITK.
    Identifie le nombre de tumeurs isolées et extrait des métriques cliniques.
    
    Parameters:
        file_path (str): Chemin vers le fichier image (NIfTI, mhd, etc.)
        
    Returns:
        dict: Statistiques de l'analyse du cas.
    """
    results = {
        "filename": os.path.basename(file_path),
        "status": "OK",
        "num_tumors": 0,
        "total_volume_cm3": 0.0,
        "tumors_details": []
    }
    
    try:
        # 1. Chargement de l'image de segmentation
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        seg_img = reader.Execute()
        
        # S'assurer que le masque est binaire (0 = fond, 1 ou + = tumeur)
        # Idéal pour nnU-Net qui peut sortir des labels multiclasses ou un seul label binaire
        binary_seg = sitk.Cast(seg_img > 0, sitk.sitkUInt8)
        
        # 2. Analyse des composantes connexes (Connected Components)
        # Regroupe les voxels connectés en "objets" (tumeurs distinctes)
        # Par défaut, utilise la connectivité complète (26-connectivité en 3D)
        connected_components = sitk.ConnectedComponent(binary_seg)
        
        # 3. Extraction des caractéristiques de forme et de géométrie
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(connected_components)
        
        # Obtenir la liste des labels (chaque label correspond à une tumeur distincte)
        labels = shape_stats.GetLabels()
        num_tumors = len(labels)
        results["num_tumors"] = num_tumors
        
        if num_tumors == 0:
            results["status"] = "WARNING_ZERO_TUMOR"
            return results
        elif num_tumors > 1:
            results["status"] = "WARNING_MULTIFOCAL"
            
        # 4. Extraction des métriques pour chaque lésion détectée
        total_volume_mm3 = 0.0
        
        for idx, label in enumerate(labels, start=1):
            # Volume en mm3 = nombre de voxels * produit du spacing (taille des voxels)
            volume_mm3 = shape_stats.GetPhysicalSize(label)
            volume_cm3 = volume_mm3 / 1000.0  # Plus parlant pour un oncologue
            total_volume_mm3 += volume_mm3
            
            # Diamètre maximal de Feret (plus grande distance entre deux points de la lésion)
            feret_diam = shape_stats.GetFeretDiameter(label)
            
            results["tumors_details"].append({
                "id": idx,
                "volume_cm3": volume_cm3,
                "feret_diameter_mm": feret_diam
            })
            
        results["total_volume_cm3"] = total_volume_mm3 / 1000.0
        
    except Exception as e:
        results["status"] = "ERROR_READING"
        results["error_msg"] = str(e)
        
    return results


def main():
    # Configuration du Parser d'arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Script de QC de masques de segmentation pour la détection de multifocalité tumorale."
    )
    parser.add_argument("-i", "--input_dir", required=True, type=str,
                        help="Dossier contenant les fichiers de segmentation (ex: .nii.gz)")
    parser.add_argument("-o", "--output_file", required=False, type=str, default="qc_report.txt",
                        help="Nom/Chemin du fichier texte de rapport généré (Défaut: qc_report.txt)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"[-] Erreur : Le dossier d'entrée '{args.input_dir}' n'existe pas.")
        sys.exit(1)
        
    # Liste des extensions courantes en imagerie médicale
    valid_extensions = ('.nii.gz', '.nii', '.mhd', '.nrrd')
    files_to_process = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(valid_extensions)]
    
    if not files_to_process:
        print(f"[-] Aucun fichier avec les extensions {valid_extensions} trouvé dans {args.input_dir}")
        sys.exit(0)
        
    print(f"[+] Début de l'analyse de {len(files_to_process)} fichiers...")
    
    # Stockage des rapports
    all_reports = []
    summary = {
        "total_cases": len(files_to_process),
        "ok_cases": 0,
        "multifocal_cases": 0,
        "zero_tumor_cases": 0,
        "error_cases": 0
    }
    
    # Boucle de traitement
    for file_path in sorted(files_to_process):
        res = analyze_segmentation(file_path)
        all_reports.append(res)
        
        # Mise à jour du résumé global
        if res["status"] == "OK":
            summary["ok_cases"] += 1
        elif res["status"] == "WARNING_MULTIFOCAL":
            summary["multifocal_cases"] += 1
        elif res["status"] == "WARNING_ZERO_TUMOR":
            summary["zero_tumor_cases"] += 1
        elif res["status"] == "ERROR_READING":
            summary["error_cases"] += 1

    # --- GÉNÉRATION DU RAPPORT EXHAUSTIF (Console & Fichier TXT) ---
    lines = []
    lines.append("==============================================================================")
    lines.append(f"RAPPORT AUTOMATIQUE DE CONTRÔLE QUALITÉ & MULTIFOCALITÉ - ONCOLOGIE")
    lines.append(f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dossier analysé : {os.path.abspath(args.input_dir)}")
    lines.append("==============================================================================\n")
    
    lines.append("------------------------------------------------------------------------------")
    lines.append("DETAIL PAR CAS :")
    lines.append("------------------------------------------------------------------------------")
    
    for r in all_reports:
        lines.append(f"\n[Fichier] : {r['filename']}")
        
        if r["status"] == "ERROR_READING":
            lines.append(f"  ⚠️ STATUT : ERREUR DE LECTURE")
            lines.append(f"  Message   : {r['error_msg']}")
            
        elif r["status"] == "WARNING_ZERO_TUMOR":
            lines.append(f"  ⚠️ STATUT : WARNING - AUCUNE TUMEUR DÉTECTÉE")
            lines.append(f"  [Info Oncologue] : Le modèle n'a trouvé aucune cible. À vérifier si discordant avec la clinique.")
            
        elif r["status"] == "WARNING_MULTIFOCAL":
            lines.append(f"  🚨 STATUT : WARNING - MULTIFOCALITÉ DÉTECTÉE ({r['num_tumors']} foyers distincts)")
            lines.append(f"  Charge tumorale totale : {r['total_volume_cm3']:.3f} cm³")
            for t in r["tumors_details"]:
                lines.append(f"    -> Lésion #{t['id']} | Volume: {t['volume_cm3']:.3f} cm³ | Diamètre de Feret Max: {t['feret_diameter_mm']:.2f} mm")
            lines.append("  [Action Requise] : Spécialiste requis pour identifier la lésion primaire/dominante.")
            
        else:
            lines.append(f"  ✅ STATUT : OK (Lésion unique)")
            t = r["tumors_details"][0]
            lines.append(f"    -> Volume Tumeur : {t['volume_cm3']:.3f} cm³")
            lines.append(f"    -> Diamètre de Feret Max : {t['feret_diameter_mm']:.2f} mm")
            
        lines.append("-" * 40)
        
    # Section Résumé Global
    lines.append("\n" + "=" * 78)
    lines.append("RÉSUMÉ STATISTIQUE GLOBAL")
    lines.append("=" * 78)
    lines.append(f"Nombre total de cas traités             : {summary['total_cases']}")
    lines.append(f"Cas standard (1 seule tumeur)           : {summary['ok_cases']} ({summary['ok_cases']/summary['total_cases']*100:.1f}%)")
    lines.append(f"Cas multifocaux (Tumeurs multiples)     : {summary['multifocal_cases']} ({summary['multifocal_cases']/summary['total_cases']*100:.1f}%) 🚨")
    lines.append(f"Cas sans aucune tumeur détectée         : {summary['zero_tumor_cases']} ({summary['zero_tumor_cases']/summary['total_cases']*100:.1f}%) ⚠️")
    lines.append(f"Cas en erreur système                   : {summary['error_cases']} ({summary['error_cases']/summary['total_cases']*100:.1f}%)")
    lines.append("==============================================================================")
    
    # 1. Écriture dans la console
    full_report_text = "\n".join(lines)
    print(full_report_text)
    
    # 2. Écriture dans le fichier .txt
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(full_report_text)
        print(f"\n[+] Rapport exporté avec succès dans : {os.path.abspath(args.output_file)}")
    except Exception as e:
        print(f"\n[-] Erreur lors de l'écriture du fichier de rapport : {e}")


if __name__ == "__main__":
    main()

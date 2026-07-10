#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Évaluateur de Métriques de Segmentation & Multifocalité (V2 - Strict NII.GZ)
===============================================================================
Rôle :
  Analyse les masques de segmentation 3D, filtre les artefacts millimétriques,
  et calcule le volume et le diamètre réel de Feret en s'assurant d'ignorer
  les fichiers doublons (.nrrd, .seg).
===============================================================================

Description   : Outil de contrôle qualité (QC) pour l'analyse des masques de segmentation 3D.
                Détecte la multifocalité (plusieurs tumeurs), les cas à 0 tumeur,
                et extrait des métriques oncologiques clés (volume, diamètre max de Feret).

Utilisation :
    python predictions_Inspector.py -i /chemin/vers/masques -o /chemin/vers/rapport.txt
"""

import os
import sys
import argparse
from datetime import datetime
import SimpleITK as sitk
import numpy as np


def analyze_segmentation(file_path, min_vol_cm3=0.005):
    """
    Analyse un masque de segmentation 3D à l'aide de SimpleITK.
    Identifie les régions tumorales isolées (composantes connexes), élimine le bruit 
    de fond (artefacts millimétriques) et extrait des métriques cliniques (volume, Feret).
    
    Parameters:
        file_path (str): Chemin absolu ou relatif vers le fichier de segmentation (.nii.gz).
        min_vol_cm3 (float): Seuil de volume minimal en cm³ en dessous duquel une lésion 
                             est considérée comme du bruit (Défaut: 0.005 cm³).
        
    Returns:
        dict: Dictionnaire contenant le statut du cas, le nombre de tumeurs réelles,
              la charge tumorale totale et le détail géométrique de chaque foyer.
    """
    # Initialisation du dictionnaire de résultats pour ce fichier
    results = {
        "filename": os.path.basename(file_path),
        "status": "OK",
        "num_tumors": 0,
        "total_volume_cm3": 0.0,
        "tumors_details": []
    }
    
    try:
        # 1. Chargement de l'image via SimpleITK
        seg_img = sitk.ReadImage(file_path)
        
        # Binarisation stricte : tout voxel > 0 devient 1 (tumeur), le reste devient 0 (fond).
        # Cast en UInt8 (entier 8 bits) requis pour les filtres de composantes connexes.
        binary_seg = sitk.Cast(seg_img > 0, sitk.sitkUInt8)
        
        # 2. Extraction des Composantes Connexes (Connected Components)
        # Ce filtre analyse la topologie 3D et attribue un identifiant unique (1, 2, 3...) 
        # à chaque groupe de voxels qui se touchent (26-connectivité par défaut en 3D).
        connected_components = sitk.ConnectedComponent(binary_seg)
        
        # 3. Initialisation de l'analyseur de forme géométrique
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        
        # CRITIQUE : Force SimpleITK à calculer la coque convexe ("Convex Hull") des formes.
        # Sans cette activation explicite, GetFeretDiameter() renvoie structurellement 0.0.
        shape_stats.SetComputeFeretDiameter(True)
        
        # Exécution du calcul des statistiques sur notre image labellisée
        shape_stats.Execute(connected_components)
        
        # Récupération de la liste de tous les labels générés (ex: [1, 2, 3, ..., 29])
        labels = shape_stats.GetLabels()
        
        valid_tumor_idx = 1
        total_volume_mm3 = 0.0
        
        # 4. Boucle d'analyse et de filtrage de chaque objet détecté
        for label in labels:
            # Récupération du volume physique de la composante (basé sur le spacing des voxels)
            volume_mm3 = shape_stats.GetPhysicalSize(label)
            volume_cm3 = volume_mm3 / 1000.0  # Conversion en cm³ (unité standard en oncologie)
            
            # --- FILTRE ANTI-BRUIT CLINIQUE ---
            # Si le volume de la lésion est inférieur au seuil, on l'exclut du rapport.
            # Cela évite de lever de fausses alertes pour des voxels isolés (artefacts de prédiction).
            if volume_cm3 < min_vol_cm3:
                continue
            
            # Récupération du diamètre maximal de Feret (plus grande distance Euclidienne entre 2 points)
            feret_diam = shape_stats.GetFeretDiameter(label)
            
            # Accumulation du volume pour la charge tumorale totale du patient
            total_volume_mm3 += volume_mm3
            
            # Ajout des détails de cette lésion valide
            results["tumors_details"].append({
                "id": valid_tumor_idx,
                "volume_cm3": volume_cm3,
                "feret_diameter_mm": feret_diam
            })
            valid_tumor_idx += 1  # Incrémentation de l'index des lésions valides
            
        # 5. Synthèse des résultats pour le fichier en cours
        num_tumors = len(results["tumors_details"])
        results["num_tumors"] = num_tumors
        results["total_volume_cm3"] = total_volume_mm3 / 1000.0
        
        # Détermination du statut de sécurité (Warning)
        if num_tumors == 0:
            results["status"] = "WARNING_ZERO_TUMOR"
        elif num_tumors > 1:
            results["status"] = "WARNING_MULTIFOCAL"
            
    except Exception as e:
        # En cas de fichier corrompu ou d'erreur SimpleITK imprévue
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
    parser.add_argument("--min_vol", type=float, default=0.005, help="Volume seuil minimal en cm3 pour éliminer le bruit.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"[-] Erreur : Le dossier d'entrée '{args.input_dir}' n'existe pas.")
        sys.exit(1)
        
    # Liste des extensions courantes en imagerie médicale
    valid_extensions = ('.nii.gz')
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
        res = analyze_segmentation(file_path, min_vol_cm3=args.min_vol)
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

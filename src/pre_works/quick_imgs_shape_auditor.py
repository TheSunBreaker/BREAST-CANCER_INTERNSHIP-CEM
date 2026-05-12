#!/usr/bin/env python3
"""
Script d'audit des dimensions spatiales des fichiers NIfTI générés.
"""

import os
import glob
import SimpleITK as sitk

def audit_nifti_shapes(root_dir: str):
    print(f"--- ANALYSE DES DIMENSIONS DANS : {root_dir} ---\n")
    
    all_nifti_files = glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)
    
    if not all_nifti_files:
        print("Aucun fichier .nii.gz trouvé.")
        return

    stats = {
        "3D": 0,
        "4D": 0,
        "anomalies": 0
    }

    for f in all_nifti_files:
        try:
            # On utilise le FileReader sans charger l'image en RAM (très rapide)
            reader = sitk.ImageFileReader()
            reader.SetFileName(f)
            reader.ReadImageInformation()
            size = reader.GetSize()
            
            # Formater le chemin pour un affichage lisible
            patient_id = os.path.basename(os.path.dirname(os.path.dirname(f)))
            filename = os.path.basename(f)
            
            # Analyse de la shape
            if len(size) == 3:
                stats["3D"] += 1
                # On n'affiche pas tous les 3D pour ne pas polluer, sauf si c'est bizarre
                if size[2] <= 2:
                     print(f"[ALERTE 2D] {patient_id} / {filename} -> Shape: {size} (Volume plat ?)")
                     stats["anomalies"] += 1
            
            elif len(size) == 4:
                stats["4D"] += 1
                print(f"[FICHIER 4D TROUVÉ] {patient_id} / {filename} -> Shape: {size} (Z={size[2]}, T={size[3]})")
                
            else:
                stats["anomalies"] += 1
                print(f"[SHAPE ANORMALE] {patient_id} / {filename} -> Shape: {size}")

        except Exception as e:
            print(f"[ERREUR LECTURE] {f} : {e}")

    print("\n--- RÉSUMÉ DE L'AUDIT ---")
    print(f"Fichiers analysés : {len(all_nifti_files)}")
    print(f"Volumes 3D normaux : {stats['3D']}")
    print(f"Volumes 4D (à splitter potentiellement) : {stats['4D']}")
    print(f"Anomalies : {stats['anomalies']}")

if __name__ == "__main__":
    # Remplacer par le chemin de l'ancien Base_IRM si nécessaire
    DOSSIER_A_AUDITER = "./Base_IRM" 
    audit_nifti_shapes(DOSSIER_A_AUDITER)

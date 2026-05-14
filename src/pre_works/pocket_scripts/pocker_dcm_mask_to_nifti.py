#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  Convertisseur Rapide de Masques Manuels (Standalone)
===============================================================================
Rôle : 
  Convertit tous les fichiers DICOM (SEG ou RTSTRUCT) présents dans le 
  MÊME répertoire que ce script en fichiers NIfTI (.nii.gz).

Utilisation :
  1. Copier/déplacer ce script dans le dossier contenant les DICOM à garder.
  2. Supprimer ou mettre ailleurs les DICOM non voulus.
  3. Lancer le script.
  (Optionnel : si vous  RTSTRUCT, copier temporairement le NIfTI 
  de l'image (ex: _T1.nii.gz ou _TEP.nii.gz) dans ce même dossier pour servir 
  de référence à Plastimatch).
===============================================================================
"""

import os
import glob
import subprocess
import argparse
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk

# --- Chemin Plastimatch (À adapter si besoin) ---
PLASTIMATCH_EXE = r"C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe"

def find_reference_nifti(current_dir: str) -> str:
    """
    Cherche un NIfTI de référence pour rastériser les RTSTRUCT, 
    en respectant strictement la logique temporelle de l'Ingesteur V6.
    """
    # 1. Dans le dossier courant (Au cas où tu l'as copié manuellement)
    niftis = glob.glob(os.path.join(current_dir, "*.nii.gz"))
    niftis = [n for n in niftis if "mask" not in n.lower()]
    if niftis: return niftis[0]

    # 2. Déduction intelligente via l'arborescence
    # current_dir est censé être ".../dicom_mask_rm_20240101/a_verifier"
    parent_dir = os.path.dirname(current_dir) # ex: dicom_mask_rm_20240101
    patient_dir = os.path.dirname(parent_dir) # ex: DUKE_001
    
    parent_name = os.path.basename(parent_dir)
    
    # On vérifie qu'on est bien dans la bonne structure
    if parent_name.startswith("dicom_mask_"):
        # On extrait le suffixe temporel (ex: "_20240101" ou "" pour la Baseline)
        # En supprimant "dicom_mask_rm" ou "dicom_mask_pet"
        suffixe = parent_name.replace("dicom_mask_rm", "").replace("dicom_mask_pet", "").replace("dicom_mask_orphelins", "")
        
        # On reconstruit le nom du dossier image cible
        target_imgs_folder = f"imgs{suffixe}"
        target_imgs_dir = os.path.join(patient_dir, target_imgs_folder)
        
        if os.path.exists(target_imgs_dir):
            target_niftis = glob.glob(os.path.join(target_imgs_dir, "*.nii.gz"))
            if target_niftis:
                # Tous les NIfTI d'un même dossier visite partagent la même géométrie spatiale.
                # Prendre le premier (ex: la phase 0 du DCE, ou le PET) est mathématiquement parfait.
                return target_niftis[0]

    return None

def convert_single_mask(dcm_path: str, ref_nifti: str = None) -> bool:
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
        modality = getattr(ds, "Modality", "")
        out_name = os.path.splitext(os.path.basename(dcm_path))[0] + "_manuel.nii.gz"
        out_path = os.path.join(os.path.dirname(dcm_path), out_name)

        # --- CAS 1 : DICOM SEG ---
        if modality == "SEG":
            reader = pydicom_seg.MultiClassReader()
            result = reader.read(pydicom.dcmread(dcm_path))
            arr = sitk.GetArrayFromImage(result.image)
            
            clean_arr = np.zeros_like(arr, dtype=np.uint8)
            for seg_val, seg_info in result.segment_infos.items():
                label = str(getattr(seg_info, "SegmentLabel", "")).upper()
                desc  = str(getattr(seg_info, "SegmentDescription", "")).upper()
                if "BACKGROUND" not in label and "BACKGROUND" not in desc:
                    clean_arr[arr == seg_val] = 1 # Valeur 1 par défaut pour le masque
            
            clean_img = sitk.GetImageFromArray(clean_arr)
            clean_img.CopyInformation(result.image)
            sitk.WriteImage(clean_img, out_path)
            print(f"[SUCCÈS] SEG converti : {out_name}")
            return True

        # --- CAS 2 : RTSTRUCT ---
        elif modality == "RTSTRUCT":
            if not ref_nifti:
                print(f"[ERREUR] RTSTRUCT ignoré ({os.path.basename(dcm_path)}).")
                print(" -> Aucune image NIfTI de référence trouvée pour la rastérisation Plastimatch.")
                print(" -> Solution : Copiez le NIfTI de l'image (ex: _T1.nii.gz) dans ce dossier.")
                return False

            commande = [
                PLASTIMATCH_EXE, "convert",
                "--input", dcm_path,
                "--fixed", ref_nifti,
                "--output-img", out_path,
                "--output-type", "uint8"
            ]
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"[SUCCÈS] RTSTRUCT converti : {out_name} (Aligné sur {os.path.basename(ref_nifti)})")
            return True

        else:
            return False # Fichier DICOM non masque

    except Exception as e:
        print(f"[ÉCHEC] {os.path.basename(dcm_path)} : {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convertit les DICOMs d'un dossier en NIfTI.")
    parser.add_argument("--dir", default=".", help="Dossier cible (Défaut : dossier courant)")
    parser.add_argument("--ref", default=None, help="Chemin forcé vers un NIfTI de référence (Optionnel)")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    print(f"\n=== SCAN DU DOSSIER : {target_dir} ===\n")

    # Liste tous les fichiers (on testera s'ils sont DICOM à l'intérieur)
    fichiers = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    
    # Exclure le script lui-même et les éventuels .nii.gz existants
    candidats = [f for f in fichiers if not f.endswith(".py") and not f.endswith(".nii.gz") and not f.endswith(".txt")]

    if not candidats:
        print("Aucun fichier candidat trouvé à convertir.")
        return

    # Recherche auto de la référence si non fournie
    ref_nifti = args.ref if args.ref else find_reference_nifti(target_dir)
    
    if ref_nifti:
        print(f"Image de référence détectée pour les RTSTRUCT : {os.path.basename(ref_nifti)}\n")
    else:
        print("Aucune image de référence détectée. (Les RTSTRUCT échoueront, les SEG fonctionneront).\n")

    convertis = 0
    for f in candidats:
        # Pydicom permet de vérifier rapidement si c'est un DICOM valide
        try:
            pydicom.dcmread(f, stop_before_pixels=True, force=False) # force=False pour ignorer les non-dicoms
        except:
            continue # Pas un dicom valide
            
        if convert_single_mask(f, ref_nifti):
            convertis += 1

    print(f"\n=== TERMINÉ : {convertis} masque(s) généré(s) ===")

if __name__ == "__main__":
    main()

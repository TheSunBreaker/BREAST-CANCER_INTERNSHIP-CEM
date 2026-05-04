#!/usr/bin/env python3
"""
Script d'ingestion DICOM robuste.
Groupe les fichiers par SeriesInstanceUID pour gérer les dossiers "fourre-tout" des hôpitaux,
extrait les métadonnées de chaque série, filtre (T1/DCE, CT, PET) et convertit en NIfTI.
"""

# IMPORTANT : Ce code ne gère pas la conversion en SUV val, ilf audra donc le faire à part

import os
import shutil
import pydicom
import SimpleITK as sitk
from collections import defaultdict

import tempfile
import subprocess

def convert_files_to_nifti_plastimatch(file_paths: list, output_path: str) -> bool:
    """
    Convertit une liste de fichiers DICOM en NIfTI en utilisant Plastimatch.
    Isole les fichiers dans un dossier temporaire pour garantir que Plastimatch 
    ne lise que la série exacte identifiée par notre routage (SeriesInstanceUID).
    """
    if not file_paths:
        return False
        
    # Création d'un dossier temporaire qui s'auto-détruira à la fin du bloc 'with'
    with tempfile.TemporaryDirectory() as tmp_dicom_dir:
        # 1. Copie des fichiers de la série dans le dossier isolé
        for f in file_paths:
            shutil.copy2(f, tmp_dicom_dir)
            
        # 2. Appel de Plastimatch
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # On fait une conversion pure (--output-type float). 
            # On ne fait PAS de recalage ici, notre spatial_standardizer le fera plus tard.
            commande = [
                "plastimatch", "convert",
                "--input", tmp_dicom_dir,
                "--output-img", output_path,
                "--output-type", "float"
            ]
            
            # stdout=subprocess.DEVNULL rend Plastimatch silencieux pour ne pas polluer la console
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   [ERREUR] Plastimatch a échoué lors de la conversion : {e}")
            return False
        except FileNotFoundError:
            print("   [ERREUR FATALE] L'outil 'plastimatch' n'est pas installé ou n'est pas dans le PATH système.")
            return False

def scan_and_group_dicoms(root_dir: str) -> dict:
    """
    Parcourt récursivement un dossier et regroupe tous les fichiers DICOM valides
    en fonction de leur SeriesInstanceUID.
    Retourne un dictionnaire : { 'SeriesInstanceUID': [liste_des_chemins_fichiers] }
    """
    print("--- 1. SCAN ET REGROUPEMENT DES SÉRIES DICOM ---")
    series_dict = defaultdict(list)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Lecture rapide de l'en-tête (sans charger les lourds pixels en mémoire)
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Le SeriesInstanceUID est l'identifiant strict d'une séquence
                if hasattr(ds, 'SeriesInstanceUID'):
                    series_dict[ds.SeriesInstanceUID].append(file_path)
            except Exception:
                # Ce n'est pas un DICOM valide (ex: un .txt ou un fichier caché OS)
                continue
                
    print(f" -> {len(series_dict)} séries uniques trouvées dans l'arborescence.\n")
    return series_dict

def get_series_metadata(file_paths: list) -> dict:
    """
    Extrait les métadonnées cliniques à partir du premier fichier d'une série.
    """
    if not file_paths:
        return {}
        
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True)
        return {
            "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
            "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")).upper(),
            "SeriesTime": str(getattr(ds, "SeriesTime", "000000")),
        }
    except Exception:
        return {}

def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, dict_anonymisation: dict = None):
    
    # 1. On scanne tout et on regroupe par série, peu importe l'organisation des dossiers
    series_groups = scan_and_group_dicoms(raw_data_root)
    mri_phases_by_patient = defaultdict(list)
    
    print("--- 2. ANALYSE ET ROUTAGE DES SÉRIES ---")
    
    for series_uid, file_paths in series_groups.items():
        meta = get_series_metadata(file_paths)
        if not meta:
            continue
            
        vrai_id = meta["PatientID"]
        modality = meta["Modality"]
        description = meta["SeriesDescription"]
        
        # Anonymisation
        patient_id = dict_anonymisation.get(vrai_id, vrai_id) if dict_anonymisation else vrai_id
        
        # --- ROUTAGE PET / CT ---
        if modality in ["PT", "CT"]:
            print(f"[{modality}] {description} (Patient: {patient_id})")
            
            if modality == "PT":
                # 1. Copie pour le calcul SUV
                tep_dicom_dir = os.path.join(out_petct_root, patient_id, "TEP", series_uid)
                os.makedirs(tep_dicom_dir, exist_ok=True)
                for f in file_paths:
                    shutil.copy2(f, tep_dicom_dir)
                print(f" -> Copie des {len(file_paths)} fichiers DICOM PET effectuée.")
                
                # 2. Génération du PET Brut (en Becquerels) via Plastimatch
                imgs_dir = os.path.join(out_petct_root, patient_id, "imgs")
                out_raw_pet_path = os.path.join(imgs_dir, f"{patient_id}_TEP_RAW.nii.gz")
                print(f" -> Conversion PET Brut via Plastimatch vers : {out_raw_pet_path}")
                convert_files_to_nifti_plastimatch(file_paths, out_raw_pet_path)
                
            else: # CT
                imgs_dir = os.path.join(out_petct_root, patient_id, "imgs")
                out_path = os.path.join(imgs_dir, f"{patient_id}_TDM.nii.gz")
                print(f" -> Conversion CT via Plastimatch vers : {out_path}")
                convert_files_to_nifti_plastimatch(file_paths, out_path)
                
        # --- ROUTAGE IRM (Filtre strict) ---
        elif modality == "MR":
            if "T1" in description or "DCE" in description:
                print(f"[IRM Validée] {description} (Patient: {patient_id})")
                mri_phases_by_patient[patient_id].append({
                    "files": file_paths,
                    "time": meta["SeriesTime"],
                    "desc": description
                })
            else:
                print(f"[IRM Secondaire Archivée] {description} (Patient: {patient_id})")
                
                # On nettoie la description pour en faire un nom de fichier valide (pas d'espaces ou de /)
                clean_desc = "".join(c if c.isalnum() else "_" for c in description)
                # On enlève les underscores multiples pour faire plus propre
                clean_desc = "_".join(filter(None, clean_desc.split("_")))
                
                # On crée le dossier d'archivage
                autres_dir = os.path.join(out_mri_root, patient_id, "autres_irm")
                os.makedirs(autres_dir, exist_ok=True)
                
                # Le fichier s'appellera par exemple : DUKE_001_T2_TSE_AXIAL.nii.gz
                out_path = os.path.join(autres_dir, f"{patient_id}_{clean_desc}.nii.gz")
                
                print(f" -> Conversion IRM Secondaire via Plastimatch vers : {out_path}")
                convert_files_to_nifti_plastimatch(file_paths, out_path)

    # --- 3. TRAITEMENT ET TRI TEMPOREL DES PHASES IRM ---
    print("\n--- 3. CONVERSION ET TRI CHRONOLOGIQUE DES PHASES IRM ---")
    for patient_id, phases in mri_phases_by_patient.items():
        # Tri chronologique basé sur le SeriesTime du DICOM
        phases_triees = sorted(phases, key=lambda x: x["time"])
        imgs_dir = os.path.join(out_mri_root, patient_id, "imgs")
        
        for index, phase in enumerate(phases_triees):
            out_path = os.path.join(imgs_dir, f"{patient_id}_{index:04d}.nii.gz")
            print(f" -> IRM Phase {index} ({phase['desc']}) via Plastimatch vers : {out_path}")
            convert_files_to_nifti_plastimatch(phase["files"], out_path)

    print("\n=== INGÉSTION TERMINÉE AVEC SUCCÈS ===")

if __name__ == "__main__":
    DOSSIER_DICOM_VRAC = "./data_hopital_brut"
    PROJET_IRM_RACINE = "./Base_IRM"
    PROJET_PETCT_RACINE = "./Base_PETCT"
    
    CORRESPONDANCES = {
        "JEAN_DUPONT_849": "DUKE_001",
        "MARIE_CURIE_112": "DUKE_002"
    }

    ingest_raw_dicoms(DOSSIER_DICOM_VRAC, PROJET_IRM_RACINE, PROJET_PETCT_RACINE, CORRESPONDANCES)

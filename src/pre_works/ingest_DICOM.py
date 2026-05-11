#!/usr/bin/env python3
r"""
Script d'ingestion DICOM robuste.
Groupe les fichiers par SeriesInstanceUID pour gérer les dossiers "fourre-tout" des hôpitaux,
extrait les métadonnées de chaque série, filtre (T1/DCE, CT, PET) et convertit en NIfTI.


ATTENTION, IL EST NECESSAIRE, POUR UTILISER CE SCRIPT, D'INSTALLER PLASTIMATCH SUR SA MACHINE VIA LE LIEN https://sourceforge.net/projects/plastimatch/postdownload POUR WINDOWS, PUIS
D'EXTRAIRE LES FICHIERS AVEC UNE COMMANDE DU STYLE 'msiexec /a "C:\Users\coul0426\Downloads\Plastimatch-1.9.4-win64.msi" /qb TARGETDIR="C:\Users\coul0426\plastimatch_portable"'. Le binaire sera alors à 
'C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe'


LE ROI DE DICOM TO NII COTE PET ET CT C'EST PLASTIMATCH. PAR CONTRE, POUR LES IRMS, PLASTIMATCH A TENDANCE A ECHOUER. SURTOUT POUR DES IRMS EXOTIQUES. COMME UN DCE 4D. EXACTEMENT
 CE QUI NOUS INTERESSE. DONC, ON A RECOURT AU ROI DE L'IRM dcm2niix. QUAND CE SERA IRM, ON AURA RECOURT A LUI. IL FAUT DONC TELECHARGER LE ZIP SUR LE GIT "https://github.com/rordenlab/dcm2niix/releases",
 ET EXTRAIRE POUR AVOIR LE ".exe". DANS MON CAS, JE L'AI MIT AU "C:\Users\coul0426\dcm2niix_portable\dcm2niix.exe". 
"""

# IMPORTANT : Ce code ne gère pas la conversion en SUV val, il faudra donc le faire à part

import os
import shutil
import pydicom
import SimpleITK as sitk
from collections import defaultdict
from tqdm import tqdm # Pour avoir une barre de progression (Le processus pouvant être asez long, c'est toujours bon d'avoir des indices visuels de progression)

import tempfile
import subprocess

# Chemin vers l'exécutable Plastimatch portable
# Le "r" devant la chaîne est crucial sous Windows pour éviter que les "\" 
# ne soient interprétés comme des caractères d'échappement.
PLASTIMATCH_EXE = r"C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe"
# --- NOUVELLE ARME POUR L'IRM ---
DCM2NIIX_EXE = r"C:\Users\coul0426\dcm2niix_portable\dcm2niix.exe"

def convert_files_to_nifti_dcm2niix(file_paths: list, output_dir: str, file_prefix: str) -> bool:
    """
    Convertit une liste de fichiers DICOM IRM en NIfTI en utilisant dcm2niix.
    Gère intelligemment les séquences 4D (DWI, DCE, T1-MAP) sans crasher.
    """
    if not file_paths:
        return False
        
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Sous-dossiers isolés pour l'input et l'output
        dicom_dir = os.path.join(tmp_dir, "dicoms")
        nifti_dir = os.path.join(tmp_dir, "nifti")
        os.makedirs(dicom_dir)
        os.makedirs(nifti_dir)
        
        for f in file_paths:
            shutil.copy2(f, dicom_dir)
            
        try:
            # Commande dcm2niix :
            # -z y : Compresse en .nii.gz
            # -f : Format du nom de fichier de sortie
            # -o : Dossier de destination
            commande = [
                DCM2NIIX_EXE,
                "-z", "y",
                "-f", file_prefix,
                "-o", nifti_dir,
                dicom_dir
            ]
            
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # dcm2niix a parfois la bonne idée de générer plusieurs fichiers pour une même séquence
            # (par exemple : file_prefix_e1.nii.gz, file_prefix_e2.nii.gz pour les échos multiples)
            generated_files = glob.glob(os.path.join(nifti_dir, "*.nii.gz"))
            if not generated_files:
                return False
                
            os.makedirs(output_dir, exist_ok=True)
            for gf in generated_files:
                dest = os.path.join(output_dir, os.path.basename(gf))
                shutil.move(gf, dest)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   [ERREUR] dcm2niix a échoué : {e}")
            return False
        except FileNotFoundError:
            print(f"   [ERREUR FATALE] dcm2niix est introuvable au chemin : {DCM2NIIX_EXE}")
            return False
            

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

            # --- MODIFICATION ICI ---
            # On remplace "plastimatch" par notre variable PLASTIMATCH_EXE
            commande = [
                PLASTIMATCH_EXE, "convert",
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
            print(f"   [ERREUR FATALE] L'exécutable Plastimatch est introuvable au chemin : {PLASTIMATCH_EXE}")
            return False

def check_pet_suv_metadata(file_paths: list) -> dict:
    """
    Vérifie si les métadonnées nécessaires au calcul SUV sont présentes.
    Retourne un dict avec les champs trouvés/manquants et l'unité native.
    """
    if not file_paths:
        return {"valid": False, "missing": ["NO_FILES"], "units": "UNKNOWN"}

    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        missing = []

        # Récupération de l'unité native (ex: BQML, CNTS, g/ml)
        units = str(getattr(ds, "Units", "UNKNOWN"))

        # Poids patient (On vérifie aussi qu'il n'est pas vide)
        if not hasattr(ds, "PatientWeight") or ds.PatientWeight is None or str(ds.PatientWeight).strip() == "":
            missing.append("PatientWeight")

        # Heure acquisition (On accepte SeriesTime comme backup si AcquisitionTime manque)
        if not hasattr(ds, "AcquisitionTime") and not hasattr(ds, "SeriesTime"):
            missing.append("AcquisitionTime/SeriesTime")

        # Infos radiopharmaceutiques
        if not hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            missing.append("RadiopharmaceuticalInformationSequence")
        else:
            rph = ds.RadiopharmaceuticalInformationSequence[0]

            if not hasattr(rph, "RadionuclideTotalDose") or rph.RadionuclideTotalDose is None:
                missing.append("RadionuclideTotalDose")

            if not hasattr(rph, "RadionuclideHalfLife") or rph.RadionuclideHalfLife is None:
                missing.append("RadionuclideHalfLife")

            if not hasattr(rph, "RadiopharmaceuticalStartTime") or rph.RadiopharmaceuticalStartTime is None:
                missing.append("RadiopharmaceuticalStartTime")

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "units": units
        }

    except Exception as e:
        return {
            "valid": False,
            "missing": [f"Erreur de lecture: {str(e)}"],
            "units": "UNKNOWN"
        }

def scan_and_group_dicoms(root_dir: str) -> dict:
    """
    Parcourt récursivement un dossier et regroupe tous les fichiers DICOM valides
    en fonction de leur SeriesInstanceUID.
    Retourne un dictionnaire : { 'SeriesInstanceUID': [liste_des_chemins_fichiers] }
    """
    print("--- 1. SCAN ET REGROUPEMENT DES SÉRIES DICOM ---")
    series_dict = defaultdict(list)

    files_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 1. On crée le chemin absolu direct (Pour prévenir tout souci de longueur de caractère de plus de 260 octets (qui ont causé de gros bugs lors de tests avec le QIN dataset), on va opter pour la stat du 
            # chemin absolu)
            abs_path = os.path.abspath(os.path.join(root, file))
            
            # 2. LE FIX MAGIQUE POUR WINDOWS : On ajoute le préfixe \\?\
            if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
                abs_path = '\\\\?\\' + abs_path
                
            files_list.append(abs_path)

    print(f"Analyse de {len(files_list)} fichiers en cours...")
    for file_path in tqdm(files_list, desc="Scan DICOM"):
        try:
            # Lecture rapide de l'en-tête (sans charger les lourds pixels en mémoire)
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                
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
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        return {
            "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
            "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")).upper(),
            "SeriesTime": str(getattr(ds, "SeriesTime", "000000")),
        }
    except Exception:
        return {}

def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, out_others_root: str):

    # --- INITIALISATION DU RAPPORT DE SYNTHÈSE ---
    rapport_erreurs = [
        "==================================================",
        "      RAPPORT DE SYNTHÈSE DE L'INGESTION DICOM    ",
        "==================================================\n",
        "--- ALERTES TEP (POUR LE CALCUL SUV FUTUR) ---"
    ]
    
    # 1. On scanne tout et on regroupe par série, peu importe l'organisation des dossiers
    series_groups = scan_and_group_dicoms(raw_data_root)
    mri_phases_by_patient = defaultdict(list)
    
    print("--- 2. ANALYSE ET ROUTAGE DES SÉRIES ---")
    
    for series_uid, file_paths in series_groups.items():
        meta = get_series_metadata(file_paths)
        if not meta:
            continue

        # On récupère directement l'ID qui est déjà anonymisé (ex: DUKE_001)
        patient_id = meta["PatientID"]
        modality = meta["Modality"]
        description = meta["SeriesDescription"]
        
        # --- ROUTAGE PET / CT ---
        if modality in ["PT", "CT"]:
            print(f"[{modality}] {description} (Patient: {patient_id})")
            
            if modality == "PT":

                # --- NOUVEAU : AUDIT QUALITÉ DU PET ---
                suv_check = check_pet_suv_metadata(file_paths)
                
                if suv_check["valid"]:
                    print(f" -> Métadonnées SUV complètes. (Unité : {suv_check['units']})")
                else:
                    alerte = f"[MANQUANT] Patient {patient_id} : {suv_check['missing']} (Unité: {suv_check['units']})"
                    print(f" -> {alerte}")
                    rapport_erreurs.append(alerte)

                if suv_check['units'] not in ["BQML", "CNTS"]:
                    alerte_unite = f"[UNITÉ ANORMALE] Patient {patient_id} : L'unité est '{suv_check['units']}'. Vérifiez si ce n'est pas déjà un SUV !"
                    print(f" -> {alerte_unite}")
                    rapport_erreurs.append(alerte_unite)
                 
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
                clean_desc = "".join(c if c.isalnum() else "_" for c in description)
                clean_desc = "_".join(filter(None, clean_desc.split("_")))
                
                autres_dir = os.path.join(out_mri_root, patient_id, "autres_irm")
                file_prefix = f"{patient_id}_{clean_desc}" # dcm2niix a besoin d'un nom de base, pas d'un fichier avec extension
                
                print(f" -> Conversion IRM Secondaire via dcm2niix vers : {autres_dir}")
                # --- ON UTILISE dcm2niix ICI ---
                convert_files_to_nifti_dcm2niix(file_paths, autres_dir, file_prefix)

        # --- NOUVEAU : INTERCEPTION DES MASQUES (RTSTRUCT et SEG) ---
        elif modality in ["RTSTRUCT", "SEG"]:
            desc_lower = description.lower()

            # Pourquoi l'heuristique series_uid[-5:] ? C'est une sécurité vitale : un médecin peut avoir dessiné 3 masques différents (par exemple : un masque de la tumeur, un masque des ganglions lymphatiques, 
            # et un masque du cœur pour éviter de l'irradier), qui auront tous la même modalité. En utilisant les 5 derniers caractères du SeriesInstanceUID, on s'assure qu'ils ne s'écrasent pas les uns les autres
            # dans le même dossier.
            
            # Heuristique simple : on regarde si le mot MR ou IRM est dans la description
            # pour deviner à quelle modalité ce masque appartient.
            if "mr" in desc_lower or "irm" in desc_lower:
                print(f" [MASQUE IRM ISOLÉ] Modality: {modality} | Desc: {description} (Patient: {patient_id})")
                # On le range dans la base IRM
                mask_dir = os.path.join(out_mri_root, patient_id, "dicom_mask_rm", series_uid[-5:])
            else:
                print(f" [MASQUE PET/CT ISOLÉ] Modality: {modality} | Desc: {description} (Patient: {patient_id})")
                # Par défaut (ou si mention de CT/PT), on le range dans la base PET/CT
                mask_dir = os.path.join(out_petct_root, patient_id, "dicom_mask_pet", series_uid[-5:])
                
            os.makedirs(mask_dir, exist_ok=True)
            for f in file_paths:
                shutil.copy2(f, mask_dir)

        # --- ARCHIVAGE DES MODALITÉS INCONNUES (The "Others" Data Lake) ---
        else:
            print(f" [ARCHIVÉ] Modalité '{modality}' : {description} (Patient: {patient_id})")
            
            # Nettoyage du nom pour le dossier
            clean_desc = "".join(c if c.isalnum() else "_" for c in description)
            clean_desc = "_".join(filter(None, clean_desc.split("_")))
            if not clean_desc:
                clean_desc = "SANS_DESCRIPTION"
            
            # Création de l'arborescence : Base_Autres / PatientID / Modalité / Description
            other_dicom_dir = os.path.join(out_others_root, patient_id, modality, f"{clean_desc}_{series_uid[-5:]}")
            os.makedirs(other_dicom_dir, exist_ok=True)
            
            # On copie les DICOMs bruts (pas de conversion Plastimatch pour éviter les crashs)
            for f in file_paths:
                shutil.copy2(f, other_dicom_dir)

    # --- 3. TRAITEMENT ET TRI TEMPOREL DES PHASES IRM ---
    print("\n--- 3. CONVERSION ET TRI CHRONOLOGIQUE DES PHASES IRM ---")
    for patient_id, phases in mri_phases_by_patient.items():
        # Tri chronologique basé sur le SeriesTime du DICOM
        phases_triees = sorted(phases, key=lambda x: x["time"])
        imgs_dir = os.path.join(out_mri_root, patient_id, "imgs")
        
        for index, phase in enumerate(phases_triees):
            file_prefix = f"{patient_id}_{index:04d}"
            print(f" -> IRM Phase {index} ({phase['desc']}) via dcm2niix vers : {imgs_dir}")
            
            # --- ON UTILISE dcm2niix ICI ---
            convert_files_to_nifti_dcm2niix(phase["files"], imgs_dir, file_prefix)

    # --- SAUVEGARDE DU RAPPORT DE SYNTHÈSE ---
    chemin_rapport = os.path.join(os.path.dirname(out_petct_root), "rapport_ingestion.txt")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write("\n".join(rapport_erreurs))
        if len(rapport_erreurs) == 4: # Si aucune erreur n'a été ajoutée (juste l'en-tête)
            f.write("\nAucune anomalie détectée sur les métadonnées TEP !\n")
            
    print(f"\n -> Un rapport de synthèse a été généré ici : {chemin_rapport}")
    print("\n=== INGÉSTION TERMINÉE AVEC SUCCÈS ===")
            
    print("\n=== INGÉSTION TERMINÉE AVEC SUCCÈS ===")

if __name__ == "__main__":
    DOSSIER_DICOM_VRAC = "./data_hopital_safe"
    PROJET_IRM_RACINE = "./Base_IRM"
    PROJET_PETCT_RACINE = "./Base_PETCT"
    PROJET_AUTRES_RACINE = "./Base_Autres" # <-- NOUVEAU

    # On passe bien les 4 dossiers
    ingest_raw_dicoms(
        DOSSIER_DICOM_VRAC, 
        PROJET_IRM_RACINE, 
        PROJET_PETCT_RACINE, 
        PROJET_AUTRES_RACINE
    )

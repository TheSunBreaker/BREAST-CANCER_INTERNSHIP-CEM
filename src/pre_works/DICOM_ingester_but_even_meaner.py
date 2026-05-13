#!/usr/bin/env python3
r"""
Script d'ingestion DICOM robuste - V3 (Longitudinale + Matching Strict des Masques).
Groupe les fichiers par SeriesInstanceUID pour gérer les dossiers "fourre-tout" des hôpitaux,
extrait les métadonnées de chaque série, filtre (T1/DCE, CT, PET) et convertit en NIfTI.

=============================================================================
NOUVEAUTÉS V3 :
- Matching des Masques : Lit le ReferencedSeriesSequence des DICOM SEG pour les 
  placer dans le même espace temporel exact que l'image annotée.
- Masques Orphelins : Isole les masques sans image parente pour éviter la corruption de la vérité terrain.
- Synchronisation PET/SUV : Les dossiers TEP et les NIfTI _RAW partagent les mêmes 
  suffixes temporels pour un couplage facile lors de la conversion SUV.
=============================================================================

ATTENTION, IL EST NECESSAIRE, POUR UTILISER CE SCRIPT, D'INSTALLER PLASTIMATCH SUR SA MACHINE VIA LE LIEN https://sourceforge.net/projects/plastimatch/postdownload POUR WINDOWS, PUIS
D'EXTRAIRE LES FICHIERS AVEC UNE COMMANDE DU STYLE 'msiexec /a "C:\Users\coul0426\Downloads\Plastimatch-1.9.4-win64.msi" /qb TARGETDIR="C:\Users\coul0426\plastimatch_portable"'. Le binaire sera alors à 
'C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe'

LE ROI DE DICOM TO NII COTE PET ET CT C'EST PLASTIMATCH. PAR CONTRE, POUR LES IRMS, PLASTIMATCH A TENDANCE A ECHOUER. SURTOUT POUR DES IRMS EXOTIQUES. COMME UN DCE 4D. EXACTEMENT
 CE QUI NOUS INTERESSE. DONC, ON A RECOURT AU ROI DE L'IRM dcm2niix. QUAND CE SERA IRM, ON AURA RECOURT A LUI. IL FAUT DONC TELECHARGER LE ZIP SUR LE GIT "https://github.com/rordenlab/dcm2niix/releases",
 ET EXTRAIRE POUR AVOIR LE ".exe". DANS MON CAS, JE L'AI MIT AU "C:\Users\coul0426\dcm2niix_portable\dcm2niix.exe". 
"""

import os
import shutil
import pydicom
import SimpleITK as sitk
from collections import defaultdict
import glob
from tqdm import tqdm
from datetime import datetime

import tempfile
import subprocess

# Chemin vers l'exécutable Plastimatch portable
PLASTIMATCH_EXE = r"C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe"
# --- NOUVELLE ARME POUR L'IRM ---
DCM2NIIX_EXE = r"C:\Users\coul0426\dcm2niix_portable\dcm2niix.exe"

# Nombre de phases IRMs de référence
REF_NB_IRMS_PHASES = 4

# ============================================================================
# FONCTIONS DE TEMPORALITÉ ET METADATA AVANCÉES
# ============================================================================

def get_series_datetime(file_paths: list) -> datetime:
    """Récupère la date et l'heure d'acquisition absolue."""
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        date_str = getattr(ds, 'AcquisitionDate', getattr(ds, 'SeriesDate', '19000101'))
        time_str = getattr(ds, 'AcquisitionTime', getattr(ds, 'SeriesTime', '000000.0'))
        
        date_str = date_str.strip() if date_str else '19000101'
        time_str = time_str.split('.')[0].strip() if time_str else '000000'
        
        if len(time_str) < 6:
            time_str = time_str.ljust(6, '0')

        return datetime.strptime(f"{date_str}{time_str[:6]}", "%Y%m%d%H%M%S")
    except Exception as e:
        return datetime(1900, 1, 1)

def generate_temporal_log(dicom_paths: list, output_dir: str):
    """Génère le DCE_temporal_log.txt avec les écarts en secondes entre chaque phase."""
    unique_times = set()
    for f in dicom_paths:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            acq_time = getattr(ds, "AcquisitionTime", getattr(ds, "ContentTime", None))
            if acq_time:
                unique_times.add(acq_time.split('.')[0])
        except:
            continue

    sorted_times = sorted(list(unique_times))
    log_path = os.path.join(output_dir, "DCE_temporal_log.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== LOG TEMPOREL DCE ===\n")
        f.write(f"Nombre de phases temporelles distinctes detectees : {len(sorted_times)}\n\n")
        
        if len(sorted_times) < 2:
            f.write("[ALERTE] Pas assez de timestamps distincts pour calculer des intervalles.\n")
            return

        dt_times = [datetime.strptime(t[:6], "%H%M%S") for t in sorted_times]
        total_delta = (dt_times[-1] - dt_times[0]).total_seconds()
        f.write(f"Temps total entre Phase 1 et Phase Finale : {total_delta} secondes\n\n")
        
        f.write("Intervalles entre les phases :\n")
        for i in range(1, len(dt_times)):
            delta = (dt_times[i] - dt_times[i-1]).total_seconds()
            f.write(f" - Phase {i-1} -> Phase {i} : {delta} secondes\n")

def get_referenced_series_uid(file_paths: list) -> str:
    """
    Tente de lire la séquence de référence dans un masque (SEG ou RTSTRUCT).
    C'est la clé pour garantir que le masque annoté correspond EXACTEMENT à la bonne image.
    """
    if not file_paths:
        return None
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        if hasattr(ds, "ReferencedSeriesSequence") and len(ds.ReferencedSeriesSequence) > 0:
            return ds.ReferencedSeriesSequence[0].SeriesInstanceUID
    except Exception:
        pass
    return None

# ============================================================================
# CONVERSIONS
# ============================================================================

def convert_files_to_nifti_dcm2niix(file_paths: list, output_dir: str, file_prefix: str, patient_root_dir: str) -> bool:
    if not file_paths:
        return False
        
    with tempfile.TemporaryDirectory() as tmp_dir:
        dicom_dir = os.path.join(tmp_dir, "dicoms")
        nifti_dir = os.path.join(tmp_dir, "nifti")
        os.makedirs(dicom_dir)
        os.makedirs(nifti_dir)
        
        for f in file_paths:
            shutil.copy2(f, dicom_dir)
            
        try:
            commande = [
                DCM2NIIX_EXE, "-z", "y", "-f", file_prefix, "-o", nifti_dir, dicom_dir
            ]
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            generated_files = glob.glob(os.path.join(nifti_dir, "*.nii.gz"))
            if not generated_files:
                return 0
                
            os.makedirs(output_dir, exist_ok=True)
            total_phases_generees = 0
            
            for gf in generated_files:
                dest = os.path.join(output_dir, os.path.basename(gf))
                shutil.move(gf, dest)
                
                split_prefix = os.path.basename(dest).replace(".nii.gz", "")
                nb_vols = split_4d_nifti_to_3d(dest, output_dir, split_prefix, patient_root_dir)
                total_phases_generees += nb_vols
                
            return total_phases_generees
        except Exception as e:
            print(f"   [ERREUR] dcm2niix a échoué : {e}")
            return 0

def split_4d_nifti_to_3d(nifti_path: str, output_dir: str, file_prefix: str, patient_root_dir: str) -> int:
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(nifti_path)
        reader.ReadImageInformation()
        size = reader.GetSize()
        
        if len(size) < 4 or size[3] <= 1:
            return 1 
            
        nb_volumes = size[3]
        img_4d = sitk.ReadImage(nifti_path)
        
        for t in range(nb_volumes):
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize([size[0], size[1], size[2], 0])
            extractor.SetIndex([0, 0, 0, t])
            img_3d = extractor.Execute(img_4d)
            
            out_path = os.path.join(output_dir, f"{file_prefix}_split{t:04d}.nii.gz")
            sitk.WriteImage(img_3d, out_path)
            
        # Archivage du 4D original
        archive_dir = os.path.join(patient_root_dir, "DCE_4D")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, os.path.basename(nifti_path).replace(".nii.gz", "_4D_original.nii.gz"))
        shutil.move(nifti_path, archive_path)
        
        return nb_volumes
    except Exception:
        return 1

def convert_files_to_nifti_plastimatch(file_paths: list, output_path: str) -> bool:
    if not file_paths:
        return False
    with tempfile.TemporaryDirectory() as tmp_dicom_dir:
        for f in file_paths:
            shutil.copy2(f, tmp_dicom_dir)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            commande = [
                PLASTIMATCH_EXE, "convert", "--input", tmp_dicom_dir,
                "--output-img", output_path, "--output-type", "float"
            ]
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return True
        except Exception:
            return False

# ============================================================================
# AUDITS METADATA
# ============================================================================
def check_mri_metadata(file_paths: list) -> dict:
    if not file_paths: return {"TR": "UNKNOWN", "TE": "UNKNOWN"}
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        return {"TR": str(getattr(ds, "RepetitionTime", "MISSING")), "TE": str(getattr(ds, "EchoTime", "MISSING"))}
    except: return {"TR": "ERROR", "TE": "ERROR"}

def check_pet_suv_metadata(file_paths: list) -> dict:
    if not file_paths: return {"valid": False, "missing": ["NO_FILES"], "units": "UNKNOWN"}
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        missing = []
        units = str(getattr(ds, "Units", "UNKNOWN"))
        if not hasattr(ds, "PatientWeight") or ds.PatientWeight is None or str(ds.PatientWeight).strip() == "": missing.append("PatientWeight")
        if not hasattr(ds, "AcquisitionTime") and not hasattr(ds, "SeriesTime"): missing.append("AcquisitionTime/SeriesTime")
        if not hasattr(ds, "RadiopharmaceuticalInformationSequence"): missing.append("RadiopharmaceuticalInformationSequence")
        else:
            rph = ds.RadiopharmaceuticalInformationSequence[0]
            if not hasattr(rph, "RadionuclideTotalDose") or rph.RadionuclideTotalDose is None: missing.append("RadionuclideTotalDose")
            if not hasattr(rph, "RadionuclideHalfLife") or rph.RadionuclideHalfLife is None: missing.append("RadionuclideHalfLife")
            if not hasattr(rph, "RadiopharmaceuticalStartTime") or rph.RadiopharmaceuticalStartTime is None: missing.append("RadiopharmaceuticalStartTime")
        return {"valid": len(missing) == 0, "missing": missing, "units": units}
    except Exception as e: return {"valid": False, "missing": [f"Erreur: {e}"], "units": "UNKNOWN"}

def scan_and_group_dicoms(root_dir: str) -> dict:
    series_dict = defaultdict(list)
    files_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            abs_path = os.path.abspath(os.path.join(root, file))
            if os.name == 'nt' and not abs_path.startswith('\\\\?\\'): abs_path = '\\\\?\\' + abs_path
            files_list.append(abs_path)

    for file_path in tqdm(files_list, desc="Scan DICOM"):
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            if hasattr(ds, 'SeriesInstanceUID'): series_dict[ds.SeriesInstanceUID].append(file_path)
        except Exception: continue
    return series_dict

def get_series_metadata(file_paths: list) -> dict:
    if not file_paths: return {}
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        return {
            "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
            "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")).upper(),
            "SeriesTime": str(getattr(ds, "SeriesTime", "000000")),
        }
    except: return {}

# ============================================================================
# L'ORCHESTRATEUR PRINCIPAL (ROUTAGE ET MATCHING)
# ============================================================================

def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, out_others_root: str):

    stats = {
        "patients": set(), "pet_traites": 0, "ct_traites": 0,
        "masques_pet": 0, "masques_irm": 0, "masques_orphelins": 0,
        "autres_modalites": 0, "irm_secondaires": 0
    }
    mri_phases_distribution = defaultdict(int)
    rapport_erreurs = ["=== RAPPORT DE SYNTHÈSE ===\n--- ALERTES TEP ---"]
    
    series_groups = scan_and_group_dicoms(raw_data_root)
    patient_data = defaultdict(lambda: defaultdict(list))
    
    print("--- 2. ANALYSE ET PRÉ-ROUTAGE CHRONOLOGIQUE ---")
    
    for series_uid, file_paths in series_groups.items():
        meta = get_series_metadata(file_paths)
        if not meta: continue

        patient_id = meta["PatientID"]
        modality = meta["Modality"]
        stats["patients"].add(patient_id)
        dt = get_series_datetime(file_paths)
        
        series_info = {"uid": series_uid, "files": file_paths, "desc": meta["SeriesDescription"], "dt": dt}

        if modality == "PT": patient_data[patient_id]["PT"].append(series_info)
        elif modality == "CT": patient_data[patient_id]["CT"].append(series_info)
        elif modality == "MR":
            if "T1" in meta["SeriesDescription"] or "DCE" in meta["SeriesDescription"]:
                patient_data[patient_id]["MR"].append(series_info)
            else: patient_data[patient_id]["MR_AUTRES"].append(series_info)
        elif modality in ["RTSTRUCT", "SEG"]:
            # Ajout immédiat de la balise de référence si elle existe
            series_info["ref_uid"] = get_referenced_series_uid(file_paths)
            if "mr" in meta["SeriesDescription"].lower() or "irm" in meta["SeriesDescription"].lower():
                patient_data[patient_id]["SEG_MR"].append(series_info)
            else: patient_data[patient_id]["SEG_PT"].append(series_info)
        else: patient_data[patient_id]["AUTRES"].append({"modality": modality, **series_info})

    # Dictionnaire mémoire pour faire matcher les Masques aux images correspondantes
    # uid_image -> "dossier_cible" (ex: "dicom_mask_rm_20230101_1030")
    uid_to_target_mask_folder = {}

    print("\n--- 3. ROUTAGE LONGITUDINAL ---")
    
    for patient_id, modalities in patient_data.items():
        print(f"\n[PATIENT] Traitement de {patient_id}...")
        
        # --- TRAITEMENT IRM PRINCIPAL ---
        if "MR" in modalities:
            mr_sorted = sorted(modalities["MR"], key=lambda x: x["dt"])
            patient_mri_root = os.path.join(out_mri_root, patient_id)
            
            for index, seq in enumerate(mr_sorted):
                date_str = seq["dt"].strftime("%Y%m%d_%H%M")
                target_folder_name = "imgs" if index == 0 else f"imgs_{date_str}"
                imgs_dir = os.path.join(patient_mri_root, target_folder_name)
                
                # Mémorisation pour les masques IRM
                uid_to_target_mask_folder[seq["uid"]] = "dicom_mask_rm" if index == 0 else f"dicom_mask_rm_{date_str}"
                
                print(f" -> [IRM DCE] Visite {index+1} dirigée vers : {target_folder_name}/")
                file_prefix = f"{patient_id}_{index:04d}"
                
                nb_gen = convert_files_to_nifti_dcm2niix(seq["files"], imgs_dir, file_prefix, patient_mri_root)
                if nb_gen > 0:
                    mri_phases_distribution[nb_gen] += 1
                    generate_temporal_log(seq["files"], imgs_dir)

        # --- TRAITEMENT PET ---
        if "PT" in modalities:
            pt_sorted = sorted(modalities["PT"], key=lambda x: x["dt"])
            patient_pet_root = os.path.join(out_petct_root, patient_id)
            
            for index, seq in enumerate(pt_sorted):
                date_str = seq["dt"].strftime("%Y%m%d_%H%M")
                
                tep_folder_name = "TEP" if index == 0 else f"TEP_{date_str}"
                imgs_folder_name = "imgs" if index == 0 else f"imgs_{date_str}"
                
                # Mémorisation pour les masques PET
                uid_to_target_mask_folder[seq["uid"]] = "dicom_mask_pet" if index == 0 else f"dicom_mask_pet_{date_str}"
                
                tep_dicom_dir = os.path.join(patient_pet_root, tep_folder_name, seq["uid"][-5:])
                imgs_dir = os.path.join(patient_pet_root, imgs_folder_name)
                os.makedirs(tep_dicom_dir, exist_ok=True)
                
                stats["pet_traites"] += 1
                for f in seq["files"]: shutil.copy2(f, tep_dicom_dir)
                
                # NOUVEAU : Synchronisation parfaite des noms pour le script SUV.
                # Le RAW reçoit le même marqueur temporel (ou "Baseline") que son dossier DICOM
                time_marker = "Baseline" if index == 0 else date_str
                out_raw_pet_path = os.path.join(imgs_dir, f"{patient_id}_TEP_{time_marker}_{seq['uid'][-5:]}_RAW.nii.gz")
                print(f" -> [PET] Visite {index+1} : Dossier {tep_folder_name}/ <---> NIfTI {os.path.basename(out_raw_pet_path)}")
                
                convert_files_to_nifti_plastimatch(seq["files"], out_raw_pet_path)

        # --- TRAITEMENT CT ---
        if "CT" in modalities:
            ct_sorted = sorted(modalities["CT"], key=lambda x: x["dt"])
            for index, seq in enumerate(ct_sorted):
                date_str = seq["dt"].strftime("%Y%m%d_%H%M")
                imgs_folder_name = "imgs" if index == 0 else f"imgs_{date_str}"
                
                # Mémorisation pour les masques CT (si attachés au CT plutôt qu'au PET)
                uid_to_target_mask_folder[seq["uid"]] = "dicom_mask_pet" if index == 0 else f"dicom_mask_pet_{date_str}"
                
                imgs_dir = os.path.join(out_petct_root, patient_id, imgs_folder_name)
                stats["ct_traites"] += 1
                out_path = os.path.join(imgs_dir, f"{patient_id}_TDM_{seq['uid'][-5:]}.nii.gz")
                convert_files_to_nifti_plastimatch(seq["files"], out_path)

        # --- TRAITEMENT MASQUES (AVEC MATCHING STRICT) ---
        for mask_type in ["SEG_MR", "SEG_PT"]:
            if mask_type in modalities:
                root_target = out_mri_root if mask_type == "SEG_MR" else out_petct_root
                
                for seq in modalities[mask_type]:
                    ref_uid = seq.get("ref_uid")
                    target_folder = None
                    
                    # Tentative de correspondance absolue avec une image ingérée
                    if ref_uid and ref_uid in uid_to_target_mask_folder:
                        target_folder = uid_to_target_mask_folder[ref_uid]
                        print(f" -> [MASQUE MATCHE] Correspondance absolue trouvée vers {target_folder}")
                    else:
                        # Si pas de lien direct, on l'isole en "orphelin"
                        target_folder = f"dicom_mask_orphelins_{seq['dt'].strftime('%Y%m%d')}"
                        stats["masques_orphelins"] += 1
                        print(f" -> [MASQUE ORPHELIN] Aucune image source trouvée. Isolé dans {target_folder}")

                    # On ajoute le patient_id au suffixe en plus du uid[-5:] pour la lisibilité
                    mask_dir = os.path.join(root_target, patient_id, target_folder, f"{patient_id}_{seq['uid'][-5:]}")
                    os.makedirs(mask_dir, exist_ok=True)
                    
                    if mask_type == "SEG_MR": stats["masques_irm"] += 1
                    else: stats["masques_pet"] += 1
                    
                    for f in seq["files"]: shutil.copy2(f, mask_dir)

    print("\n=== INGÉSTION V3 TERMINÉE ===")

if __name__ == "__main__":
    DOSSIER_DICOM_VRAC = "./data_hopital_safe"
    PROJET_IRM_RACINE = "./Base_IRM"
    PROJET_PETCT_RACINE = "./Base_PETCT"
    PROJET_AUTRES_RACINE = "./Base_Autres" 

    ingest_raw_dicoms(
        DOSSIER_DICOM_VRAC, 
        PROJET_IRM_RACINE, 
        PROJET_PETCT_RACINE, 
        PROJET_AUTRES_RACINE
    )

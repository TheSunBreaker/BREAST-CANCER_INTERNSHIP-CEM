#!/usr/bin/env python3
r"""
Script d'ingestion DICOM robuste - V6 (L'Ingesteur Clinique Définitif).

=============================================================================
NOUVEAUTÉS V6 (Le Filtre Dynamique Multicentrique) :
- Anti-Reconstructions : Rejette automatiquement les MIP, Soustractions (SUB), 
  Scouts, et reformats (SAG/COR) grâce au tag DICOM 'ImageType' et aux mots-clés.
- Tri Temporel Absolu : Utilise 'TemporalPositionIdentifier' en priorité absolue 
  pour ordonner les phases DCE, avec fallback sur l'heure d'acquisition, 
  puis sur le SeriesNumber.
- Architecture : [Patient] -> [Visite (StudyUID)] -> [Phases DCE Pures (TempPos)]
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

# ============================================================================
# CONFIGURATIONS ET CHEMINS DES EXÉCUTABLES
# ============================================================================
# Plastimatch garantit une conversion spatiale rigoureuse (conservation des grilles pour le PET/CT)
PLASTIMATCH_EXE = r"C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe"
# dcm2niix est beaucoup plus souple pour lire les b-values, flip angles et dimensions 4D des IRM complexes
DCM2NIIX_EXE = r"C:\Users\coul0426\dcm2niix_portable\dcm2niix.exe"

# Standard clinique attendu pour notre projet de prédiction pCR
REF_NB_IRMS_PHASES = 4


# ============================================================================
# MODULE 1 : ANALYSE TEMPORELLE ET CLINIQUE (METADATA)
# ============================================================================

def get_series_datetime(file_paths: list) -> datetime:
    """
    Extrait l'heure absolue de la séquence pour distinguer la Baseline des suivis.
    On privilégie AcquisitionDate/Time (heure réelle où le patient était dans la machine)
    plutôt que SeriesDate/Time (qui peut parfois refléter l'heure de reconstruction informatique).
    """
    try:
        # stop_before_pixels=True est vital ici : on charge l'en-tête en quelques millisecondes 
        # sans charger les mégaoctets de pixels en RAM.
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        date_str = getattr(ds, 'AcquisitionDate', getattr(ds, 'SeriesDate', '19000101'))
        time_str = getattr(ds, 'AcquisitionTime', getattr(ds, 'SeriesTime', '000000.0'))
        
        # Nettoyage des strings (les machines GE ou Siemens ajoutent parfois des millisecondes après un point)
        date_str = date_str.strip() if date_str else '19000101'
        time_str = time_str.split('.')[0].strip() if time_str else '000000'
        
        # Padding si le format de l'heure est tronqué par le PACS de l'hôpital
        if len(time_str) < 6:
            time_str = time_str.ljust(6, '0')

        return datetime.strptime(f"{date_str}{time_str[:6]}", "%Y%m%d%H%M%S")
    except Exception as e:
        # Fallback de sécurité absolu pour ne pas crasher l'ingesteur
        return datetime(1900, 1, 1)

def generate_temporal_log(dicom_paths: list, output_dir: str):
    """
    Parcourt CHAQUE slice d'une série DCE pour repérer les changements d'AcquisitionTime.
    C'est la seule façon fiable de savoir combien de temps sépare la phase native (T1)
    des phases post-contraste, information vitale pour la pharmacocinétique de la tumeur.
    """
    unique_times = set()
    for f in dicom_paths:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            acq_time = getattr(ds, "AcquisitionTime", getattr(ds, "ContentTime", None))
            if acq_time:
                # On tronque pour ignorer les microsecondes entre les coupes axiales d'une MÊME phase 3D
                unique_times.add(acq_time.split('.')[0])
        except:
            continue

    # Le tri permet de remettre les phases dans l'ordre chronologique d'injection
    sorted_times = sorted(list(unique_times))
    log_path = os.path.join(output_dir, "DCE_temporal_log.txt")
    
    # On ajoute "a" (append) au lieu de "w" pour accumuler les logs si plusieurs séries 3D
    # forment la séquence DCE globale de cette visite.
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== BLOC TEMPOREL (Série dynamique) ===\n")
        f.write(f"Timestamps distincts detectes : {len(sorted_times)}\n")
        
        if len(sorted_times) < 2:
            f.write(" -> Volume statique ou metadata temps ecrasee.\n")
            return

        dt_times = [datetime.strptime(t[:6], "%H%M%S") for t in sorted_times]
        f.write(f"Temps total : {(dt_times[-1] - dt_times[0]).total_seconds()} s\n")
        for i in range(1, len(dt_times)):
            f.write(f" - Ecart {i-1}->{i} : {(dt_times[i] - dt_times[i-1]).total_seconds()} s\n")

# --- NOUVEAU : FILTRES CLINIQUES ANTI-RECONSTRUCTIONS ---
# La liste très permissive pour attraper toutes les vraies séquences (Constructeurs: GE, Philips, Siemens...)

DCE_INCLUDE_TERMS = [
    "DCE", "DYNAMIC", "DYN", "VIBRANT", "THRIVE", "E-THRIVE", 
    "POST", "PRE", "T1", "PERF", "DISCO", "LAVA", "TWIST"
]

# La liste agressive pour tuer les parasites (On a retiré "MAP" isolé et remplacé par "T1-MAP")
EXCLUDE_TERMS = [
    "SUB", "SOUSTRACTION", "MIP", "ADC", "TRACE", 
    "SCOUT", "LOC", "LOCALIZER", "REFORMAT", "MPR", 
    "COR", "SAG", "CORONAL", "SAGITTAL",
    "KTRANS", "KEP", "VE", "PARAMETRIC", "TTP", "MTT", "CBF", "CBV", "T1-MAP"
]

def looks_like_dce(desc: str) -> bool:
    """Pré-filtre très permissif pour attraper tout ce qui ressemble à de la perfusion/dynamique."""
    desc_upper = desc.upper()
    return any(term in desc_upper for term in DCE_INCLUDE_TERMS)

def is_valid_dce_phase(desc: str, image_type: list) -> bool:
    """Filtre anti-reconstructions strict."""
    desc_upper = desc.upper()
    img_type_str = [str(x).upper() for x in image_type] if image_type else []
    
    # 1. Anti-Parasites strict
    for term in EXCLUDE_TERMS:
        if term in desc_upper: return False
        if any(term in t for t in img_type_str): return False
            
    # 2. Gestion intelligente du DERIVED
    if "DERIVED" in img_type_str or "SECONDARY" in img_type_str:
        # On demande que la description matche notre whitelist permissive
        if not looks_like_dce(desc):
            return False
            
    return True

def get_referenced_series_uid(file_paths: list) -> str:
    """
    Extrait l'identifiant clinique de l'image sur laquelle le radiologue a dessiné le masque.
    Sans ça, comparer des dates peut lier un masque de Baseline à une image de Suivi.
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

def check_mri_metadata(file_paths: list) -> dict:
    """Audit du Temps de Répétition (TR) et Temps d'Écho (TE) pour vérifier l'homogénéité du protocole IRM."""
    if not file_paths: return {"TR": "UNKNOWN", "TE": "UNKNOWN"}
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        return {"TR": str(getattr(ds, "RepetitionTime", "MISSING")), "TE": str(getattr(ds, "EchoTime", "MISSING"))}
    except: return {"TR": "ERROR", "TE": "ERROR"}

def check_pet_suv_metadata(file_paths: list) -> dict:
    """
    Vérifie en amont si toutes les informations (Dose, Poids, Demi-vie) sont 
    présentes dans le DICOM pour garantir le futur calcul de la SUVbw par l'autre script.
    """
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


# ============================================================================
# MODULE 2 : CONVERSION ET MANIPULATION NIFTI
# ============================================================================

def convert_files_to_nifti_dcm2niix(file_paths: list, output_dir: str, file_prefix: str, patient_root_dir: str) -> bool:
    """
    Isoler les fichiers d'une série exacte dans un dossier temporaire est obligatoire,
    sinon dcm2niix risque de scanner tout le disque et de mélanger les patients.
    """
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
            # -z y = compression GZIP (.nii.gz), indispensable pour économiser l'espace disque
            commande = [DCM2NIIX_EXE, "-z", "y", "-f", file_prefix, "-o", nifti_dir, dicom_dir]
            subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            generated_files = glob.glob(os.path.join(nifti_dir, "*.nii.gz"))
            if not generated_files:
                return 0
                
            os.makedirs(output_dir, exist_ok=True)
            total_phases_generees = 0
            
            for gf in generated_files:
                dest = os.path.join(output_dir, os.path.basename(gf))
                shutil.move(gf, dest)
                
                # S'il y a un 4D, on délègue la tâche à SimpleITK pour le découpage
                split_prefix = os.path.basename(dest).replace(".nii.gz", "")
                nb_vols = split_4d_nifti_to_3d(dest, output_dir, split_prefix, patient_root_dir)
                total_phases_generees += nb_vols
                
            return total_phases_generees
        except Exception as e:
            print(f"   [ERREUR] dcm2niix a échoué : {e}")
            return 0

def split_4d_nifti_to_3d(nifti_path: str, output_dir: str, file_prefix: str, patient_root_dir: str) -> int:
    """
    Certaines IRM de QIN-Breast contiennent toute la dynamique du contraste dans un seul fichier.
    nnU-Net déteste le 4D (il attend des canaux séparés). Cette fonction extrait chaque phase 3D.
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(nifti_path)
        reader.ReadImageInformation()
        size = reader.GetSize()
        
        # Filtre anti-faux positifs : on s'assure qu'il y a bien une 4ème dimension (T) avec T > 1
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
            
        # On sauvegarde le 4D original dans un dossier sécurisé, hors de portée de nnU-Net
        archive_dir = os.path.join(patient_root_dir, "DCE_4D")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, os.path.basename(nifti_path).replace(".nii.gz", "_4D_original.nii.gz"))
        shutil.move(nifti_path, archive_path)
        
        return nb_volumes
    except Exception:
        return 1

def convert_files_to_nifti_plastimatch(file_paths: list, output_path: str) -> bool:
    """
    Conversion Plastimatch stricte. On utilise '--output-type float' pour garantir 
    qu'on ne perd aucune décimale sur les valeurs SUV ou HU (Hounsfield Units).
    """
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
            # NOUVEAU : On enlève le DEVNULL pour que les erreurs critiques s'affichent
            subprocess.run(commande, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"    [CRASH PLASTIMATCH] Erreur d'exécution : {e}")
            return False
        except FileNotFoundError:
            print(f"    [CRASH PLASTIMATCH] Exécutable introuvable au chemin : {PLASTIMATCH_EXE}")
            return False
        except Exception as e:
            print(f"    [CRASH PLASTIMATCH] Erreur inconnue : {e}")
            return False


# ============================================================================
# MODULE 3 : SCAN DISQUE ET IDENTIFICATION DICOM
# ============================================================================

def scan_and_group_dicoms(root_dir: str) -> dict:
    """
    Parcourt le disque entier et utilise le SeriesInstanceUID comme clé unique de regroupement.
    C'est le seul moyen de reconstruire une acquisition cohérente quand les hôpitaux 
    fournissent des arborescences de dossiers cassées.
    """
    series_dict = defaultdict(list)
    files_list = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            abs_path = os.path.abspath(os.path.join(root, file))
            # Fix critique sous Windows : autorise des chemins absolus > 260 caractères
            if os.name == 'nt' and not abs_path.startswith('\\\\?\\'): 
                abs_path = '\\\\?\\' + abs_path
            files_list.append(abs_path)

    for file_path in tqdm(files_list, desc="Scan DICOM"):
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            if hasattr(ds, 'SeriesInstanceUID'): 
                series_dict[ds.SeriesInstanceUID].append(file_path)
        except Exception: 
            continue # Fichier non DICOM ignoré silencieusement
    return series_dict

def get_series_metadata(file_paths: list) -> dict:
    """Extrait la carte d'identité de base de la série depuis le premier fichier trouvé."""
    if not file_paths: return {}
    try:
        ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
        return {
            "PatientID": str(getattr(ds, "PatientID", "UNKNOWN")),
            "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "UNKNOWN")).upper(),
            "StudyUID": str(getattr(ds, "StudyInstanceUID", "UNKNOWN_STUDY")),
            # NOUVEAU V6 : Tags cruciaux pour le tri et le filtrage DCE
            "ImageType": getattr(ds, "ImageType", []),
            "TempPos": getattr(ds, "TemporalPositionIdentifier", None),
            "SeriesNumber": getattr(ds, "SeriesNumber", 9999)
        }
    except: return {}


# ============================================================================
# MODULE 4 : L'ORCHESTRATEUR PRINCIPAL (ROUTAGE ET MATCHING)
# ============================================================================

def ingest_raw_dicoms(raw_data_root: str, out_mri_root: str, out_petct_root: str, out_others_root: str):

    # --- INITIALISATION DE L'AUDIT ---
    stats = {
        "patients": set(),
        "pet_traites": 0,
        "ct_traites": 0,
        "masques_pet": 0,
        "masques_irm": 0,
        "masques_orphelins": 0,
        "autres_modalites": 0,
        "irm_secondaires": 0,
        "irm_rejetees": 0
    }
    mri_phases_distribution = defaultdict(int)
    
    # La liste rapport_erreurs va accumuler les soucis en cours de vol (ex: Poids patient manquant)
    rapport_erreurs = ["=== RAPPORT DE SYNTHÈSE DES ERREURS ===\n\n--- ALERTES TEP (POUR CALCUL SUV) ---"]
    
    # Étape 1 : Le grand tri primaire (on groupe les fichiers par UID)
    series_groups = scan_and_group_dicoms(raw_data_root)
    
    # Structure logique: patient_data[ID][Modalité] = [liste de séries ordonnables]
    patient_data = defaultdict(lambda: defaultdict(list))
    
    print("--- 2. ANALYSE ET PRÉ-ROUTAGE CHRONOLOGIQUE ---")
    
    for series_uid, file_paths in series_groups.items():
        meta = get_series_metadata(file_paths)
        if not meta: continue

        patient_id = meta["PatientID"]
        modality = meta["Modality"]
        stats["patients"].add(patient_id)
        
        # On calcule le timestamp absolu qui va nous servir à distinguer Baseline et Follow-ups
        dt = get_series_datetime(file_paths)
        
        # Ajout du study_uid dans les infos pour grouper par visite plus tard
        series_info = {
            "uid": series_uid, 
            "files": file_paths, 
            "desc": meta["SeriesDescription"], 
            "dt": dt,
            "study_uid": meta["StudyUID"],
            "temp_pos": meta["TempPos"],
            "series_num": meta["SeriesNumber"],
            "modality": modality
        }
        # On ventile dans des "paniers" selon la modalité
        if modality == "PT": 
            patient_data[patient_id]["PT"].append(series_info)
        elif modality == "CT": 
            patient_data[patient_id]["CT"].append(series_info)
        elif modality == "MR":
            # NOUVEAU : On utilise la fonction permissive au lieu du simple test "T1/DCE"
            if looks_like_dce(meta["SeriesDescription"]):
                if is_valid_dce_phase(meta["SeriesDescription"], meta["ImageType"]):
                    patient_data[patient_id]["MR"].append(series_info)
                else:
                    print(f"   [FILTRE] Rejet de l'IRM dérivée/reconstruite : {meta['SeriesDescription']}")
                    stats["irm_rejetees"] += 1
                    patient_data[patient_id]["MR_AUTRES"].append(series_info)
            else: 
                patient_data[patient_id]["MR_AUTRES"].append(series_info)
        elif modality in ["RTSTRUCT", "SEG"]:
            # On cherche désespérément à lier ce masque à l'image via le ReferencedSeriesSequence
            series_info["ref_uid"] = get_referenced_series_uid(file_paths)
            if "mr" in meta["SeriesDescription"].lower() or "irm" in meta["SeriesDescription"].lower():
                patient_data[patient_id]["SEG_MR"].append(series_info)
            else: 
                patient_data[patient_id]["SEG_PT"].append(series_info)
        else: 
            patient_data[patient_id]["AUTRES"].append(series_info)

    # Dictionnaire mémoire crucial : permet de dire "L'image avec tel UID a été rangée dans tel dossier cible"
    uid_to_target_mask_folder = {}

    print("\n--- 3. ROUTAGE LONGITUDINAL ---")
    
    for patient_id, modalities in patient_data.items():
        print(f"\n[PATIENT] Traitement de {patient_id}...")
        
        # --------------------------------------------------------------------
        # TRAITEMENT IRM PRINCIPAL (Regroupement par Étude/Visite)
        # --------------------------------------------------------------------
        if "MR" in modalities:
            patient_mri_root = os.path.join(out_mri_root, patient_id)
            
            # 1. Grouper les séries par StudyUID (L'examen du jour)
            studies = defaultdict(list)
            for seq in modalities["MR"]:
                studies[seq["study_uid"]].append(seq)
                
            # 2. Trier les Études par l'heure de leur première séquence
            sorted_studies = sorted(studies.values(), key=lambda seqs: min(s["dt"] for s in seqs))
            
            # 3. Traiter chaque visite
            for study_index, study_seqs in enumerate(sorted_studies):
                # La date de la visite est dictée par sa toute première séquence
                study_date_str = min(s["dt"] for s in study_seqs).strftime("%Y%m%d_%H%M")
                
                # Baseline = Visite 0
                target_folder_name = "imgs" if study_index == 0 else f"imgs_{study_date_str}"
                imgs_dir = os.path.join(patient_mri_root, target_folder_name)
                print(f" -> [IRM VISITE {study_index+1}] Date {study_date_str} - Cible : {target_folder_name}/")
                
                # Vider/Créer le log temporel pour cette nouvelle visite
                os.makedirs(imgs_dir, exist_ok=True)
                with open(os.path.join(imgs_dir, "DCE_temporal_log.txt"), "w") as f:
                    f.write(f"=== LOG TEMPOREL GLOBAL - VISITE {study_index+1} ===\n")
                
                # 4. À l'intérieur de la visite, on trie les séries chronologiquement
               
                # Le Tri de Phase Ultime de la V6 :
                # _1. TemporalPositionIdentifier (Si le constructeur l'a rempli, c'est la vérité)
                # _2. Heure d'acquisition (Le plus fiable généralement)
                # _3. SeriesNumber (Fallback si les temps sont écrasés)
                seq_sorted = sorted(study_seqs, key=lambda x: (
                    x["temp_pos"] if x["temp_pos"] is not None else 99999,
                    x["dt"],
                    x["series_num"]
                ))
                
                phases_dans_visite = 0
                
                for phase_index, seq in enumerate(seq_sorted):
                    # Mémorisation pour le masque
                    uid_to_target_mask_folder[seq["uid"]] = "dicom_mask_rm" if study_index == 0 else f"dicom_mask_rm_{study_date_str}"
                    
                    # On numérote la phase séquentiellement dans la visite
                    file_prefix = f"{patient_id}_{phase_index:04d}"

                    # --- RESTAURATION : Audit des paramètres TR/TE ---
                    mri_audit = check_mri_metadata(seq["files"])
                    print(f"    -> IRM Série {phase_index} ({seq['desc']} | TR:{mri_audit['TR']} TE:{mri_audit['TE']}) via dcm2niix")
                  
                    nb_gen = convert_files_to_nifti_dcm2niix(seq["files"], imgs_dir, file_prefix, patient_mri_root)

                    # --- AJOUT: Notification du découpage ---
                    if nb_gen > 1:
                        print(f"       [SPLIT 4D] -> Série découpée en {nb_gen} phases 3D sur le disque.")
                    elif nb_gen == 1:
                        print(f"       [VOLUME 3D] -> 1 phase générée.")
                      
                    if nb_gen > 0:
                        phases_dans_visite += nb_gen
                        generate_temporal_log(seq["files"], imgs_dir)
                
                # Bilan statistique global
                mri_phases_distribution[phases_dans_visite] += 1

        # --------------------------------------------------------------------
        # TRAITEMENT UNIFIÉ PET ET CT (Synchronisation des dossiers)
        # --------------------------------------------------------------------
        if "PT" in modalities or "CT" in modalities:
            patient_petct_root = os.path.join(out_petct_root, patient_id)
            
            # 1. On groupe les PET et les CT dans les mêmes "Études" (Visites)
            petct_studies = defaultdict(list)
            for seq in modalities.get("PT", []): petct_studies[seq["study_uid"]].append(seq)
            for seq in modalities.get("CT", []): petct_studies[seq["study_uid"]].append(seq)

            # 2. On trie les visites par ordre chronologique
            sorted_petct_studies = sorted(petct_studies.values(), key=lambda seqs: min(s["dt"] for s in seqs))

            for study_index, study_seqs in enumerate(sorted_petct_studies):
                # L'heure du tout premier scan (souvent le CT) devient le nom officiel du dossier
                study_date_str = min(s["dt"] for s in study_seqs).strftime("%Y%m%d_%H%M")
                
                tep_folder_name = "TEP" if study_index == 0 else f"TEP_{study_date_str}"
                imgs_folder_name = "imgs" if study_index == 0 else f"imgs_{study_date_str}"
                imgs_dir = os.path.join(patient_petct_root, imgs_folder_name)
                
                # 3. On trie les séquences à l'intérieur de la visite
                seq_sorted = sorted(study_seqs, key=lambda x: x["dt"])
                
                for seq in seq_sorted:
                    # Mémorisation pour les masques (Qu'il soit dessiné sur le PET ou le CT, il ira au même endroit)
                    uid_to_target_mask_folder[seq["uid"]] = "dicom_mask_pet" if study_index == 0 else f"dicom_mask_pet_{study_date_str}"
                    
                    if seq["modality"] == "PT":
                        tep_dicom_dir = os.path.join(patient_petct_root, tep_folder_name, seq["uid"][-5:])
                        os.makedirs(tep_dicom_dir, exist_ok=True)
                        stats["pet_traites"] += 1
                        
                        # --- AUDIT SUV ---
                        suv_check = check_pet_suv_metadata(seq["files"])
                        if suv_check["valid"]:
                            print(f"    -> Métadonnées SUV complètes. (Unité : {suv_check['units']})")
                        else:
                            alerte = f"[MANQUANT] Patient {patient_id} Visite {study_index+1} : {suv_check['missing']} (Unité: {suv_check['units']})"
                            print(f"    -> {alerte}")
                            rapport_erreurs.append(alerte)
                        if suv_check['units'] not in ["BQML", "CNTS"]:
                            alerte_unite = f"[UNITÉ ANORMALE] Patient {patient_id} Visite {study_index+1} : L'unité est '{suv_check['units']}'."
                            rapport_erreurs.append(alerte_unite)
                        
                        for f in seq["files"]: shutil.copy2(f, tep_dicom_dir)
                        
                        time_marker = "Baseline" if study_index == 0 else study_date_str
                        out_raw_pet_path = os.path.join(imgs_dir, f"{patient_id}_TEP_{time_marker}_{seq['uid'][-5:]}_RAW.nii.gz")
                        print(f" -> [PET VISITE {study_index+1}] NIfTI généré : {os.path.basename(out_raw_pet_path)}")
                        convert_files_to_nifti_plastimatch(seq["files"], out_raw_pet_path)
                        
                    elif seq["modality"] == "CT":
                        stats["ct_traites"] += 1
                        out_path = os.path.join(imgs_dir, f"{patient_id}_TDM_{seq['uid'][-5:]}.nii.gz")
                        print(f" -> [CT VISITE {study_index+1}] NIfTI généré : {os.path.basename(out_path)}")
                        convert_files_to_nifti_plastimatch(seq["files"], out_path)
                      
        # --------------------------------------------------------------------
        # TRAITEMENT MASQUES (AVEC MATCHING STRICT)
        # --------------------------------------------------------------------
        for mask_type in ["SEG_MR", "SEG_PT"]:
            if mask_type in modalities:
                root_target = out_mri_root if mask_type == "SEG_MR" else out_petct_root
                
                for seq in modalities[mask_type]:
                    ref_uid = seq.get("ref_uid")
                    target_folder = None
                    
                    # C'est ici que la magie opère : si le masque pointe vers une image qu'on a traitée,
                    # il sera envoyé exactement dans l'espace temporel de cette image.
                    if ref_uid and ref_uid in uid_to_target_mask_folder:
                        target_folder = uid_to_target_mask_folder[ref_uid]
                        print(f" -> [MASQUE MATCHE] Correspondance absolue trouvée vers l'image annotée ({target_folder})")
                    else:
                        # Sans image parente trouvée, on isole le masque pour ne pas qu'il parasite
                        # la vérité terrain de la Baseline par erreur.
                        target_folder = f"dicom_mask_orphelins_{seq['dt'].strftime('%Y%m%d')}"
                        stats["masques_orphelins"] += 1
                        print(f" -> [MASQUE ORPHELIN] Aucune image source trouvée. Isolé dans {target_folder}")

                    # Le uid[-5:] protège contre la fusion de lésions multiples dans un même dossier
                    mask_dir = os.path.join(root_target, patient_id, target_folder, f"{patient_id}_{seq['uid'][-5:]}")
                    os.makedirs(mask_dir, exist_ok=True)
                    
                    if mask_type == "SEG_MR": stats["masques_irm"] += 1
                    else: stats["masques_pet"] += 1
                    
                    for f in seq["files"]: shutil.copy2(f, mask_dir)

        # --------------------------------------------------------------------
        # TRAITEMENT DES SÉQUENCES IRM REJETÉES / SECONDAIRES
        # --------------------------------------------------------------------
        if "MR_AUTRES" in modalities:
            for seq in modalities["MR_AUTRES"]:
                print(f" -> [IRM REJETÉE/SECONDAIRE] Archivage de : {seq['desc']}")
                stats["irm_secondaires"] += 1
                
                clean_desc = "".join(c if c.isalnum() else "_" for c in seq["desc"])
                clean_desc = "_".join(filter(None, clean_desc.split("_")))
                autres_dir = os.path.join(out_mri_root, patient_id, "autres_irm")
                file_prefix = f"{patient_id}_{clean_desc}_{seq['uid'][-5:]}" # Précaution au cas où plusieurs séries secondaires auraient la même description
                
                # NOUVEAU : On n'utilise PAS notre fonction de split pour les secondaires.
                # On exécute juste dcm2niix en direct et on laisse tel quel.
                os.makedirs(autres_dir, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for f in seq["files"]: shutil.copy2(f, tmp_dir)
                    try:
                        commande = [DCM2NIIX_EXE, "-z", "y", "-f", file_prefix, "-o", autres_dir, tmp_dir]
                        subprocess.run(commande, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    except Exception as e:
                        print(f"    [ERREUR dcm2niix archive] : {e}")

        # --------------------------------------------------------------------
        # TRAITEMENT DES AUTRES MODALITÉS (RTDOSE, RTPLAN, PR, etc.)
        # --------------------------------------------------------------------
        if "AUTRES" in modalities:
            for seq in modalities["AUTRES"]:
                print(f" -> [AUTRE MODALITÉ] Archivage brut de : {seq['modality']} ({seq['desc']})")
                stats["autres_modalites"] += 1
                
                clean_desc = "".join(c if c.isalnum() else "_" for c in seq["desc"])
                clean_desc = "_".join(filter(None, clean_desc.split("_"))) or "SANS_DESCRIPTION"
                other_dicom_dir = os.path.join(out_others_root, patient_id, seq["modality"], f"{clean_desc}_{seq['uid'][-5:]}")
                os.makedirs(other_dicom_dir, exist_ok=True)
                
                # Copie brute des DICOM, pas de NIfTI pour les formats inconnus
                for f in seq["files"]:
                    shutil.copy2(f, other_dicom_dir)

        # --------------------------------------------------------------------
        # SÉCURITÉ VISUELLE : SI LE PATIENT EST TOTALEMENT VIDE DE DONNÉES CIBLES
        # --------------------------------------------------------------------
        if not any(k in modalities for k in ["MR", "PT", "CT", "SEG_MR", "SEG_PT"]):
            print(" -> [ALERTE] Patient sans aucune donnée d'imagerie principale exploitable (Ni DCE, ni PET, ni CT, ni Masque).")      

    # --------------------------------------------------------------------
    # 4. ÉCRITURE DES RAPPORTS ET LOGS FINAUX (CORRIGÉ)
    # --------------------------------------------------------------------
    rapport_statistique = [
        "\n==================================================",
        "                 BILAN STATISTIQUE                ",
        "==================================================",
        f"Patients totaux analysés      : {len(stats['patients'])}",
        f"Volumes PET traités           : {stats['pet_traites']}",
        f"Volumes CT traités            : {stats['ct_traites']}",
        f"Masques PET/CT trouvés         : {stats['masques_pet']}",
        f"Masques IRM trouvés            : {stats['masques_irm']}",
        f"Masques Orphelins (DANGER)    : {stats['masques_orphelins']}",
        f"Séquences IRM dérivées écartées : {stats['irm_rejetees']} (MIP, SUB, etc.)",
        f"Séquences IRM secondaires     : {stats['irm_secondaires']}",
        f"Séquences exotiques archivées : {stats['autres_modalites']}",
        "\n--- DISTRIBUTION DES PHASES IRM (T1/DCE) ---"
    ]
    
    if not mri_phases_distribution:
        rapport_statistique.append("-> Aucune séquence IRM principale (T1/DCE) validée.")
    else:
        for nb_phases, nb_visites in sorted(mri_phases_distribution.items()):
            rapport_statistique.append(f"-> {nb_visites} visite(s) possèdent {nb_phases} phase(s) (Baseline ou Suivi).")
            if nb_phases != REF_NB_IRMS_PHASES:
                rapport_statistique[-1] += " [HORS STANDARD]"

    # Affichage en console pour un suivi en temps réel
    for ligne in rapport_statistique:
        print(ligne)

    # Écriture définitive du fichier de log général à la racine du projet
    chemin_rapport = os.path.join(os.path.dirname(out_petct_root), "rapport_ingestion_v6.txt")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write("\n".join(rapport_erreurs))
        f.write("\n")
        f.write("\n".join(rapport_statistique))
        
    print(f"\n -> Rapport détaillé écrit avec succès ici : {chemin_rapport}")
    print("\n=== INGÉSTION V6 TERMINÉE AVEC SUCCÈS ===")

if __name__ == "__main__":
    # Définition des chemins relatifs
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

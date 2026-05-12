import argparse
from pathlib import Path
import os
import csv
import pydicom
import SimpleITK as sitk

from utils.suv_conversion import (  
    extract_patient_parameters,   # Extrait dose/poids/demi-vie depuis l'en-tête DICOM
    compute_suv_factors,          # Calcule le facteur mathématique SUVbw
    write_normalized_image        # Multiplie l'image par le facteur et sauvegarde
)

# =====================================================================
# FONCTIONS UTILITAIRES DE RECHERCHE ET FORMATAGE DES MÉTADONNÉES
# =====================================================================

def _find_pet_header_with_rph(dir_path: Path):
    """
    Parcourt RÉCURSIVEMENT un dossier pour trouver UN fichier DICOM TEP (PT) valide
    qui contient notre fameuse séquence d'informations radiopharmaceutiques.
    Grâce au rglob("*"), peu importe si l'ingesteur a mis les fichiers dans un sous-dossier UID !
    """
    for f in dir_path.rglob("*"):
        if not f.is_file():
            continue
        try:
            # stop_before_pixels=True : Astuce d'optimisation vitale. 
            # On ne charge QUE le texte (l'en-tête), pas la lourde matrice d'image.
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
            
        # On vérifie que c'est bien un TEP ("PT") ET qu'il a les infos d'injection
        if getattr(ds, "Modality", None) == "PT" and hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            return ds, f
            
    return None, None

def _load_csv_params(csv_path: Path):
    """
    Charge un CSV de secours (Fallback). Si le PACS de l'hôpital a effacé le poids 
    de la patiente dans le DICOM (ça arrive souvent pour des raisons de RGPD), 
    on peut le fournir manuellement via ce CSV.
    """
    if not csv_path:
        return {}
    db = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("subject_id", "")).strip()
            if not sid:
                continue
            db[sid] = row
    return db

def _row_to_params(row: dict):
    """
    Convertit une ligne brute du CSV de secours en un dictionnaire de types propres (float, str)
    compréhensible par la fonction compute_suv_factors().
    """
    def fget(name, default=""):
        v = row.get(name, default)
        return "" if v is None else str(v).strip()

    def ffloat(s, default=0.0):
        try:
            return float(s)
        except Exception:
            return default

    return {
        "injected_dose": ffloat(fget("injected_dose", 0.0)),
        "patient_weight": ffloat(fget("patient_weight", 0.0)),
        "patient_height": ffloat(fget("patient_height", 0.0)),
        "half_life": ffloat(fget("half_life", 0.0)),
        "injection_time": fget("injection_time", "000000"),
        "series_time": fget("series_time", "000000"),
        "patient_sex": fget("sex", "UNKNOWN")
    }


# =====================================================================
# FONCTION PRINCIPALE (L'ORCHESTRATEUR SUV)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convertit un NIfTI TEP brut (Plastimatch) en TEP SUVbw en utilisant les en-têtes DICOM."
    )
    # L'input_root correspondra à notre "PROJET_PETCT_RACINE"
    parser.add_argument(
    "input_root",
    type=Path,
    nargs="?",
    default="Base_PETCT",
    help="Dossier racine contenant les subject_xxx"
    )
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Fichier CSV optionnel de secours pour les paramètres cliniques manquants.")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les fichiers SUV existants.")
    args = parser.parse_args()

    input_root: Path = args.input_root
    
    # Création du fichier de log pour tracer toutes les erreurs (Très important pour la reproductibilité)
    global_log = os.path.join(input_root, "suv_conversion_log.txt")
    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n================= SUV CONVERSION START =================\n")
        log.write(f"Input root: {input_root}\n")

    csv_db = _load_csv_params(args.metadata_csv) if args.metadata_csv else {}

    # Découverte de tous les dossiers patients dans la racine
    subjects = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subjects:
        print(f"[ERREUR] Aucun dossier patient trouvé sous: {input_root}")
        return

    # Initialisation des compteurs pour le résumé de fin de script
    n_total, n_converted, n_skipped_no_raw, n_skipped_params, n_exists = 0, 0, 0, 0, 0

    print(f"Trouvé {len(subjects)} dossier(s) patient.")
    
    for subj_dir in subjects:
        subject_id = subj_dir.name
        n_total += 1
        print(f"\n[SUBJ] {subject_id}")

        # --- 1. DÉFINITION DES CHEMINS ---
        # Le dossier contenant les DICOM originaux (pour lire les métadonnées)
        tep_dicom_dir = subj_dir / "TEP"
        # Le dossier contenant les images NIfTI (pour lire le RAW et écrire le SUV)
        imgs_dir = subj_dir / "imgs"
        
        # Le fichier généré par Plastimatch lors de l'ingestion ! (Valeurs en Bq/mL)
        raw_tep_path = imgs_dir / f"{subject_id}_TEP_RAW.nii.gz"
        # Le fichier cible que l'on veut créer (Valeurs en SUVbw)
        out_tep_path = imgs_dir / f"{subject_id}_TEP_SUV.nii.gz"

        # --- 2. VÉRIFICATIONS PRÉALABLES ---
        if out_tep_path.exists() and not args.overwrite:
            print(f"[OK  ] Le fichier SUV existe déjà, on ignore (utiliser --overwrite pour forcer) : {out_tep_path.name}")
            n_exists += 1
            continue

        if not raw_tep_path.exists():
            print(f"[SKIP] TEP Brut introuvable (Plastimatch n'a pas tourné ?) : {raw_tep_path.name}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: Fichier TEP_RAW manquant.\n")
            n_skipped_no_raw += 1
            continue

        # --- 3. CHARGEMENT DE LA GÉOMÉTRIE (VIA LE NIFTI DE PLASTIMATCH) ---
        # On lit l'image NIfTI brute. sitkFloat32 garantit qu'on a la précision requise pour la multiplication mathématique.
        try:
            pet_img = sitk.ReadImage(str(raw_tep_path), sitk.sitkFloat32)
        except Exception as e:
            print(f"[ERREUR] Impossible de lire {raw_tep_path.name}: {e}")
            continue

        # --- 4. RECHERCHE DE LA PHYSIQUE (VIA LES DICOMS) ---
        # On fouille dans le dossier TEP pour trouver le poids de la patiente et l'heure d'injection
        ds, _ = _find_pet_header_with_rph(tep_dicom_dir)

        # Le bouclier Anti-Double Conversion !
        unite_native = str(getattr(ds, "Units", "UNKNOWN")) if ds else "UNKNOWN"
        
        if unite_native == "G/ML":
            print(f"[OK  ] Le PET est DÉJÀ en SUV (Unité: G/ML). Copie directe sans conversion mathématique.")
            import shutil
            shutil.copy2(raw_tep_path, out_tep_path)
            n_converted += 1
            continue # On passe au patient suivant direct !

        # Si ce n'est pas G/ML (ex: BQML), on continue le code normal...
        
        params = None
        
        if ds is not None:
            try:
                params = extract_patient_parameters(ds)
            except Exception:
                params = None

        # Si le DICOM était corrompu ou anonymisé trop brutalement, on tente le CSV
        if params is None:
            row = csv_db.get(subject_id)
            if row:
                params = _row_to_params(row)

        if params is None:
            print(f"[SKIP] Impossible de trouver les métadonnées (Poids/Dose) pour {subject_id}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: Métadonnées manquantes (Pas de RPH dans le DICOM, pas de CSV).\n")
            n_skipped_params += 1
            continue

        # Sécurité : Si la demi-vie du traceur est absente, on assume que c'est du FDG (Fluor-18)
        # La demi-vie physique du Fluor-18 est d'environ 109.77 minutes, soit 6586.2 secondes.
        if not params.get("half_life") or params["half_life"] == 0.0:
            params["half_life"] = 6586.2

        # --- 5. CALCUL DU FACTEUR DE CONVERSION ---
        try:
            factors = compute_suv_factors(params)
            # SUVbw = Standardized Uptake Value par Body Weight (Poids corporel)
            suv_factor = float(factors.get("SUVbw", 0.0))
        except Exception as e:
            print(f"[SKIP] Échec du calcul mathématique du facteur SUV ({e})")
            n_skipped_params += 1
            continue

        # --- 6. APPLICATION ET SAUVEGARDE ---
        try:
            # Magie de SimpleITK : on donne l'image brute, la fonction de ta librairie 
            # la multiplie par suv_factor et sauvegarde le tout !
            write_normalized_image(
                image=pet_img,
                output_path=str(out_tep_path),
                factor=suv_factor,
                log_path=global_log
            )
            print(f"[OK  ] Succès : {out_tep_path.name} généré (Facteur: {suv_factor:.6f})")
            n_converted += 1
        except Exception as e:
            print(f"[ERREUR] Impossible de sauvegarder le NIfTI SUV: {e}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[ERREUR] {subject_id}: Échec de l'écriture du fichier: {e}\n")

    # =================================================================
    # RÉSUMÉ DU RUN
    # =================================================================
    print("\n===== RÉSUMÉ DE LA CONVERSION SUV =====")
    print(f"Patients totaux analysés : {n_total}")
    print(f"Convertis avec succès    : {n_converted}")
    print(f"Ignorés (Pas de RAW)     : {n_skipped_no_raw}")
    print(f"Ignorés (Pas de Params)  : {n_skipped_params}")
    print(f"Ignorés (Déjà existant)  : {n_exists}")

    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n===== Summary =====\n")
        log.write(f"Subjects total         : {n_total}\n")
        log.write(f"Converted (SUVbw)      : {n_converted}\n")
        log.write(f"Skipped (no RAW pet)   : {n_skipped_no_raw}\n")
        log.write(f"Skipped (no params)    : {n_skipped_params}\n")
        log.write(f"Skipped (exists)       : {n_exists}\n")
        log.write("================== SUV CONVERSION END ==================\n")

if __name__ == "__main__":
    main()

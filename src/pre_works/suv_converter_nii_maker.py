import argparse
from pathlib import Path
import os
import csv
import pydicom
import SimpleITK as sitk
import shutil

from utils.suv_conversion import (  
    extract_patient_parameters,   # Extrait dose/poids/demi-vie depuis l'en-tête DICOM
    compute_suv_factors,          # Calcule le facteur mathématique SUVbw
    write_normalized_image        # Multiplie l'image par le facteur et sauvegarde
)

# =====================================================================
# FONCTIONS UTILITAIRES DE RECHERCHE ET FORMATAGE DES MÉTADONNÉES
# (Aucune modification majeure ici, la logique de recherche DICOM reste la même)
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
    """Charge un CSV de secours (Fallback) pour les poids effacés par le PACS."""
    if not csv_path: return {}
    db = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("subject_id", "")).strip()
            if not sid: continue
            db[sid] = row
    return db

def _row_to_params(row: dict):
    """Convertit une ligne brute du CSV de secours en dictionnaire typé."""
    def fget(name, default=""): return "" if row.get(name, default) is None else str(row.get(name, default)).strip()
    def ffloat(s, default=0.0):
        try: return float(s)
        except: return default
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
# FONCTION PRINCIPALE (L'ORCHESTRATEUR SUV LONGITUDINAL)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convertit les NIfTI TEP bruts (Baseline & Suivis) en TEP SUVbw en utilisant les en-têtes DICOM."
    )
    # L'input_root correspondra à notre "PROJET_PETCT_RACINE"
    parser.add_argument(
        "input_root", type=Path, nargs="?", default="Base_PETCT",
        help="Dossier racine contenant les subject_xxx"
    )
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Fichier CSV optionnel de secours pour les paramètres cliniques manquants.")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les fichiers SUV existants.")
    args = parser.parse_args()

    input_root: Path = args.input_root
    
    # Création du fichier de log général
    global_log = os.path.join(input_root, "suv_conversion_log.txt")
    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n================= SUV CONVERSION START (LONGITUDINAL) =================\n")
        log.write(f"Input root: {input_root}\n")

    csv_db = _load_csv_params(args.metadata_csv) if args.metadata_csv else {}

    # Découverte de tous les dossiers patients dans la racine
    subjects = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subjects:
        print(f"[ERREUR] Aucun dossier patient trouvé sous: {input_root}")
        return

    # Initialisation des compteurs globaux
    n_total, n_converted, n_skipped_no_raw, n_skipped_params, n_exists = 0, 0, 0, 0, 0

    print(f"Trouvé {len(subjects)} dossier(s) patient.")
    
    for subj_dir in subjects:
        subject_id = subj_dir.name
        print(f"\n[SUBJ] {subject_id}")

        # --------------------------------------------------------------------
        # NOUVEAUTÉ V4 : RECHERCHE DYNAMIQUE DES VISITES LONGITUDINALES
        # --------------------------------------------------------------------
        # Plutôt que de chercher bêtement "TEP", on liste TOUS les dossiers 
        # du patient qui commencent par "TEP" (TEP, TEP_20230514_1430, etc.)
        tep_folders = [d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("TEP")]
        
        if not tep_folders:
            print(f"  -> Aucun dossier TEP trouvé pour ce patient.")
            continue

        for tep_dicom_dir in tep_folders:
            n_total += 1
            
            # --- 1. ROUTAGE DYNAMIQUE (LE JEU DES CORRESPONDANCES) ---
            # Si le dossier est "TEP" (Baseline), l'image est dans "imgs"
            # Si le dossier est "TEP_20240101" (Suivi), l'image est dans "imgs_20240101"
            if tep_dicom_dir.name == "TEP":
                imgs_dir_name = "imgs"
                visite_label = "Baseline"
            else:
                # On extrait la date de suivi (en retirant "TEP_")
                suffixe = tep_dicom_dir.name.replace("TEP", "") 
                imgs_dir_name = f"imgs{suffixe}"
                visite_label = f"Suivi {suffixe.strip('_')}"
            
            imgs_dir = subj_dir / imgs_dir_name
            print(f"  -> Traitement Visite: {visite_label} (Dossiers: {tep_dicom_dir.name} -> {imgs_dir.name})")

            if not imgs_dir.exists():
                print(f"    [SKIP] Dossier cible {imgs_dir.name} introuvable pour cette visite.")
                n_skipped_no_raw += 1
                continue

            # --- 2. RECHERCHE DU FICHIER _RAW ---
            # L'ingesteur a généré des fichiers du type "ID_TEP_Baseline_A1B2C_RAW.nii.gz"
            # On cherche donc tous les fichiers RAW dans ce dossier cible.
            raw_files = list(imgs_dir.glob("*_TEP_*_RAW.nii.gz"))
            
            if not raw_files:
                print(f"    [SKIP] Aucun fichier TEP Brut (*_RAW.nii.gz) trouvé dans {imgs_dir.name}.")
                with open(global_log, "a", encoding="utf-8") as log:
                    log.write(f"[SKIP] {subject_id} ({visite_label}): Fichier TEP_RAW manquant.\n")
                n_skipped_no_raw += 1
                continue

            # Pour chaque NIfTI RAW trouvé dans cette visite (normalement il y en a 1, mais la boucle sécurise)
            for raw_tep_path in raw_files:
                
                # --- 3. DÉFINITION DU FICHIER DE SORTIE ---
                # On remplace dynamiquement le tag "_RAW.nii.gz" par "_SUV.nii.gz"
                # Ex: "DUKE_TEP_Baseline_A1B2C_RAW.nii.gz" -> "DUKE_TEP_Baseline_A1B2C_SUV.nii.gz"
                out_tep_name = raw_tep_path.name.replace("_RAW.nii.gz", "_SUV.nii.gz")
                out_tep_path = imgs_dir / out_tep_name

                # --- 4. VÉRIFICATIONS PRÉALABLES ---
                if out_tep_path.exists() and not args.overwrite:
                    print(f"    [OK  ] Le fichier SUV existe déjà : {out_tep_path.name}")
                    n_exists += 1
                    continue

                # --- 5. CHARGEMENT DE LA GÉOMÉTRIE (VIA LE NIFTI BRUT) ---
                try:
                    pet_img = sitk.ReadImage(str(raw_tep_path), sitk.sitkFloat32)
                except Exception as e:
                    print(f"    [ERREUR] Impossible de lire {raw_tep_path.name}: {e}")
                    continue

                # --- 6. RECHERCHE DE LA PHYSIQUE (VIA LES DICOMS) ---
                # On va lire le dossier TEP de CETTE visite spécifique pour récupérer la bonne dose injectée
                ds, _ = _find_pet_header_with_rph(tep_dicom_dir)

                # Bouclier Anti-Double Conversion (Si l'hôpital a déjà fait le travail)
                SUV_UNITS = {"GML", "G/ML", "SUV", "SUVBW"}

                unite_native = str(getattr(ds, "Units", "UNKNOWN")).upper().strip() if ds else "UNKNOWN"
                
                if unite_native in SUV_UNITS:
                    print(f"    [OK] PET déjà normalisé SUV ({unite_native}) -> copie directe.")
                    shutil.copy2(raw_tep_path, out_tep_path)
                    n_converted += 1
                    continue

                # Extraction des paramètres
                params = None
                if ds is not None:
                    try: params = extract_patient_parameters(ds)
                    except Exception: params = None

                # Fallback CSV si DICOM vide
                if params is None:
                    row = csv_db.get(subject_id)
                    if row: params = _row_to_params(row)

                if params is None:
                    print(f"    [SKIP] Impossible de trouver les métadonnées (Poids/Dose) pour {raw_tep_path.name}")
                    with open(global_log, "a", encoding="utf-8") as log:
                        log.write(f"[SKIP] {subject_id} ({visite_label}): Métadonnées manquantes.\n")
                    n_skipped_params += 1
                    continue

                # Sécurité : Si demi-vie absente, on assume FDG (Fluor-18)
                if not params.get("half_life") or params["half_life"] == 0.0:
                    params["half_life"] = 6586.2

                # --- 7. CALCUL DU FACTEUR DE CONVERSION ---
                try:
                    factors = compute_suv_factors(params)
                    suv_factor = float(factors.get("SUVbw", 0.0))
                except Exception as e:
                    print(f"    [SKIP] Échec du calcul mathématique SUV ({e})")
                    n_skipped_params += 1
                    continue

                # --- 8. APPLICATION ET SAUVEGARDE ---
                try:
                    write_normalized_image(
                        image=pet_img,
                        output_path=str(out_tep_path),
                        factor=suv_factor,
                        log_path=global_log
                    )
                    print(f"    [OK  ] Succès : {out_tep_path.name} généré (Facteur: {suv_factor:.6f})")
                    n_converted += 1
                except Exception as e:
                    print(f"    [ERREUR] Impossible de sauvegarder le NIfTI SUV: {e}")
                    with open(global_log, "a", encoding="utf-8") as log:
                        log.write(f"[ERREUR] {subject_id} ({visite_label}): Échec écriture: {e}\n")

    # =================================================================
    # RÉSUMÉ DU RUN
    # =================================================================
    print("\n===== RÉSUMÉ DE LA CONVERSION SUV (LONGITUDINALE) =====")
    print(f"Acquisitions totales analysées : {n_total}")
    print(f"Convertis avec succès        : {n_converted}")
    print(f"Ignorés (Pas de RAW)         : {n_skipped_no_raw}")
    print(f"Ignorés (Pas de Params)      : {n_skipped_params}")
    print(f"Ignorés (Déjà existant)      : {n_exists}")

    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n===== Summary =====\n")
        log.write(f"Acquisitions total     : {n_total}\n")
        log.write(f"Converted (SUVbw)      : {n_converted}\n")
        log.write(f"Skipped (no RAW pet)   : {n_skipped_no_raw}\n")
        log.write(f"Skipped (no params)    : {n_skipped_params}\n")
        log.write(f"Skipped (exists)       : {n_exists}\n")
        log.write("================== SUV CONVERSION END ==================\n")

if __name__ == "__main__":
    main()

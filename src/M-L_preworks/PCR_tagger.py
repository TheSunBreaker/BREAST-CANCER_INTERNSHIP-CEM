#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal pCR Label Tagger
---------------------------
Ce script centralise l'ajout de la variable cible 'pcrstatus' (vérité terrain) 
aux fichiers de radiomiques. Il lit la clinique une seule fois, et peut traiter 
le fichier PET/CT et le fichier IRM simultanément si les deux sont fournis.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, List
import pandas as pd

# =====================================================================
# 1. UTILITAIRES DE LECTURE (Robustesse)
# =====================================================================


def sniff_delimiter(path: Path, fallback: str = ",") -> str:
    """
    Tente de deviner si le CSV utilise des virgules, des points-virgules, etc.
    C'est crucial car les exports Excel français utilisent souvent le point-virgule.
    """
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return fallback

def load_file_robust(path: Path, prefer: Optional[str] = None) -> pd.DataFrame:
    """
    Charge un fichier CSV ou EXCEL de manière sécurisée en tentant plusieurs séparateurs.
    """
    # 1. Gérer nativement les fichiers Excel
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    # 2. Gérer les CSV
    if prefer:
        try:
            return pd.read_csv(path, sep=prefer)
        except Exception:
            pass

    delim = sniff_delimiter(path, fallback=",")
    try:
        return pd.read_csv(path, sep=delim, engine="python")
    except Exception:
        # Fallback de la dernière chance avec la virgule standard
        return pd.read_csv(path)
        

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes : tout en minuscules, sans espaces.
    Gère aussi l'harmonisation des identifiants et sécurise leur typage.
    """
    df = df.copy()
    
    # Nettoyage classique
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Remplacement dynamique
    rename_map = {}
    if "case_id" in df.columns:
        rename_map["case_id"] = "subject_id"
    if "patient_id" in df.columns:
        rename_map["patient_id"] = "subject_id"
        
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        
    # SÉCURITÉ ABSOLUE : Forcer l'ID en texte (string) pour éviter les échecs de merge
    # On retire aussi les éventuels espaces ou zéros initiaux inutiles
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
        # Optionnel : si nos ID contiennent des décimales bizarres dues à l'import (ex: '123.0')
        df["subject_id"] = df["subject_id"].str.replace(r'\.0$', '', regex=True)
        
    return df

# =====================================================================
# 2. LOGIQUE MÉTIER DE FUSION
# =====================================================================

def process_modality(
    features_csv: Path, 
    output_csv: Path, 
    clinical_map: pd.DataFrame, 
    modality_name: str
):
    """
    Fonction réutilisable qui applique le pCR à une modalité spécifique (PET/CT ou IRM).
    """
    print(f"\n--- Traitement de la modalité : {modality_name} ---")
    
    # 1. Chargement et nettoyage des radiomiques
    features_df = load_file_robust(features_csv)
    features_df = normalize_columns(features_df)
    
    # Vérification de sécurité stricte
    if "subject_id" not in features_df.columns:
        print(f"[ERREUR] Impossible de trouver l'identifiant patient dans {modality_name}.")
        print("Colonnes trouvées :", list(features_df.columns))
        return

    # 2. Le Merge (Jointure gauche)
    # On colle la colonne pcrstatus à côté de subject_id.
    merged = features_df.merge(clinical_map, on="subject_id", how="left")

    # 3. Calcul des statistiques pour le rapport
    initial_n = len(features_df)
    assigned_n = merged["pcrstatus"].notna().sum()
    skipped_n = initial_n - assigned_n
    
    # On identifie exactement qui a sauté (pratique pour débugger des erreurs de frappe)
    skipped_subjects = (
        merged.loc[merged["pcrstatus"].isna(), "subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    # 4. Nettoyage final : on jette les patients qui n'ont pas de vérité terrain
    # (Un patient sans pCR ne sert à rien pour l'entraînement IA)
    merged = merged.dropna(subset=["pcrstatus"])

    # 5. Sauvegardes
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    
    # Double sauvegarde en Excel pour pouvoir l'ouvrir facilement à la main
    output_excel = output_csv.with_suffix(".xlsx")
    merged.to_excel(output_excel, index=False, sheet_name="Features_with_pCR")

    # 6. Affichage du compte-rendu
    print(f" Enregistré : {output_csv}")
    print(f" Sujets initiaux : {initial_n}")
    print(f" pCR assignés    : {assigned_n}")
    print(f" Ignorés (sans pCR) : {skipped_n}")
    if skipped_subjects:
        print(f" Sujets ignorés : {', '.join(skipped_subjects)}")


# =====================================================================
# 3. CHEF D'ORCHESTRE
# =====================================================================

def main():
    # Définition des arguments du terminal
    parser = argparse.ArgumentParser(
        description="Injecte le 'pcrstatus' clinique dans les CSV de radiomiques (PET/CT et/ou IRM)."
    )
    
    # Le fichier clinique est obligatoire (c'est lui qui contient la vérité terrain)
    parser.add_argument("--clinical", type=Path, required=True, 
                        help="Chemin vers le fichier des caractéristiques cliniques (contenant pcrstatus).")
    
    # Fichiers d'entrée optionnels
    parser.add_argument("--petct", type=Path, default=None, 
                        help="Chemin vers le fichier de radiomiques PET/CT brut.")
    parser.add_argument("--mri", type=Path, default=None, 
                        help="Chemin vers le fichier de radiomiques IRM aplati (Flattened).")
    
    # Fichiers de sortie optionnels
    parser.add_argument("--out-petct", type=Path, default=None, 
                        help="Chemin de sortie pour le PET/CT mergé.")
    parser.add_argument("--out-mri", type=Path, default=None, 
                        help="Chemin de sortie pour l'IRM mergée.")
    
    args = parser.parse_args()

    # Vérification : il faut au moins un fichier à traiter !
    if not args.petct and not args.mri:
        print("[ERREUR] Vous devez spécifier au moins --petct ou --mri à traiter.")
        return

    # --- 1. Chargement du dictionnaire clinique (La source de vérité) ---
    print("\n[INFO] Chargement du fichier Clinique...")
    clinical_df = load_file_robust(args.clinical, prefer=";")
    clinical_df = normalize_columns(clinical_df)

    if "subject_id" not in clinical_df.columns or "pcrstatus" not in clinical_df.columns:
        raise KeyError("Le fichier clinique DOIT contenir 'subject_id' (ou patient_id) et 'pcrstatus'.")

    # Création du "dictionnaire" de référence. 
    # drop_duplicagarantit qu'un patient n'est listé qu'une seule fois.
    clinical_map = (
        clinical_df[["subject_id", "pcrstatus"]]
        .drop_duplicates(subset=["subject_id"], keep="last")
    )

    # --- 2. Traitement du PET/CT (Si demandé) ---
    if args.petct:
        # Si l'utilisateur n'a pas donné de nom de sortie, on en génère un automatiquement
        out_path = args.out_petct if args.out_petct else args.petct.parent / "PET_CT_features_with_pcr.csv"
        process_modality(args.petct, out_path, clinical_map, modality_name="PET/CT")

    # --- 3. Traitement de l'IRM (Si demandé) ---
    if args.mri:
        out_path = args.out_mri if args.out_mri else args.mri.parent / "MRI_features_with_pcr.csv"
        process_modality(args.mri, out_path, clinical_map, modality_name="IRM (DCE)")

    print("\n=== Terminé avec succès ===")

if __name__ == "__main__":
    main()

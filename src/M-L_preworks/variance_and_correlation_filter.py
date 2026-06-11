#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radiomics Pre-filter Tool
-------------------------
Ce script nettoie un dataset de radiomique avant l'étape de Machine Learning.
Il effectue deux opérations :
1. Suppression des variables à variance nulle (constantes).
2. Suppression des variables fortement corrélées (seuil paramétrable).

Nouveautés : Support CSV/Excel, génération d'un rapport TXT détaillé, et CLI.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

def prefilter_radiomics(input_path: Path, output_dir: Path, corr_threshold: float = 0.95):
    print(f"[INFO] Chargement des données depuis : {input_path}")
    
    # ==========================================
    # 1. LECTURE ROBUSTE (CSV ou EXCEL)
    # ==========================================
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        # Tente de lire un CSV (fallback par défaut)
        df = pd.read_csv(input_path)
        
    # On met de côté les colonnes d'identification et la cible
    metadata_cols = ['subject_id', 'pcrstatus', 'patient_id', 'label'] 
    present_metadata = [col for col in metadata_cols if col in df.columns]
    
    df_meta = df[present_metadata]
    df_features = df.drop(columns=present_metadata)
    
    initial_features_count = df_features.shape[1]
    initial_columns = df_features.columns.tolist()
    print(f"[INFO] Nombre de caractéristiques initiales (hors métadonnées) : {initial_features_count}")

    # ==========================================
    # 2. FILTRE DE VARIANCE NULLE
    # ==========================================
    print("\n--- Étape 1 : Filtre de Variance ---")
    df_features_num = df_features.select_dtypes(include=[np.number])
    
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_features_num)
    
    # Identification des variables conservées et supprimées
    kept_vars_mask = selector.get_support()
    features_kept_var = df_features_num.columns[kept_vars_mask]
    
    # Création de la liste des variables évincées à cette étape
    dropped_by_variance = [col for i, col in enumerate(df_features_num.columns) if not kept_vars_mask[i]]
    
    df_features = df_features_num[features_kept_var]
    
    print(f"-> Supprimées (constantes) : {len(dropped_by_variance)}")
    print(f"-> Restantes : {df_features.shape[1]}")

    # ==========================================
    # 3. FILTRE DE CORRÉLATION
    # ==========================================
    print(f"\n--- Étape 2 : Filtre de Corrélation (> {corr_threshold}) ---")
    print("Calcul de la matrice de corrélation (Pearson)...")
    
    corr_matrix = df_features.corr().abs()

    # Triangle supérieur pour éviter de supprimer les 2 variables d'une même paire
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identification des colonnes à supprimer
    dropped_by_correlation = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    
    df_features = df_features.drop(columns=dropped_by_correlation)
    
    print(f"-> Supprimées (redondantes) : {len(dropped_by_correlation)}")
    print(f"-> Caractéristiques finales retenues : {df_features.shape[1]}")

    # ==========================================
    # 4. RECONSTRUCTION ET DOUBLE SAUVEGARDE
    # ==========================================
    print("\n--- Sauvegarde des données ---")
    final_df = pd.concat([df_meta, df_features], axis=1)
    
    # Création du dossier de sortie si nécessaire
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construction des noms de fichiers de sortie basés sur le nom d'entrée
    base_name = input_path.stem + "_filtered"
    csv_out = output_dir / f"{base_name}.csv"
    excel_out = output_dir / f"{base_name}.xlsx"
    txt_out = output_dir / f"{base_name}_report.txt"

    # Exportation systématique dans les deux formats
    final_df.to_csv(csv_out, index=False)
    final_df.to_excel(excel_out, index=False)
    print(f"[OK] Fichier CSV généré   : {csv_out}")
    print(f"[OK] Fichier Excel généré : {excel_out}")

    # ==========================================
    # 5. GÉNÉRATION DU RAPPORT TEXTE
    # ==========================================
    print("\n--- Génération du rapport ---")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("      RAPPORT DE PRÉFILTRAGE RADIOMIQUE   \n")
        f.write("=========================================\n\n")
        f.write(f"Fichier source : {input_path.name}\n")
        f.write(f"Seuil de corrélation : {corr_threshold}\n\n")
        
        f.write(f"Variables initiales (hors métadonnées) : {initial_features_count}\n")
        f.write(f"Variables finales conservées           : {df_features.shape[1]}\n")
        f.write(f"Total des variables supprimées         : {len(dropped_by_variance) + len(dropped_by_correlation)}\n")
        f.write("-----------------------------------------\n\n")
        
        f.write(f"[1] ÉVINCÉES POUR VARIANCE NULLE ({len(dropped_by_variance)} variables)\n")
        f.write("Raison : Ces variables contiennent la même valeur pour tous les patients, elles n'apportent aucun pouvoir prédictif.\n")
        if dropped_by_variance:
            for var in dropped_by_variance:
                f.write(f"    - {var}\n")
        else:
            f.write("    (Aucune)\n")
            
        f.write(f"\n[2] ÉVINCÉES POUR HAUTE CORRÉLATION ({len(dropped_by_correlation)} variables)\n")
        f.write(f"Raison : Ces variables ont un coefficient de corrélation de Pearson > {corr_threshold} avec une autre variable déjà conservée (redondance d'information).\n")
        if dropped_by_correlation:
            for var in dropped_by_correlation:
                f.write(f"    - {var}\n")
        else:
            f.write("    (Aucune)\n")

    print(f"[OK] Rapport détaillé généré : {txt_out}")

# ==========================================
# GESTION DE LA LIGNE DE COMMANDE (CLI)
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Filtre les variables de radiomique (variance nulle et haute corrélation)."
    )
    
    # Arguments obligatoires et optionnels
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Chemin vers le fichier d'entrée (.csv ou .xlsx).")
                        
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."),
                        help="Dossier de sortie (par défaut : dossier courant).")
                        
    parser.add_argument("-t", "--threshold", type=float, default=0.95,
                        help="Seuil de corrélation de Pearson (défaut : 0.95).")

    args = parser.parse_args()

    # Vérification de l'existence du fichier
    if not args.input.exists():
        print(f"[ERREUR] Le fichier d'entrée '{args.input}' est introuvable.")
        return

    # Lancement du pipeline
    prefilter_radiomics(args.input, args.outdir, args.threshold)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def prefilter_radiomics(input_csv_path, output_csv_path, corr_threshold=0.95):
    """
    Nettoie un dataset de radiomique avant le Machine Learning.
    1. Supprime les variables à variance nulle.
    2. Supprime les variables fortement corrélées.
    """
    print(f"Chargement des données depuis {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # On met de côté les colonnes d'identification et la cible (si présente)
    metadata_cols = ['subject_id', 'pcrstatus'] 
    present_metadata = [col for col in metadata_cols if col in df.columns]
    
    df_meta = df[present_metadata]
    df_features = df.drop(columns=present_metadata)
    
    initial_features = df_features.shape[1]
    print(f"Nombre de caractéristiques initiales : {initial_features}")

    # 1. Filtre de Variance Nulle
    print("\n--- Étape 1 : Filtre de Variance ---")
    # Conserve uniquement les colonnes numériques pour éviter les crashs
    df_features_num = df_features.select_dtypes(include=[np.number])
    
    # On utilise VarianceThreshold (0 = variance nulle, càd constante)
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df_features_num)
    
    # Récupération du nom des colonnes conservées
    features_kept = df_features_num.columns[selector.get_support()]
    df_features = df_features_num[features_kept]
    
    print(f"Caractéristiques supprimées (constantes) : {initial_features - len(features_kept)}")
    print(f"Restantes : {df_features.shape[1]}")

    # 2. Filtre de Corrélation
    print(f"\n--- Étape 2 : Filtre de Corrélation (> {corr_threshold}) ---")
    print("Calcul de la matrice de corrélation (Pearson)... Cela peut prendre un moment.")
    
    # Calcul de la matrice de corrélation absolue
    corr_matrix = df_features.corr().abs()

    # On ne garde que le triangle supérieur de la matrice pour éviter de supprimer les 2 variables d'une même paire
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # On trouve les colonnes dont au moins une corrélation dépasse le seuil
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    
    # On supprime ces colonnes
    df_features = df_features.drop(columns=to_drop)
    
    print(f"Caractéristiques supprimées (redondantes) : {len(to_drop)}")
    print(f"Caractéristiques finales retenues : {df_features.shape[1]}")

    # 3. Reconstruction et Sauvegarde
    print("\n--- Sauvegarde ---")
    final_df = pd.concat([df_meta, df_features], axis=1)
    final_df.to_csv(output_csv_path, index=False)
    print(f"Dataset allégé sauvegardé sous : {output_csv_path}")

# --- Utilisation ---
# prefilter_radiomics("data_brute_PETCT.csv", "data_prefiltree_PETCT.csv", corr_threshold=0.95)

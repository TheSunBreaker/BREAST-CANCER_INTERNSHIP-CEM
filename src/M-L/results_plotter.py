#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting Script for Nested CV Results
-------------------------------------
Ce script lit les fichiers CSV générés par le pipeline de Machine Learning
et produit des visualisations prêtes pour des publications ou des présentations.

Fonctionnalités :
1. Heatmaps de Matrice de Confusion (Agrégées sur les Folds)
2. Barplots des performances (ROC-AUC) par modèle et modalité
3. Barplots des Top-N Variables les plus influentes pour le meilleur modèle

Utilisation en ligne de commande :
    python plot_results.py -i ./_modality_outputs -o ./_plots
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================================
# CONFIGURATION VISUELLE (Esthétique des graphiques)
# =====================================================================
# Applique un style clair et professionnel adapté aux présentations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

# =====================================================================
# FONCTIONS DE TRACÉ
# =====================================================================

def plot_model_comparison(summary_df: pd.DataFrame, outdir: Path):
    """
    Génère un graphique en barres comparant l'AUC-ROC moyen de chaque modèle.
    """
    print("[PLOT] Génération de la comparaison des modèles (ROC-AUC)...")
    
    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(
        data=summary_df, 
        x="set", 
        y="roc_auc_mean", 
        hue="model"
    )
    
    plt.title("Comparaison des performances (ROC-AUC) par Modalité et Modèle", fontsize=16, pad=20)
    plt.xlabel("Combinaison de Modalités", fontsize=14)
    plt.ylabel("ROC-AUC Moyen (Nested CV)", fontsize=14)
    
    # Ligne de référence à 0.5 (hasard)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.legend(title="Modèles ML", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.0, 1.05) 
    plt.tight_layout()
    
    save_path = outdir / "ROC_AUC_Comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Sauvegardé : {save_path}")

def plot_confusion_matrices(summary_df: pd.DataFrame, outdir: Path, top_n: int = 5):
    """
    Génère des heatmaps de matrice de confusion pour les N meilleurs modèles.
    """
    print(f"[PLOT] Génération des {top_n} meilleures Matrices de Confusion...")
    
    if "roc_auc_mean" not in summary_df.columns:
        print("[ERREUR] La colonne 'roc_auc_mean' est absente. Matrice annulée.")
        return
        
    top_models = summary_df.sort_values(by="roc_auc_mean", ascending=False).head(top_n)
    
    cm_dir = outdir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in top_models.iterrows():
        set_name = row["set"]
        model = row["model"]
        auc = row["roc_auc_mean"]
        
        tn = int(row.get("tn_sum", 0))
        fp = int(row.get("fp_sum", 0))
        fn = int(row.get("fn_sum", 0))
        tp = int(row.get("tp_sum", 0))
        
        cm = np.array([[tn, fp], 
                       [fn, tp]])
        
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Non-pCR (Prédit)", "pCR (Prédit)"],
                    yticklabels=["Non-pCR (Réel)", "pCR (Réel)"],
                    annot_kws={"size": 16, "weight": "bold"})
                    
        plt.title(f"Matrice de Confusion (CV cumulée)\n{set_name} | {model} (AUC: {auc:.2f})", fontsize=14, pad=15)
        plt.tight_layout()
        
        safe_set_name = set_name.replace("+", "_")
        save_path = cm_dir / f"CM_Top{idx+1}_{safe_set_name}_{model}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"  -> {top_n} matrices sauvegardées dans : {cm_dir}")

def plot_feature_importance(summary_df: pd.DataFrame, indir: Path, outdir: Path, top_n_features: int = 15):
    """
    Trouve le meilleur modèle global, lit son fichier de variables sélectionnées,
    et trace un graphique horizontal des variables les plus influentes.
    """
    print(f"[PLOT] Génération du graphique des {top_n_features} Top Variables...")

    if "roc_auc_mean" not in summary_df.columns:
        return

    # 1. Identifier le meilleur modèle
    best_model_row = summary_df.sort_values(by="roc_auc_mean", ascending=False).iloc[0]
    set_name = best_model_row["set"]
    model = best_model_row["model"]
    
    # 2. Chercher le fichier _selected_features correspondant dans le dossier des résultats
    feature_file = indir / f"{set_name}__{model}_selected_features.csv"
    
    if not feature_file.exists():
        print(f"  [WARN] Fichier de features introuvable : {feature_file}")
        return
        
    df_feat = pd.read_csv(feature_file)
    
    if "weight_or_importance" not in df_feat.columns:
        print(f"  [WARN] Pas de poids/importance trouvés dans {feature_file}. Plot annulé.")
        return
        
    # 3. Traitement : Tri par valeur absolue de l'importance
    df_feat["abs_importance"] = df_feat["weight_or_importance"].abs()
    df_feat = df_feat.sort_values(by="abs_importance", ascending=False).head(top_n_features)
    
    # 4. Tracé du graphique
    plt.figure(figsize=(10, 8))
    
    # On utilise un barplot horizontal pour avoir la place d'écrire les longs noms des radiomiques
    sns.barplot(
        data=df_feat, 
        x="weight_or_importance", 
        y="feature", 
        hue="feature",       # Assign hue to silence seaborn warning
        palette="viridis",   # Palette scientifique adaptée aux daltoniens
        legend=False
    )
    
    plt.title(f"Top {top_n_features} Variables les plus influentes\nMeilleur Modèle : {set_name} | {model}", fontsize=14, pad=15)
    plt.xlabel("Poids (Régression) ou Importance (Arbres)", fontsize=12)
    plt.ylabel("Nom de la Caractéristique", fontsize=12)
    
    # Ajouter une ligne verticale à 0 pour bien séparer les poids positifs et négatifs
    plt.axvline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    
    # 5. Sauvegarde
    feat_dir = outdir / "feature_importance"
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    safe_set_name = set_name.replace("+", "_")
    save_path = feat_dir / f"Feature_Importance_{safe_set_name}_{model}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  -> Graphique sauvegardé : {save_path}")

# =====================================================================
# FONCTION PRINCIPALE & PARSER CLI
# =====================================================================

def main(input_dir: str, output_dir: str):
    """
    Chef d'orchestre du script de visualisation.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = in_path / "summary_metrics.csv"
    
    if not summary_file.exists():
        print(f"Erreur : Le fichier {summary_file} est introuvable.")
        print("Avez-vous bien lancé le script de Machine Learning en premier ?")
        return

    summary_df = pd.read_csv(summary_file)
    
    print("\n=== DÉBUT DE LA GÉNÉRATION DES GRAPHIQUES ===")
    
    plot_model_comparison(summary_df, out_path)
    plot_confusion_matrices(summary_df, out_path, top_n=5)
    plot_feature_importance(summary_df, in_path, out_path, top_n_features=15)
    
    print("\nTous les graphiques ont été générés avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de graphiques pour les résultats de Radiomique M-L.")
    
    parser.add_argument("-i", "--input", type=str, default="./_modality_outputs",
                        help="Dossier contenant les CSV générés par le ML (défaut: ./_modality_outputs)")
    
    parser.add_argument("-o", "--output", type=str, default="./_plots",
                        help="Dossier où sauvegarder les images générées (défaut: ./_plots)")
    
    args = parser.parse_args()
    
    main(args.input, args.output)

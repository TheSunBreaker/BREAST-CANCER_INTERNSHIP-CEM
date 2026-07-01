#!/usr/bin/env python3
"""
Nested CV comparison across modalities (and their fusions) with multiple classifiers.

Models compared per set:
- LogisticRegression
- SVC (linear & RBF, probability=True)
- RandomForestClassifier
- MLPClassifier (small, early_stopping)

Preprocessing (inside folds): median impute -> StandardScaler -> SelectKBest(mutual_info_classif)
Fusion sets use ID intersection; fused feature names are prefixed per modality.

Outputs in ./_modality_outputs:
  - summary_metrics.csv  (one row per set x model, mean±sd of ROC-AUC, PR-AUC, BalAcc, F1)
  - <set>__<model>_outercv_metrics.csv  (per outer-fold metrics + best params)
  - <set>__<model>_selected_features.csv (optional, refit on all data with representative best params)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier # Très pratique pour implémenter ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from boruta import BorutaPy # Boruta

import joblib # Pour sauvegarder le meilleur modèle trouvé

# ----------------- CONFIG -----------------
FILES = {
    "Clinical": r"clinicals.xlsx",
    "MRI": r"radiomics_dce.xlsx",
    "PETCT": r"radiomics_pet-ct.xlsx",
}
POSSIBLE_LABELS = ["pcr", "pcrstatus", "label"]
POSSIBLE_IDS    = ["subject_id", "patient_id", "id"]

# Small-N friendly CV sizes
OUTER_N_SPLITS = 3
INNER_N_SPLITS = 3
RANDOM_STATE   = 42

# --- MAINTENANT ---
# Feature-count grid pour la réduction douce (SelectKBest) avant l'Elastic Net
K_GRID         = [50, 100, 200, 300]

# --- DÉFINITION DES SÉLECTEURS ---
# 1. Sélecteur Linéaire (ElasticNet robuste)
selector_elasticnet = SelectFromModel(
    LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=2000, random_state=42) # 0.5 veut dire fifty fifty entre régulations L1 et L2
)
# 2. Sélecteur Arbre (Rapide et Robuste)
selector_extratrees = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100, random_state=42)
)
# 3. Boruta (Très lourd, très très lourd, donc à utiliser seulement si beaucoup de coeurs. Dans mon cas, sera testé sur serveur à 48 coeurs)
rf_boruta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1) # n_jobs à -1 pour dire on parrallélise sur tous les coeurs disponibles
selector_boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=42)

# Per-model param grids (small & safe for tiny cohorts)
GRIDS = {

    "ET": {
        "kbest_soft__k": K_GRID,
        "multivariate_selector": ["passthrough", selector_extratrees, selector_boruta], # <-- Sélecteurs Non-Linéaire !
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 3, 5],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__class_weight": ["balanced"],
        "clf__random_state": [RANDOM_STATE],
    },

    "HGB": {
        "kbest_soft__k": K_GRID,
        "multivariate_selector": ["passthrough", selector_extratrees, selector_boruta], # <-- Sélecteurs Non-Linéaire !
        "clf__learning_rate": [0.01, 0.1],
        "clf__max_iter": [100, 200],
        "clf__max_depth": [1, 3, 5],
        "clf__min_samples_leaf": [2, 5],
        "clf__random_state": [RANDOM_STATE],
        # HistGradientBoosting ne prend pas 'class_weight="balanced"' nativement dans sklearn, 
        # mais on gère le déséquilibre via min_samples_leaf pour stabiliser les feuilles.
    },

   "KNN": {
        "kbest_soft__k": K_GRID,
        "multivariate_selector": [selector_extratrees, selector_boruta], # <-- Sélecteurs Non-Linéaire !
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
  
    "LR": {
        "kbest_soft__k": K_GRID, # Renommé pour correspondre au pipeline
        "multivariate_selector": ["passthrough", selector_elasticnet], # Ligne droite avec ligne droite
        "clf__C":   [0.1, 1.0, 3.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear"],
        "clf__class_weight": ["balanced"],
    },
    "SVM": {
        "kbest_soft__k": K_GRID, # Renommé pour correspondre au pipeline
        "multivariate_selector": ["passthrough", selector_elasticnet],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__gamma": ["scale"],   # for rbf
        "clf__class_weight": ["balanced"],
        "clf__probability": [True],
    },
    "RF": {
        "kbest_soft__k": K_GRID, # Renommé pour correspondre au pipeline
        "multivariate_selector": ["passthrough", selector_extratrees, selector_boruta], # Le serveur va transpirer en tout cas de tous ses coeurs
        "clf__n_estimators": [200],
        "clf__max_depth": [None, 3, 5],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__class_weight": ["balanced"],
        "clf__random_state": [RANDOM_STATE],
    },
    "MLP": {
        "kbest_soft__k": K_GRID, # Renommé pour correspondre au pipeline
        "multivariate_selector": [selector_extratrees, selector_boruta],
        "clf__hidden_layer_sizes": [(16,), (32,), (32,16)],
        "clf__alpha": [1e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 3e-3],
        # let us disable early_stopping on tiny folds
        "clf__early_stopping": [False, True],
        # give the model a decent budget
        "clf__max_iter": [400],
        "clf__random_state": [RANDOM_STATE],
        # only used when early_stopping=True
        "clf__validation_fraction": [0.2],
        "clf__n_iter_no_change": [10],
    },
}

# Which sets (singles + fusions) to run
SETS_TO_RUN = [
    ("Clinical",), ("MRI",), ("PETCT",),
    ("Clinical","MRI"),
    ("Clinical","PETCT"),
    ("MRI","PETCT"),
    ("Clinical","MRI","PETCT"),
]

# Write feature lists (refit on all data with representative best params)
WRITE_FEATURE_LISTS = True

SAVE_DIR = Path("./_modality_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------------------


# ----------------- IO & BUILDERS -----------------


@dataclass
class LabeledFrame:
    X: pd.DataFrame
    y: pd.Series
    id_series: Optional[pd.Series]
    y_name: str
    id_name: Optional[str]

POSSIBLE_LABELS = ["pcr", "pcrstatus", "label"]
POSSIBLE_IDS    = ["subject_id", "patient_id", "id"]

# Classe pour la mise en place du premier filtre (statistique : Partie Corrélation)
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.support_ = None
        
    def fit(self, X, y=None):
        # Convertir en DataFrame pour calculer la corrélation facilement
        df = pd.DataFrame(X) # Marche même si X est un array NumPy du StandardScaler
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # Trouver les colonnes à supprimer
        to_drop = [c for c in upper.columns if any(upper[c] > self.threshold)]

        # Créer le masque booléen (True = on garde, False = on jette)
        self.support_ = np.ones(df.shape[1], dtype=bool)
        self.support_[to_drop] = False
        return self
        
    def transform(self, X):
        # Appliquer le masque (gère NumPy array et DataFrame)
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.support_]
        return X[:, self.support_]
        
    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_

def _find_column(df: pd.DataFrame, candidates):
    lower = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower:
            return lower[key]
    for col in df.columns:
        cl = col.lower()
        if any(re.fullmatch(rf".*{re.escape(k)}.*", cl) for k in candidates):
            return col
    return None

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        return pd.read_excel(p)
    except Exception:
        return pd.read_csv(p)

def _clean_ids(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    bad = s.isna() | (s == "") | (s == "0") | (s == "NAN") | (s == "NONE") | (s == "NULL")
    s[bad] = np.nan
    return s

def _normalize_pcr(series: pd.Series) -> pd.Series:
    s = series.copy()

    # try numeric first
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.where(s_num.isin([0,1]), s_num, np.nan), index=s.index, dtype="float")

    s_str = s[~s_num.isin([0,1])].astype(str).str.strip().str.lower()

    mapping = {
        "1":1, "0":0,
        "pcr":1, "pcr (yes)":1, "yes":1, "true":1, "y":1, "complete":1,
        "npcr":0, "non pcr":0, "no pcr":0, "no":0, "false":0, "n":0, "partial":0,
    }
    mapped = s_str.map(mapping)
    # regex fallback
    mapped = mapped.where(~mapped.isna(),
                          np.where(s_str.str.contains(r"\bpcr\b"), 1,
                          np.where(s_str.str.contains(r"\b(npcr|no\s*pcr|non\s*pcr)\b"), 0, np.nan)))
    out.loc[mapped.index] = out.loc[mapped.index].fillna(mapped)

    return out  # float with 0/1/NaN

def load_single(name: str, path: str):
    df = _load(path)
    orig_shape = df.shape

    y_col = _find_column(df, POSSIBLE_LABELS)
    if y_col is None:
        raise ValueError(f"[{name}] label column not found (looked for {POSSIBLE_LABELS}).")

    id_col = _find_column(df, POSSIBLE_IDS)
    if id_col is not None:
        df[id_col] = _clean_ids(df[id_col])
        before = len(df)
        df = df[df[id_col].notna()].reset_index(drop=True)
        after = len(df)
        if after < before:
            print(f"[CLEAN] [{name}] dropped {before-after} rows with bad IDs.")
    else:
        print(f"[WARN] [{name}] no ID column found (looked for {POSSIBLE_IDS}).")

    # Normalize label
    y_norm = _normalize_pcr(df[y_col])
    bad_lab = y_norm.isna()
    if bad_lab.any():
        print(f"[CLEAN] [{name}] dropping {bad_lab.sum()} rows with invalid labels in '{y_col}'.")
        print("Examples:", df.loc[bad_lab, y_col].astype(str).unique()[:5])
    df = df.loc[~bad_lab].reset_index(drop=True)
    y  = y_norm.loc[~bad_lab].astype(int).reset_index(drop=True)

    # Features (numeric coercion)
    exclude = [y_col] + ([id_col] if id_col else [])
    X = df.drop(columns=exclude, errors="ignore").copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    ids = df[id_col] if id_col is not None else None

    print(f"[INFO] [{name}] original shape: {orig_shape}, after cleaning: {df.shape}")

    return LabeledFrame(X=X, y=y, id_series=ids, y_name=y_col, id_name=id_col)
  
def intersect_and_concatenate(frames: List[Tuple[str, LabeledFrame]]) -> LabeledFrame:
    for name, lf in frames:
        if lf.id_series is None:
            raise ValueError(f"[{name}] has no ID column; cannot fuse.")
    normalized = []
    for name, lf in frames:
        ids = lf.id_series.astype(str).str.strip().str.upper()
        normalized.append((name, lf, ids))
    common = set(normalized[0][2])
    for _, _, ids in normalized[1:]:
        common &= set(ids)
    if not common:
        raise ValueError(f"No ID overlap for: {', '.join(n for n,_ in frames)}")

    aligned = []
    for name, lf, ids in normalized:
        mask = ids.isin(common)
        Xs   = lf.X.loc[mask].copy()
        ys   = lf.y.loc[mask].copy()
        id_s = ids.loc[mask].copy()
        order = np.argsort(id_s.values.astype(object))
        Xs, ys, id_s = Xs.iloc[order].reset_index(drop=True), ys.iloc[order].reset_index(drop=True), id_s.iloc[order].reset_index(drop=True)
        Xs.columns = [f"{name}__{c}" for c in Xs.columns]
        aligned.append((name, Xs, ys, id_s))
    y0 = aligned[0][2]
    for name, _, y_sub, _ in aligned[1:]:
        if not np.array_equal(y0.values, y_sub.values):
            raise ValueError(f"Label mismatch after alignment (check {name}).")
    X_concat = pd.concat([x for _, x, _, _ in aligned], axis=1)
    return LabeledFrame(X=X_concat, y=y0, id_series=aligned[0][3], y_name=frames[0][1].y_name, id_name=frames[0][1].id_name)

def build_set(set_names: Tuple[str,...], singles: Dict[str, LabeledFrame]) -> Tuple[str, LabeledFrame]:
    if len(set_names) == 1:
        return set_names[0], singles[set_names[0]]
    return "+".join(set_names), intersect_and_concatenate([(n, singles[n]) for n in set_names])
# --------------------------------------------------

def base_preproc():
    # C'est ici qu'on construit "l'entonnoir" étape par étape
    return [
        # 1. Gestion des valeurs manquantes
        ("imp", SimpleImputer(strategy="median")),

        # 2. Filtre de variance (avant propos de l'étage 1 du filtre des features)
        ("filter_variance", VarianceThreshold(threshold=0.0)),
      
        # 3. Standardisation (z-score, calculé uniquement sur le fold d'entraînement)
        ("scaler", StandardScaler()),
        
        # 3. ########################### Filtres selecteurs de varaibles ###########################
        ##### Etage 1 :
        # Filtre de corrélation intra-CV
        ("filter_corr", CorrelationFilter(threshold=0.95)),
        ##### Etage 2 :
        # Filtre univarié ("Réduction douce")
        # On garde entre les 100 à 300 meilleures variables pour soulager l'étage suivant.
        # Ce 'k' (50, 100, 200, 300) sera d'ailleurs mis dans notre GridSearch !
        ("kbest_soft", SelectKBest(mutual_info_classif, k=300)), 
        ##### Etage 3 :
        # Sélection Multivariée (Selon modèle) mis à tunnel simple par défaut
        ("multivariate_selector", "passthrough")
    ]

def get_model_and_grid(tag: str):
    """Return (estimator_instance, param_grid) for the given model tag."""
    if tag == "LR":
        clf = LogisticRegression(random_state=RANDOM_STATE)
    elif tag == "ET":
        clf = ExtraTreesClassifier()
    elif tag == "HGB":
        clf = HistGradientBoostingClassifier()
    elif tag == "KNN":
        clf = KNeighborsClassifier()
    elif tag == "SVM":
        clf = SVC(random_state=RANDOM_STATE)
    elif tag == "RF":
        clf = RandomForestClassifier()
    elif tag == "MLP":
        clf = MLPClassifier()
    else:
        raise ValueError(tag)
    return clf, GRIDS[tag]
# ---------------------------------------------------


# ----------------- EVALUATION -----------------
def cap_k_grid(X: pd.DataFrame, y: pd.Series, grid: List[int]) -> List[int]:
    kmax = max(1, min(X.shape[1], len(y) - 1))
    feasible = sorted({k for k in grid if 1 <= k <= kmax})
    return feasible if feasible else [min(1, kmax)]

def nested_cv_once(set_name: str, model_tag: str, X: pd.DataFrame, y: pd.Series,
                   outer_splits: int, inner_splits: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # cap splits by minority class count
    min_class = y.value_counts().min()
    n_outer = max(2, min(outer_splits, int(min_class), len(y)))
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)

    # grid adjusted for k per dataset
    _, base_grid = get_model_and_grid(model_tag)
    grid = dict(base_grid)  # shallow copy
    grid["kbest_soft__k"] = cap_k_grid(X, y, base_grid["kbest_soft__k"])

    rows = []
    fold = 0
    for tr_idx, te_idx in outer.split(X, y):
        fold += 1
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        # inner CV also capped
        min_class_tr = ytr.value_counts().min()
        n_inner = max(2, min(inner_splits, int(min_class_tr), len(ytr)))
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)

        clf, _ = get_model_and_grid(model_tag)
        pipe = Pipeline(base_preproc() + [("clf", clf)])

        # If using MLP and the training fold is tiny, force early_stopping=False
        grid = dict(grid)  # copy the grid you built earlier
        if model_tag == "MLP":
            # Need at least 1 sample per class in MLP’s internal validation set.
            # With very small ytr, that can fail; so disable early stopping for safety.
            # You can be more nuanced (e.g., if len(ytr) * 0.2 < 2*#classes).
            n_classes = ytr.nunique()
            if len(ytr) * 0.2 < 2 * n_classes or min_class_tr < 3:
                grid["clf__early_stopping"] = [False]

        gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=inner,
        refit=True,
        n_jobs=1,
        verbose=1,
        error_score=np.nan,   # <— tolerate occasional bad fits
        )

        gs.fit(Xtr, ytr)

        best = gs.best_estimator_
        proba = _predict_proba_safe(best, Xte)
        pred = (proba >= 0.5).astype(int)

        # --- NOUVEAU : Calcul de la matrice de confusion ---
        # .ravel() aplatit la matrice 2x2 en 4 variables distinctes
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()

        rows.append({
            "set": set_name,
            "model": model_tag,
            "fold": fold,
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "best_params": str(gs.best_params_),
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "bal_acc": float(balanced_accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            # --- NOUVEAU : Sauvegarde des compteurs pour un plot ultérieur plus aisé de la matrice de confusion ---
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["set","model"])
        .agg({
            "roc_auc": ["mean", "std"],
            "pr_auc": ["mean", "std"],
            "bal_acc": ["mean", "std"],
            "f1": ["mean", "std"],
            # NOUVEAU : On fait la SOMME des patients sur tous les folds
            "tn": "sum", 
            "fp": "sum", 
            "fn": "sum", 
            "tp": "sum"
        })
    )
    # Aplatir les multi-index de colonnes proprement
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary.columns]
    summary = summary.reset_index()
                     
    return folds, summary

def _predict_proba_safe(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    # Use predict_proba if available; else decision_function scaled to [0,1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        # min-max scale for AUC-compatible scores
        d = (d - d.min()) / (d.max() - d.min() + 1e-12)
        return d
    # fallback
    preds = model.predict(X)
    return preds.astype(float)

def refit_write_features(set_name: str, model_tag: str, X: pd.DataFrame, y: pd.Series,
                         best_params_example: dict, outdir: Path):
    # Refit on ALL data with a "representative" best param set (from outer CV mode)
    clf, _ = get_model_and_grid(model_tag)
    
    # Sanitize params against current data (cap k if needed)
    k_grid = cap_k_grid(X, y, K_GRID)
    chosen_k = best_params_example.get("kbest_soft__k", k_grid[-1])
    chosen_k = max(1, min(chosen_k, k_grid[-1]))

    # Assemble pipeline & set params
    pipe = Pipeline(base_preproc() + [("clf", clf)])
    # On force la valeur de k, et on injecte le reste des hyperparamètres
    pipe.set_params(**{**best_params_example, "kbest_soft__k": chosen_k})
    pipe.fit(X, y)
                           
    # Sauvegarde du modèle complet (Pipeline + Poids)
    joblib.dump(pipe, outdir / f"{set_name}__{model_tag}_FINAL_MODEL.joblib")
                           
    # --- LOGIQUE D'EXTRACTION À 4 ÉTAGES (Robuste et en cascade) ---
    try:
        # Étage 1 : Filtre de Variance
        mask_var = pipe.named_steps["filter_variance"].get_support()
        feat_after_var = np.array(X.columns)[mask_var]
        
        # Étage 2 : Filtre de Corrélation
        mask_corr = pipe.named_steps["filter_corr"].get_support()
        feat_after_corr = feat_after_var[mask_corr] # On cascade sur le résultat précédent !
        
        # Étage 3 : Variables survivantes au KBest
        mask_kbest = pipe.named_steps["kbest_soft"].get_support()
        feat_after_kbest = feat_after_corr[mask_kbest]
        
        # Étage 4 : Variables survivantes au sélecteur Multivarié (ElasticNet, ET, ou Boruta)
        multivariate_step = pipe.named_steps["multivariate_selector"]
        
        if multivariate_step != "passthrough":
            # Astuce pour gérer BorutaPy qui n'a parfois pas de fonction get_support()
            if hasattr(multivariate_step, 'get_support'):
                mask_multi = multivariate_step.get_support()
            else:
                mask_multi = multivariate_step.support_
                
            selected = list(feat_after_kbest[mask_multi])
        else:
            selected = list(feat_after_kbest)
            
    except Exception as e:
        print(f"[WARN] Erreur d'extraction des features en cascade : {e}")
        selected = list(X.columns)  # Fallback de sécurité
        
    coefs = None
                           
    try:
        coefs = pipe.named_steps["clf"].coef_.ravel()
    except Exception:
        try:
            imp = pipe.named_steps["clf"].feature_importances_
            coefs = imp
        except Exception:
            coefs = None

    if coefs is not None and len(selected) == len(coefs):
        order = np.argsort(np.abs(coefs))[::-1]
        feat_df = pd.DataFrame({"feature": [selected[i] for i in order],
                                "weight_or_importance": [float(coefs[i]) for i in order]})
    else:
        feat_df = pd.DataFrame({"feature": selected})

    feat_df.to_csv(outdir / f"{set_name}__{model_tag}_selected_features.csv", index=False)
# ---------------------------------------------------


def main():
    # Load singles
    singles = {name: load_single(name, path) for name, path in FILES.items()}

    # Build sets
    sets: Dict[str, LabeledFrame] = {}
    for set_names in SETS_TO_RUN:
        try:
            set_name, lf = build_set(set_names, singles)
            sets[set_name] = lf
        except Exception as e:
            print(f"[SKIP] {'+'.join(set_names)} -> {e}")

    all_sum = []
    for set_name, lf in sets.items():
        # Skip too-small sets
        if lf.y.value_counts().min() < 2 or len(lf.y) < 4:
            print(f"[WARN] Skipping {set_name}: too small for CV (n={len(lf.y)}).")
            continue

        for model_tag in ["LR", "ET", "HGB", "KNN", "SVM", "RF", "MLP"]:
            try:
                folds, summ = nested_cv_once(set_name, model_tag, lf.X, lf.y,
                                             OUTER_N_SPLITS, INNER_N_SPLITS)
            except Exception as e:
                print(f"[ERR] {set_name}__{model_tag}: {e}")
                continue

            # write per-fold file
            folds.to_csv(SAVE_DIR / f"{set_name}__{model_tag}_outercv_metrics.csv", index=False)
            all_sum.append(summ)

            # representative best params (mode across folds)
            try:
                best_series = folds["best_params"].mode().iloc[0]
                best_params = eval(best_series) if isinstance(best_series, str) else dict(best_series)
                if WRITE_FEATURE_LISTS:
                    refit_write_features(set_name, model_tag, lf.X, lf.y, best_params, SAVE_DIR)
            except Exception as e:
                print(f"[WARN] Could not refit/write features for {set_name}__{model_tag}: {e}")

    if all_sum:
        summary = pd.concat(all_sum, ignore_index=True)
        summary = summary.sort_values("roc_auc_mean", ascending=False)
        summary.to_csv(SAVE_DIR / "summary_metrics.csv", index=False)
        print("\n=== Top results by ROC-AUC (mean) ===")
        print(summary.to_string(index=False))
        print(f"\nSaved outputs in: {SAVE_DIR.resolve()}")
    else:
        print("No results written (likely due to size/overlap constraints).")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import re
import argparse
from typing import List, Optional, Any

# =============================================================================
# FONCTIONS DE RECHERCHE ET DE NETTOYAGE
# =============================================================================

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Cherche le nom exact ou approximatif d'une colonne dans un DataFrame.
    Très utile pour les données cliniques où le nommage n'est pas standardisé.
    
    Args:
        df: Le DataFrame pandas contenant les données brutes.
        candidates: Une liste de noms de colonnes potentiels (ex: ["Birth date", "Birth"]).
        
    Returns:
        Le nom exact de la colonne tel qu'il est dans le DataFrame, ou None si introuvable.
    """
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    
    # 1. Recherche exacte (insensible à la casse)
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in low:
            return low[cand_l]
            
    # 2. Recherche par sous-chaîne (si "Birth" est dans "Date of Birth Patient")
    for cand in candidates:
        cand_l = cand.lower()
        for c in cols:
            if cand_l in c.lower():
                return c
                
    return None

def parse_date(val: Any) -> pd.Timestamp:
    """
    Tente de convertir une valeur en date complète (Année, Mois, Jour).
    Si seul l'année est fournie (ex: 1980), pandas la convertira au 01/01/1980, 
    ce qui est un compromis acceptable pour les données manquantes.
    """
    if pd.isna(val): 
        return pd.NaT
    # On force la conversion en datetime. errors="coerce" mettra NaT si échec.
    return pd.to_datetime(val, errors="coerce")

def extract_T(val: Any) -> float:
    """Extrait le stade de la tumeur (T) sous forme numérique."""
    if pd.isna(val): 
        return np.nan
    s = str(val).upper()
    # TIS (in situ) ou T0 correspondent à une absence de tumeur invasive
    if "TIS" in s or "T0" in s: 
        return 0.0
    # Cherche T suivi (ou non d'un espace) et d'un chiffre entre 1 et 4
    m = re.search(r"T\s*([1-4])", s)
    return float(m.group(1)) if m else np.nan 
    # group(0) aurait retourné toute la correspondance là où group(1) retourne uniquement le chiffre de groupe identifié par '([1-4])', les parenthèses permettent de captuer un groupe

def extract_N(val: Any) -> float:
    """Extrait l'envahissement ganglionnaire (N) sous forme numérique."""
    if pd.isna(val): 
        return np.nan
    s = str(val).upper()
    m = re.search(r"N\s*([0-3])", s)
    return float(m.group(1)) if m else np.nan

def clean_grade(val: Any) -> float:
    """Extrait le grade tumoral (Scarff-Bloom-Richardson ou Nottingham) de 1 à 3."""
    if pd.isna(val): 
        return np.nan
    m = re.search(r"([1-3])", str(val))
    return float(m.group(1)) if m else np.nan

def marker_generic(val: Any) -> float:
    """
    Logique générique pour encoder les récepteurs hormonaux (ER/PR).
    Transforme les termes textuels en valeurs continues (0.0, 0.5, 1.0) ou pourcentages.
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).strip().lower()
    
    # Cas limites
    if "equivocal" in s or "borderline" in s: 
        return 0.5
    
    # Cas positifs forts
    if any(tok in s for tok in ["pos", "positive", "3+", "2+", "+"]):
        if "neg" in s or "negative" in s: 
            return 0.0 # Gère les erreurs de saisie type "positive/negative"
        return 1.0
        
    # Cas négatifs
    if any(tok in s for tok in ["neg", "negative"]) or s == "-": 
        return 0.0
        
    # Valeurs binaires explicites
    if s in {"1", "0"}: 
        return float(s)
        
    # Si la valeur est un pourcentage, on l'extrait (sans imposer de seuil artificiel ici)
    nums = re.findall(r"\d+\.?\d*", s)
    if nums:
        v = float(nums[0])
        # Si c'est écrit "0.10" avec un "%", on corrige. 
        if "%" in s and v <= 1: 
            v *= 100
        # On laisse un nan sauf si une règle stricte est définie.
    return np.nan

def marker_ER(val: Any) -> float:
    """Règle spécifique pour le récepteur Oestrogène (ER)."""
    if pd.isna(val): 
        return np.nan
    s = str(val).strip().lower().replace("à", "a")
    # Capture "5 a 10%" ou "5-10%" et le considère comme positif (1.0)
    if re.search(r"5\s*(?:a|to|-)\s*10\s*%?", s):
        return 1.0
    # Si la règle spécifique ne s'applique pas, on passe au parseur générique
    return marker_generic(val)

def her2_code(val: Any) -> float:
    """
    Encodage spécifique du statut HER2 :
      0.0 -> Négatif (Score 0)
      1.0 -> Négatif (Score 1)
      2.0 -> Equivoque (Score 2 avec ISH négatif)
      NaN -> Positif fort (3) ou non renseigné (à gérer par imputation plus tard)
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).strip().lower().replace(" ", "")
    if s in {"0", "0+", "ihc0", "score0"}: 
        return 0.0
    if s in {"1", "1+", "ihc1", "score1"}: 
        return 1.0
    if ("2" in s) and ("ish" in s) and ("neg" in s):
        return 2.0
    return np.nan

def ki67(val: Any) -> float:
    """Nettoie et convertit l'indice de prolifération Ki-67 en pourcentage float."""
    if pd.isna(val): 
        return np.nan
    s = str(val).replace("%", "").replace(",", ".").strip()
    try:
        v = float(s)
        # Transforme les décimales (0.15) en pourcentages (15.0)
        if v <= 1: 
            v *= 100
        return v
    except ValueError: 
        return np.nan

def hist_code(val: Any) -> int:
    """
    Encodage du type histologique sans One-Hot Encoding (OHE).
      0 = Carcinome canalaire infiltrant (NST / IDC)
      1 = Carcinome lobulaire infiltrant (ILC)
      2 = Autre ou inconnu
    """
    if pd.isna(val): 
        return 2
    s = str(val).lower()
    if "nst" in s or "no special type" in s or "ductal" in s or "idc" in s: 
        return 0
    if "lobul" in s or "ilc" in s: 
        return 1
    return 2

def parse_ntil_category(val: Any) -> float:
    """
    Catégorisation des lymphocytes infiltrant la tumeur (nTIL).
    Divise la valeur par 10 et prend l'entier inférieur pour créer des tranches.
    Ex: 25% -> Catégorie 2.
    """
    if pd.isna(val): 
        return np.nan
    s = str(val).strip().lower().replace(" ", "")
    if s in {"na", "n/a", ""}: 
        return np.nan
        
    # Gère les valeurs inférieures à un seuil (ex: "<10%")
    if s.startswith("<"):
        nums = re.findall(r"\d+\.?\d*", s)
        if nums:
            v = float(nums[0])
            # Soustrait un epsilon pour forcer le basculement dans la catégorie inférieure
            # Ex: "<10" devient 9.99999 -> catégorie 0
            v = max(0.0, v - 1e-6)
            return int(np.floor(v / 10.0))
        return np.nan
        
    # Gère les pourcentages standards
    s_clean = s.replace("%", "")
    try:
        v = float(s_clean)
        # Note : 1% reste 1, donc catégorie 0. 
        if v <= 1: 
            v *= 1.0 
        return int(np.floor(v / 10.0))
    except ValueError:
        return np.nan


# =============================================================================
# PIPELINE PRINCIPAL (Exécuté si le script est lancé directement)
# =============================================================================

def main():
    # 1. Configuration du Parser d'arguments
    # Cela te permet de lancer : python prep_clinical.py -i input.xlsx -o output.xlsx
    parser = argparse.ArgumentParser(description="Nettoyage et encodage des données cliniques pour le projet CEM Breast Cancer.")
    parser.add_argument("-i", "--input", type=str, 
                        default=r"clinicals.xlsx",
                        help="Chemin vers le fichier Excel brut d'entrée.")
    parser.add_argument("-o", "--output", type=str, 
                        default=r"ready_steady_clinicals.xlsx",
                        help="Chemin vers le fichier Excel encodé de sortie.")
    args = parser.parse_args()

    print(f"Chargement des données depuis : {args.input}")
    
    # 2. Chargement du fichier
    try:
        df = pd.read_excel(args.input)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return

    # 3. Résolution dynamique des noms de colonnes
    c_birth  = find_col(df, ["Birth date", "Date of birth", "Year of birth", "Birth"])
    c_diag   = find_col(df, ["Date first diagnosis", "First diagnosis", "Diagnosis date"])
    c_T      = find_col(df, ["Stade T", "T stage", "T staging"])
    c_N      = find_col(df, ["Stade N", "N stage", "N staging"])
    c_hist   = find_col(df, ["Histology (NST, lobular, others)", "Histology"])
    c_grade  = find_col(df, ["Grading", "Grade"])
    c_er     = find_col(df, ["ER", "Estrogen"])
    c_pr     = find_col(df, ["PR", "Progesterone"])
    c_her2   = find_col(df, ["HER2 status", "HER2"])
    c_ki67   = find_col(df, ["Ki-67", "Ki67", "Ki 67"])
    c_ntil   = find_col(df, ["nTILS", "nTIL", "TILs", "TIL"])
    c_acro   = find_col(df, ["ACRONYME", "ACRONYM"])
    c_ref    = find_col(df, ["Reference ID", "ReferenceID", "Ref ID", "RefID", "PatientID", "Patient ID", "ID", "SubjectID", "Subject"])

    # 4. Construction du DataFrame encodé
    encoded = pd.DataFrame()

    # Identifiants (laissés intacts)
    if c_acro: encoded["ACRONYME"] = df[c_acro]
    if c_ref:  encoded["ReferenceID"] = df[c_ref]

    # Traitement précis des dates et de l'âge
    if c_birth: 
        birth_dates = df[c_birth].apply(parse_date)
    else:
        birth_dates = None

    if c_diag:  
        diag_dates = df[c_diag].apply(parse_date)
        # On garde l'année de diagnostic comme feature (ex: effet cohorte)
        encoded["DiagnosisYear"] = diag_dates.dt.year
    else:
        diag_dates = None

    # Calcul de l'âge précis (en années décimales) si les deux dates existent
    if birth_dates is not None and diag_dates is not None:
        # Soustraction des dates (donne des Timedelta), puis conversion en jours et division par 365.25
        encoded["AgeAtDiagnosis"] = (diag_dates - birth_dates).dt.days / 365.25

    # Encodage des stades et caractéristiques tumorales
    if c_T:     encoded["T_stage_num"]    = df[c_T].apply(extract_T)
    if c_N:     encoded["N_stage_num"]    = df[c_N].apply(extract_N)
    if c_grade: encoded["Grade"]          = df[c_grade].apply(clean_grade)

    # Encodage des biomarqueurs
    if c_er:    encoded["ER_pos"]         = df[c_er].apply(marker_ER)
    if c_pr:    encoded["PR_pos"]         = df[c_pr].apply(marker_generic)
    if c_her2:  encoded["HER2_code"]      = df[c_her2].apply(her2_code)
    if c_ki67:  encoded["Ki67_percent"]   = df[c_ki67].apply(ki67)

    # Encodage histologique et immunologique
    if c_hist:  encoded["Histology_code"] = df[c_hist].apply(hist_code)
    if c_ntil:  encoded["nTIL_cat"]       = df[c_ntil].apply(parse_ntil_category)

    print("[INFO] Application du One-Hot Encoding et de la binarisation stricte...")

    # 1. Binarisation stricte des récepteurs hormonaux (ER / PR)
    # Si la valeur est > 0, on la considère positive (1.0). Si 0, c'est négatif (0.0).
    # Les NaN restent NaN pour l'imputation ultérieure dans la GridSearch.
    if "ER_pos" in encoded.columns:
        encoded["ER_binary"] = encoded["ER_pos"].apply(
            lambda x: 1.0 if x > 0 else (0.0 if x == 0 else np.nan)
        )
        encoded.drop(columns=["ER_pos"], inplace=True) # On retire la colonne continue

    if "PR_pos" in encoded.columns:
        encoded["PR_binary"] = encoded["PR_pos"].apply(
            lambda x: 1.0 if x > 0 else (0.0 if x == 0 else np.nan)
        )
        encoded.drop(columns=["PR_pos"], inplace=True)

    # 2. One-Hot Encoding (OHE) pour les variables catégorielles
    # On utilise drop_first=True pour éviter le piège de la colinéarité parfaite 
    # (très important pour ta LogisticRegression et le SVC)
    cols_to_encode = []
    if "Histology_code" in encoded.columns:
        cols_to_encode.append("Histology_code")
    if "HER2_code" in encoded.columns:
        cols_to_encode.append("HER2_code")

    if cols_to_encode:
        # get_dummies va transformer Histology_code en Histology_code_1.0, Histology_code_2.0, etc.
        # dummy_na=False évite de créer une colonne pour les valeurs manquantes (l'imputer s'en chargera)
        encoded = pd.get_dummies(encoded, columns=cols_to_encode, drop_first=True, dummy_na=False)
        
        # Astuce : Convertir les booléens générés par get_dummies en float (0.0 / 1.0) 
        # pour éviter des bugs de type (bool vs float) dans scikit-learn
        for col in encoded.columns:
            if encoded[col].dtype == bool:
                encoded[col] = encoded[col].astype(float)

    # 5. Sauvegarde
    encoded.to_excel(args.output, index=False)
    print(f"Dataset encodé sauvegardé avec succès ({len(encoded)} patients) : {args.output}")

if __name__ == "__main__":
    main()

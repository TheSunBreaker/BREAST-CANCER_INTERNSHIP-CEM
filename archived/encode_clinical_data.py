import pandas as pd
import numpy as np
import re

INFILE  = r"D:\Home\esalasvilla\Documents\Stage\Breast_Project\Data\Data2predict.xlsx"
OUTFILE = r"D:\Home\esalasvilla\Documents\Stage\Breast_Project\Data\Data2predict_encoded.xlsx"

# ---------- Column finder (robust to slight name variations) ----------
def find_col(df, candidates):
    cols = [c for c in df.columns]
    low  = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        # exact
        if cand_l in low:
            return low[cand_l]
    # substring search
    for cand in candidates:
        cand_l = cand.lower()
        for c in cols:
            if cand_l in c.lower():
                return c
    return None

# ---------- Helpers ----------
def parse_year(val):
    if pd.isna(val): return np.nan
    # try datetime, else just first 4 digits
    dt = pd.to_datetime(val, errors="coerce")
    if pd.isna(dt):
        return pd.to_numeric(str(val)[:4], errors="coerce")
    return dt.year

def extract_T(val):
    if pd.isna(val): return np.nan
    s = str(val).upper()
    if "TIS" in s or "T0" in s: return 0
    m = re.search(r"T\s*([1-4])", s)
    return float(m.group(1)) if m else np.nan

def extract_N(val):
    if pd.isna(val): return np.nan
    s = str(val).upper()
    m = re.search(r"N\s*([0-3])", s)
    return float(m.group(1)) if m else np.nan

def clean_grade(val):
    if pd.isna(val): return np.nan
    m = re.search(r"([1-3])", str(val))
    return float(m.group(1)) if m else np.nan

def marker_generic(val):
    """Generic ER/PR mapping (pos/neg/etc.)."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if "equivocal" in s or "borderline" in s: return 0.5
    if any(tok in s for tok in ["pos","positive","3+","2+","+"]):
        if "neg" in s or "negative" in s: return 0.0
        return 1.0
    if any(tok in s for tok in ["neg","negative"]) or s == "-": return 0.0
    if s in {"1","0"}: return float(s)
    # try numeric percent like "10%" -> 10 (we don't impose a cutoff unless asked)
    nums = re.findall(r"\d+\.?\d*", s)
    if nums:
        v = float(nums[0])
        if "%" in s and v <= 1: v *= 100
        # leave as NaN unless explicit rule is given
    return np.nan

def marker_ER(val):
    """ER with special rule: '5 à 10 %' (and variants) -> 1."""
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace("à", "a")
    if re.search(r"5\s*(?:a|to|-)\s*10\s*%?", s):
        return 1.0
    return marker_generic(val)

def her2_code(val):
    """
    HER2 status mapping (exact as requested):
      '0' -> 0
      '1' -> 1
      '2DISHneg' (variants like '2 DISH neg', '2 FISH neg') -> 2
      else -> NaN
    """
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace(" ", "")
    if s in {"0", "0+", "ihc0", "score0"}: return 0.0
    if s in {"1", "1+", "ihc1", "score1"}: return 1.0
    # accept dish/fish variants
    if ("2" in s) and ("ish" in s) and ("neg" in s):
        return 2.0
    return np.nan

def ki67(val):
    if pd.isna(val): return np.nan
    s = str(val).replace("%","").replace(",",".").strip()
    try:
        v = float(s)
        if v <= 1: v *= 100
        return v
    except: 
        return np.nan

def hist_code(val):
    """
    Single categorical code (no one-hot):
      0 = NST/ductal/IDC/no special type
      1 = Lobular/ILC
      2 = Other/unknown
    """
    if pd.isna(val): return 2
    s = str(val).lower()
    if "nst" in s or "no special type" in s or "ductal" in s or "idc" in s: return 0
    if "lobul" in s or "ilc" in s: return 1
    return 2

def parse_ntil_category(val):
    """
    Build nTIL category: floor(percent / 10), giving 0,1,2,3,...
    Handles inputs like: 20, '10', '<10', '< 10', '<5%', '1', '3', 'NA'
    """
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace(" ", "")
    if s in {"na", "n/a", ""}: 
        return np.nan
    # '<10', '<10%', '<5%'
    if s.startswith("<"):
        nums = re.findall(r"\d+\.?\d*", s)
        if nums:
            v = float(nums[0])
            # treat '<10' as 9 to land in category 0
            v = max(0.0, v - 1e-6)
            return int(np.floor(v / 10.0))
        return np.nan
    # raw percent or number
    s_clean = s.replace("%","")
    try:
        v = float(s_clean)
        if v <= 1:  # e.g., '1' could be 1%, still category 0
            v *= 1.0
        return int(np.floor(v / 10.0))
    except:
        return np.nan

# ---------- Load ----------
df = pd.read_excel(INFILE)

# ---------- Resolve column names ----------
c_birth  = find_col(df, ["Birth date","Date of birth","Year of birth","Birth"])
c_diag   = find_col(df, ["Date first diagnosis","First diagnosis","Diagnosis date"])
c_T      = find_col(df, ["Stade T","T stage","T staging"])
c_N      = find_col(df, ["Stade N","N stage","N staging"])
c_hist   = find_col(df, ["Histology (NST, lobular, others)","Histology"])
c_grade  = find_col(df, ["Grading","Grade"])
c_er     = find_col(df, ["ER","Estrogen"])
c_pr     = find_col(df, ["PR","Progesterone"])
c_her2   = find_col(df, ["HER2 status","HER2"])
c_ki67   = find_col(df, ["Ki-67","Ki67","Ki 67"])
c_ntil   = find_col(df, ["nTILS","nTIL","TILs","TIL"])
c_acro   = find_col(df, ["ACRONYME","ACRONYM"])
c_ref    = find_col(df, ["Reference ID","ReferenceID","Ref ID","RefID","PatientID","Patient ID","ID","SubjectID","Subject"])

# ---------- Build encoded-only DataFrame ----------
encoded = pd.DataFrame()

if c_acro: encoded["ACRONYME"] = df[c_acro]          # untouched
if c_ref:  encoded["ReferenceID"] = df[c_ref]        # untouched (normalized name)

if c_birth: encoded["BirthYear"] = df[c_birth].apply(parse_year)
if c_diag:  encoded["DiagnosisYear"] = df[c_diag].apply(parse_year)
if c_birth and c_diag:
    encoded["AgeAtDiagnosis"] = encoded["DiagnosisYear"] - encoded["BirthYear"]

if c_T:     encoded["T_stage_num"] = df[c_T].apply(extract_T)
if c_N:     encoded["N_stage_num"] = df[c_N].apply(extract_N)
if c_grade: encoded["Grade"]       = df[c_grade].apply(clean_grade)

if c_er:    encoded["ER_pos"]      = df[c_er].apply(marker_ER)
if c_pr:    encoded["PR_pos"]      = df[c_pr].apply(marker_generic)
if c_her2:  encoded["HER2_code"]   = df[c_her2].apply(her2_code)
if c_ki67:  encoded["Ki67_percent"]= df[c_ki67].apply(ki67)

if c_hist:  encoded["Histology_code"] = df[c_hist].apply(hist_code)
if c_ntil:  encoded["nTIL_cat"]       = df[c_ntil].apply(parse_ntil_category)

# ---------- Save ONLY the encoded set (plus ACRONYME/ReferenceID) ----------
encoded.to_excel(OUTFILE, index=False)
print(f"✅ Saved encoded dataset (no raw clinical columns): {OUTFILE}")

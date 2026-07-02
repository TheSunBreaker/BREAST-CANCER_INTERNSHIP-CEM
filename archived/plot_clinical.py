import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from textwrap import wrap

# ======== CONFIG ========
INFILE = r"D:\Home\esalasvilla\Downloads\Data2predict_encoded 2.xlsx"  # path to your data
CATEGORY_COL = "Grade"                   # categorical predictor (e.g., Stage, Grade, Histology_code)
OUTCOME_COL = "pcrstatus"                # binary outcome: 1=pCR, 0=non-pCR
SAVEFIG = "pcr_by_category_barplot_grade.png"
ORDER = None   # e.g., ORDER=[1,2,3] to force order; else sorted unique values
# ========================


# Load
df = pd.read_excel(INFILE)

# Keep rows where outcome is 0/1
df = df[df[OUTCOME_COL].isin([0,1])].copy()

# Clean/ensure categorical
cats = ORDER if ORDER is not None else sorted(df[CATEGORY_COL].dropna().unique().tolist())

# Compute counts and percentages per category
summary = []
for c in cats:
    sub = df[df[CATEGORY_COL] == c]
    n = len(sub)
    n_pcr = int((sub[OUTCOME_COL] == 1).sum())
    n_non = n - n_pcr
    p_pcr = 100.0 * n_pcr / n if n > 0 else np.nan
    p_non = 100.0 * n_non / n if n > 0 else np.nan

    # Fisher exact: 2x2 (category c vs not-c) x (pCR vs non-pCR)
    rest = df[df[CATEGORY_COL] != c]
    table = [
        [n_pcr, n_non],                                  # in category c
        [(rest[OUTCOME_COL] == 1).sum(), (rest[OUTCOME_COL] == 0).sum()]  # in other categories
    ]
    # Guard against zeros that break odds ratio; Fisher handles zeros fine
    _, pval = fisher_exact(table, alternative='two-sided')

    summary.append({"category": c, "n": n, "pcr%": p_pcr, "nonpcr%": p_non, "pval": pval})

sum_df = pd.DataFrame(summary)

# --- Plot ---
x = np.arange(len(sum_df))
width = 0.38

fig, ax = plt.subplots(figsize=(10,6), dpi=120)

bars1 = ax.bar(x - width/2, sum_df["pcr%"], width, label="pCR (1)")
bars2 = ax.bar(x + width/2, sum_df["nonpcr%"], width, label="Non‑pCR (0)")

# Labels & layout
ax.set_ylabel("Percentage within category (%)")
ax.set_title(f"{OUTCOME_COL} by {CATEGORY_COL}")
xticks = [str(c) for c in sum_df["category"]]
ax.set_xticks(x, xticks, rotation=30, ha="right")
ax.set_ylim(0, max(sum_df[["pcr%","nonpcr%"]].max()) * 1.20)
ax.legend()

# Annotate p-values above the category pair
for i, row in sum_df.iterrows():
    y = max(row["pcr%"], row["nonpcr%"])
    text = f"p = {row['pval']:.3g}"
    ax.text(i, y + (ax.get_ylim()[1]*0.03), text, ha="center", va="bottom")

# Optional: add value labels on bars
def autolabel(rects):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2.0, h + 1.0, f"{h:.0f}%", ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig(SAVEFIG)
plt.show()

print("Saved plot to:", SAVEFIG)
print("\nSummary:\n", sum_df)

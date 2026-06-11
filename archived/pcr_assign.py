#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


def sniff_delimiter(path: Path, fallback: str = ",") -> str:
    """Try to sniff the delimiter. Fall back if sniffing fails."""
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return fallback


def load_csv_robust(path: Path, prefer: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV trying:
      1) preferred delimiter if specified (e.g., ';' for clinical),
      2) sniffed delimiter,
      3) comma fallback.
    """
    if prefer:
        try:
            return pd.read_csv(path, sep=prefer)
        except Exception:
            pass

    delim = sniff_delimiter(path, fallback=",")
    try:
        return pd.read_csv(path, sep=delim, engine="python")
    except Exception:
        # final fallback to comma
        return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def main(
    clinical_csv: Path,
    petct_csv: Path,
    output_csv: Path,
    write_summary: Optional[Path] = None
) -> Tuple[int, int, int, List[str]]:
    # Load
    clinical_df = load_csv_robust(clinical_csv, prefer=";")
    petct_df = load_csv_robust(petct_csv)

    # Normalize headers
    clinical_df = normalize_columns(clinical_df)
    petct_df = normalize_columns(petct_df)

    # Checks
    if "subject_id" not in clinical_df.columns:
        raise KeyError("Column 'subject_id' not found in clinical CSV.")
    if "pcrstatus" not in clinical_df.columns:
        raise KeyError("Column 'pcrstatus' not found in clinical CSV.")
    if "subject_id" not in petct_df.columns:
        raise KeyError("Column 'subject_id' not found in PET/CT CSV.")

    # Build mapping
    clinical_map = (
        clinical_df[["subject_id", "pcrstatus"]]
        .drop_duplicates(subset=["subject_id"], keep="last")
    )

    # Merge
    merged = petct_df.merge(clinical_map, on="subject_id", how="left")

    # Stats before dropping NaN
    initial_n = len(petct_df)
    assigned_n = merged["pcrstatus"].notna().sum()
    skipped_n = initial_n - assigned_n
    skipped_subjects = (
        merged.loc[merged["pcrstatus"].isna(), "subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    # Drop rows where pcrstatus is missing
    merged = merged.dropna(subset=["pcrstatus"])

    # Save merged CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    # Also save Excel version
    output_excel = output_csv.with_suffix(".xlsx")
    merged.to_excel(output_excel, index=False, sheet_name="Sheet1")

    # Optional: write a summary CSV
    if write_summary:
        summary_df = pd.DataFrame(
            {
                "Initial subjects (rows)": [initial_n],
                "Assigned pCR values": [assigned_n],
                "Skipped (no pCR found)": [skipped_n],
            }
        )
        write_summary.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(write_summary, index=False)

    # Console report
    print("=== Merge pCR into PET/CT report ===")
    print(f"Clinical file: {clinical_csv}")
    print(f"PET/CT file:   {petct_csv}")
    print(f"Output CSV:    {output_csv}")
    print(f"Output Excel:  {output_excel}")
    print("------------------------------------")
    print(f"Initial subjects (rows): {initial_n}")
    print(f"Assigned pCR values:     {assigned_n}")
    print(f"Skipped (no pCR found):  {skipped_n}")
    if skipped_subjects:
        print("Skipped subject_id list:")
        print(", ".join(skipped_subjects))
    else:
        print("No subjects were skipped.")

    return initial_n, assigned_n, skipped_n, skipped_subjects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge pcrstatus from a clinical CSV into a PET/CT features CSV via subject_id."
    )
    parser.add_argument(
        "--clinical", type=Path, required=True,
        help="Path to Clinical_features.csv"
    )
    parser.add_argument(
        "--petct", type=Path, required=True,
        help="Path to PET_CT_features.csv"
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Path to write merged CSV (e.g., PET_CT_features_with_pcr.csv)"
    )
    parser.add_argument(
        "--write-summary", type=Path, default=None,
        help="Optional path to write a small summary CSV"
    )
    args = parser.parse_args()

    main(
        clinical_csv=args.clinical,
        petct_csv=args.petct,
        output_csv=args.out,
        write_summary=args.write_summary,
    )

import argparse
from pathlib import Path
import os
import csv
import pydicom
import SimpleITK as sitk

from src.suv_conversion import (  
    read_dicom_series,            # loads a DICOM series as sitk.Image
    extract_patient_parameters,   # dose/weight/times/half-life/sex from a PET DICOM header
    compute_suv_factors,          # computes SUV factors; we'll use SUVbw
    write_normalized_image        # scales & writes an image; appends run info to a log
)


def _find_pet_header_with_rph(dir_path: Path):
    """
    Return a PET (Modality 'PT') DICOM dataset from dir_path that contains
    RadiopharmaceuticalInformationSequence. Else (None, None).
    """
    for f in dir_path.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", None) == "PT" and hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            return ds, f
    return None, None


def _load_csv_params(csv_path: Path):
    """
    Load fallback SUV parameters from CSV into a dict by subject_id.
    Expected columns (all optional except subject_id; missing become empty strings):
      subject_id, injected_dose, patient_weight, patient_height, half_life,
      injection_time, series_time, sex
    NOTE: Units must be consistent with your suv_conv.compute_suv_factors() expectations.
    """
    if not csv_path:
        return {}
    db = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("subject_id", "")).strip()
            if not sid:
                continue
            db[sid] = row
    return db


def _row_to_params(row: dict):
    """
    Convert a CSV row dict into the params dict expected by compute_suv_factors().
    Missing values default to 0/UNKNOWN.
    """
    def fget(name, default=""):
        v = row.get(name, default)
        return "" if v is None else str(v).strip()

    # Convert numerics safely
    def ffloat(s, default=0.0):
        try:
            return float(s)
        except Exception:
            return default

    return {
        "injected_dose": ffloat(fget("injected_dose", 0.0)),     # Keep units consistent with your DICOM/site
        "patient_weight": ffloat(fget("patient_weight", 0.0)),   # kg
        "patient_height": ffloat(fget("patient_height", 0.0)),   # meters
        "half_life": ffloat(fget("half_life", 0.0)),             # seconds (F-18 ~ 6586.2)
        "injection_time": fget("injection_time", "000000"),      # HHMMSS(.ffffff) as string
        "series_time": fget("series_time", "000000"),            # HHMMSS(.ffffff) as string
        "patient_sex": fget("sex", "UNKNOWN")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert PET (TEP) DICOM to SUVbw NIfTI, ignoring IRM/TDM if present."
    )
    parser.add_argument("input_root", type=Path, help="Root with subject_xxx subfolders (each may contain IRM/, TDM/, TEP/).")
    parser.add_argument("output_root", type=Path, help="Root where subject_xxx/subject_xxx_TEP.nii.gz will be written.")
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Optional CSV fallback with SUV parameters per subject_id "
                             "(columns: subject_id, injected_dose, patient_weight, patient_height, half_life, injection_time, series_time, sex)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    global_log = os.path.join(output_root, "suv_conversion_log.txt")
    # Open once to stamp the run header (use utf-8 to avoid Windows codepage issues)
    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n================= SUV CONVERSION START =================\n")
        log.write(f"Input root: {input_root}\nOutput root: {output_root}\n")

    csv_db = _load_csv_params(args.metadata_csv) if args.metadata_csv else {}

    subjects = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not subjects:
        print(f"No subject folders found under: {input_root}")
        with open(global_log, "a", encoding="utf-8") as log:
            log.write("No subjects found. Exiting.\n")
            log.write("================== SUV CONVERSION END ==================\n")
        return

    n_total = 0
    n_converted = 0
    n_skipped_no_tep = 0
    n_skipped_read_fail = 0
    n_skipped_params = 0
    n_exists = 0

    print(f"Found {len(subjects)} subject folder(s).")
    for subj_dir in subjects:
        subject_id = subj_dir.name
        n_total += 1
        print(f"\n[SUBJ] {subject_id}")

        # Only PET is handled. IRM/ and TDM/ are ignored by design.
        tep_dir = subj_dir / "TEP"
        if not tep_dir.exists():
            print(f"[SKIP] No PET folder found at {tep_dir}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: no TEP/ folder.\n")
            n_skipped_no_tep += 1
            continue

        out_subj_dir = output_root / subject_id
        out_subj_dir.mkdir(parents=True, exist_ok=True)
        out_tep = out_subj_dir / f"{subject_id}_TEP_SUV.nii.gz"

        if out_tep.exists() and not args.overwrite:
            print(f"[OK  ] Output exists, skipping (use --overwrite to redo): {out_tep}")
            n_exists += 1
            continue

        # (1) Read PET image
        pet_img = read_dicom_series(str(tep_dir))
        if pet_img is None:
            print(f"[SKIP] PET: could not read series in {tep_dir}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: PET read failed from {tep_dir}\n")
            n_skipped_read_fail += 1
            continue

        # (2) Get PET header with radiopharm info. If missing, try CSV fallback.
        ds, _ = _find_pet_header_with_rph(tep_dir)
        params = None
        if ds is not None:
            try:
                params = extract_patient_parameters(ds)
            except Exception as e:
                params = None

        if params is None:
            # Fallback: CSV row for this subject (if provided)
            row = csv_db.get(subject_id)
            if row:
                params = _row_to_params(row)

        if params is None:
            print(f"[SKIP] PET: no radiopharmaceutical metadata found and no CSV fallback for {subject_id}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: missing PET parameters (no RPH in DICOM and no CSV row)\n")
            n_skipped_params += 1
            continue

        # Optional: if half-life missing but you know tracer is F-18, you can set a default
        if not params.get("half_life") or params["half_life"] == 0.0:
            # F-18 physical half-life ~6586.2 s
            params["half_life"] = 6586.2

        try:
            factors = compute_suv_factors(params)
            suv_factor = float(factors.get("SUVbw", 0.0))
        except Exception as e:
            print(f"[SKIP] PET: SUV factor computation failed ({e})")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[SKIP] {subject_id}: SUV factor computation failed: {e}\n")
            n_skipped_params += 1
            continue

        # (3) Scale & write SUVbw PET NIfTI (write_normalized_image appends to log)
        try:
            write_normalized_image(
                image=pet_img,
                output_path=str(out_tep),
                factor=suv_factor,
                log_path=global_log
            )
            print(f"[OK  ] Wrote PET SUVbw NIfTI: {out_tep}")
            n_converted += 1
        except Exception as e:
            print(f"[ERROR] PET write failed: {e}")
            with open(global_log, "a", encoding="utf-8") as log:
                log.write(f"[ERROR] {subject_id}: PET write failed: {e}\n")
            n_skipped_read_fail += 1

    # Summary
    print("\n===== Summary =====")
    print(f"Subjects total         : {n_total}")
    print(f"Converted (SUVbw)      : {n_converted}")
    print(f"Skipped (no TEP/)      : {n_skipped_no_tep}")
    print(f"Skipped (read failure) : {n_skipped_read_fail}")
    print(f"Skipped (no params)    : {n_skipped_params}")
    print(f"Skipped (exists)       : {n_exists}")

    with open(global_log, "a", encoding="utf-8") as log:
        log.write("\n===== Summary =====\n")
        log.write(f"Subjects total         : {n_total}\n")
        log.write(f"Converted (SUVbw)      : {n_converted}\n")
        log.write(f"Skipped (no TEP/)      : {n_skipped_no_tep}\n")
        log.write(f"Skipped (read failure) : {n_skipped_read_fail}\n")
        log.write(f"Skipped (no params)    : {n_skipped_params}\n")
        log.write(f"Skipped (exists)       : {n_exists}\n")
        log.write("================== SUV CONVERSION END ==================\n")


if __name__ == "__main__":
    main()
 

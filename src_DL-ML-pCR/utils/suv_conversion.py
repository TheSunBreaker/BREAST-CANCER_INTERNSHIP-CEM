import os
import math
import pydicom
import SimpleITK as sitk
import datetime
from typing import Optional, Dict


def convert_time_to_seconds(timestr: str) -> float:
    if not timestr or timestr == "MODULE_INIT_NO_VALUE":
        return 0.0
    try:
        hh = int(timestr[0:2]) if len(timestr) >= 2 else 0
        mm = int(timestr[2:4]) if len(timestr) >= 4 else 0
        ss = float(timestr[4:]) if len(timestr) > 4 else 0.0
        return hh * 3600 + mm * 60 + ss
    except Exception as e:
        print(f"Time parsing error: {e}")
        return 0.0


def decay_correction(injected_dose: float, series_time: str, injection_time: str, half_life: float) -> float:
    scan_time_seconds = convert_time_to_seconds(series_time)
    start_time_seconds = convert_time_to_seconds(injection_time)
    decay_time = scan_time_seconds - start_time_seconds
    decayed_dose = injected_dose * math.pow(2.0, -(decay_time / half_life))
    return decayed_dose


def get_metadata_value(dataset: pydicom.dataset.FileDataset, tag: tuple, default=None):
    try:
        return str(dataset.get(tag).value)
    except Exception:
        return default


def read_dicom_series(dicom_dir: str) -> Optional[sitk.Image]:
    reader = sitk.ImageSeriesReader()

    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        print(f"[ERROR] No DICOM series found in {dicom_dir}")
        return None

    try:
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
        reader.SetFileNames(series_file_names) #added line
    except Exception as e:
        print(f"[ERROR] Could not get file names for series: {e}")
        return None


    try:
        image = reader.Execute()
        print(f"[INFO] Loaded DICOM series from {dicom_dir} with {len(series_file_names)} slices")
        print(f"[INFO] Image shape: {sitk.GetArrayFromImage(image).shape}")
        return image
    except RuntimeError as e:
        print(f"[ERROR] Failed to read DICOM series: {e}")
        return None


def extract_patient_parameters(ds: pydicom.dataset.FileDataset) -> Dict:
    rph = ds.RadiopharmaceuticalInformationSequence[0]
    return {
        "injected_dose": float(getattr(rph, "RadionuclideTotalDose", 0.0)),
        "patient_weight": float(getattr(ds, "PatientWeight", 0.0)),
        "patient_height": float(getattr(ds, "PatientSize", 0.0)),  # meters
        "half_life": float(getattr(rph, "RadionuclideHalfLife", 0.0)),
        "injection_time": getattr(rph, "RadiopharmaceuticalStartTime", "000000"),
        "series_time": getattr(ds, "SeriesTime", "000000"),
        "patient_sex": getattr(ds, "PatientSex", "UNKNOWN")
    }


def compute_suv_factors(params: Dict) -> Dict:
    try:
        print("Injected Dose (MBq):", params["injected_dose"])
        print("Patient Weight (kg):", params["patient_weight"])
        print("Half-life (s):", params["half_life"])
        print("Injection time:", params["injection_time"])
        print("Series time:", params["series_time"])

        dose_kbq = params["injected_dose"] / 1000  # convert MBq to kBq
        decayed_dose = decay_correction(dose_kbq, params["series_time"], params["injection_time"], params["half_life"])
        weight_kg = params["patient_weight"]
        height_cm = params["patient_height"] * 100

        factors = {
            "SUVbw": weight_kg / decayed_dose if decayed_dose > 0 else 0.0,
            "SUVlbm": 0.0,
            "SUVbsa": 0.0,
            "SUVibw": 0.0
        }

        if height_cm > 0:
            if params["patient_sex"] == "M":
                lbm = 1.10 * weight_kg - 128 * (weight_kg / params["patient_height"]) ** 2
                ibw = 48.0 + 1.06 * (height_cm - 152)
            else:
                lbm = 1.07 * weight_kg - 148 * (weight_kg / params["patient_height"]) ** 2
                ibw = 45.5 + 0.91 * (height_cm - 152)
            ibw = min(ibw, weight_kg)
            bsa = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)

            factors["SUVlbm"] = lbm / decayed_dose
            factors["SUVbsa"] = bsa / decayed_dose
            factors["SUVibw"] = ibw / decayed_dose

        return factors
    except Exception as e:
        print(f"[ERROR] Could not compute SUV factors: {e}")
        return {
            "SUVbw": 0.0,
            "SUVlbm": 0.0,
            "SUVbsa": 0.0,
            "SUVibw": 0.0
        }



def write_normalized_image(image: sitk.Image, output_path: str, factor: float, log_path: str) -> None:
    pet_np = sitk.GetArrayFromImage(image)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n========== SUV conversion run: {} ==========\n".format(datetime.datetime.now()))
        log_file.write(f"-> PET image parameters BEFORE conversion:\n")
        log_file.write(f"- Shape: {pet_np.shape}\n")
        log_file.write(f"- Spacing: {image.GetSpacing()}\n")
        log_file.write(f"- Origin: {image.GetOrigin()}\n")
        log_file.write(f"- Direction: {image.GetDirection()}\n")
        log_file.write(f"\n-> SUV factor used: {factor}\n")

    if factor == 0:
        print(f"Skipping {output_path}: SUV factor is 0")
        with open(log_path, "a") as log_file:
            log_file.write(f"[WARNING] Skipped {output_path}: SUV factor is 0\n")
            log_file.write("========== End of run ==========\n")
        return

    try:
        scaled = sitk.ShiftScale(image, shift=0.0, scale=factor)
        sitk.WriteImage(scaled, output_path)
        print(f"Saved normalized image to: {output_path}")

        with open(log_path, "a") as log_file:
            log_file.write(f"-> Final status:\n")
            log_file.write(f"- Saved normalized image to: {output_path}\n")
            log_file.write("========== End of run ==========\n")
    except Exception as e:
        print(f"[ERROR] Could not save normalized image: {e}")
        with open(log_path, "a") as log_file:
            log_file.write(f"[ERROR] Could not save normalized image: {e}\n")
            log_file.write("========== End of run ==========\n")


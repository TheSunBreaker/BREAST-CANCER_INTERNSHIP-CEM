""""
Petit script pour ausculter les masques appartenant à un même patient pour une même visite, histoire de mieux savoir comment les traiter :
Les fusionner ? En prendre un seul ? Etc. 
A mettre dans le mêem dossier que les masques FORMAT DICOM à ausculter. 
""""

import os
import pydicom
from collections import defaultdict
from datetime import datetime

MASK_DIR = r"."

def safe_get(ds, key, default="UNKNOWN"):
    return str(getattr(ds, key, default))

def parse_datetime(ds):
    date = safe_get(ds, "StructureSetDate", "19000101")
    time = safe_get(ds, "StructureSetTime", "000000").split(".")[0]

    if len(time) < 6:
        time = time.ljust(6, "0")

    try:
        return datetime.strptime(date + time[:6], "%Y%m%d%H%M%S")
    except:
        return None

def inspect_rtstruct(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

        print("\n====================================================")
        print(f"FICHIER : {os.path.basename(path)}")
        print("====================================================")

        modality = safe_get(ds, "Modality")
        print(f"Modalité : {modality}")

        dt = parse_datetime(ds)
        print(f"Date masque : {dt}")

        print(f"SeriesInstanceUID :")
        print(f"  {safe_get(ds, 'SeriesInstanceUID')}")

        print(f"\nReferenced Series UID :")

        ref_uid = "NOT_FOUND"

        if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
            try:
                study_seq = ds.ReferencedFrameOfReferenceSequence[0]
                rt_ref_study = study_seq.RTReferencedStudySequence[0]
                rt_ref_series = rt_ref_study.RTReferencedSeriesSequence[0]

                ref_uid = rt_ref_series.SeriesInstanceUID
            except:
                pass

        elif hasattr(ds, "ReferencedSeriesSequence"):
            try:
                ref_uid = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
            except:
                pass

        print(f"  {ref_uid}")

        print("\n--- ROIS ---")

        roi_names = {}

        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                roi_number = roi.ROINumber
                roi_name = safe_get(roi, "ROIName")

                roi_names[roi_number] = roi_name

                print(f"ROI #{roi_number} : {roi_name}")

        print("\n--- CONTOURS ---")

        total_contours = 0

        if hasattr(ds, "ROIContourSequence"):
            for contour in ds.ROIContourSequence:

                roi_num = getattr(contour, "ReferencedROINumber", "UNKNOWN")
                roi_name = roi_names.get(roi_num, "UNKNOWN")

                nb_contours = len(getattr(contour, "ContourSequence", []))

                total_contours += nb_contours

                print(
                    f"ROI {roi_name} "
                    f"(#{roi_num}) -> {nb_contours} contours"
                )

        print(f"\nTOTAL CONTOURS : {total_contours}")

        print("\n--- OBSERVATIONS ---")

        if hasattr(ds, "RTROIObservationsSequence"):
            for obs in ds.RTROIObservationsSequence:

                roi_num = getattr(obs, "ReferencedROINumber", "UNKNOWN")
                roi_name = roi_names.get(roi_num, "UNKNOWN")

                interp = safe_get(obs, "RTROIInterpretedType")

                print(
                    f"ROI {roi_name} "
                    f"(#{roi_num}) -> Type : {interp}"
                )

        return {
            "file": path,
            "ref_uid": ref_uid,
            "datetime": dt,
            "roi_names": list(roi_names.values()),
            "total_contours": total_contours
        }

    except Exception as e:
        print(f"\n[ERREUR] {path}")
        print(e)
        return None


results = []

for root, _, files in os.walk(MASK_DIR):
    for f in files:

        fullpath = os.path.join(root, f)

        try:
            ds = pydicom.dcmread(fullpath, stop_before_pixels=True, force=True)

            if ds.Modality in ["RTSTRUCT", "SEG"]:
                result = inspect_rtstruct(fullpath)

                if result:
                    results.append(result)

        except:
            continue


print("\n\n####################################################")
print("################### COMPARAISON ####################")
print("####################################################")

grouped = defaultdict(list)

for r in results:
    grouped[r["ref_uid"]].append(r)

for ref_uid, group in grouped.items():

    if len(group) < 2:
        continue

    print("\n====================================================")
    print(f"IMAGE RÉFÉRENCÉE : {ref_uid}")
    print(f"NOMBRE DE MASQUES : {len(group)}")
    print("====================================================")

    for g in sorted(group, key=lambda x: x["datetime"] or datetime.min):

        print(f"\nFichier : {os.path.basename(g['file'])}")
        print(f"Date : {g['datetime']}")
        print(f"ROIs : {g['roi_names']}")
        print(f"Nb contours : {g['total_contours']}")

    print("\nINTERPRÉTATION POSSIBLE :")

    roi_sets = [tuple(sorted(g["roi_names"])) for g in group]

    if len(set(roi_sets)) == 1:
        print(" -> Même type de ROI.")
        print(" -> Possible corrections/versioning/inter-annotateurs.")
    else:
        print(" -> ROIs différentes détectées.")
        print(" -> Possible lésions multiples ou structures distinctes.")

"""
Petit script pour ausculter les masques appartenant à un même patient pour une même visite, histoire de mieux savoir comment les traiter :
Les fusionner ? En prendre un seul ? Etc. 
A mettre dans le même dossier que les masques FORMAT DICOM à ausculter. 
"""

import os
import pydicom
import numpy as np

MASK_DIR = r"."

def safe_get(obj, attr, default="UNKNOWN"):
    return str(getattr(obj, attr, default))

for root, _, files in os.walk(MASK_DIR):

    for f in files:

        path = os.path.join(root, f)

        try:
            if not f.lower().endswith(".dcm"):
                continue
            ds = pydicom.dcmread(path, force=True)

            if ds.Modality != "SEG":
                continue

            print("\n====================================================")
            print(f"FICHIER : {f}")
            print("====================================================")

            print(f"Modality : {ds.Modality}")

            print("\n--- IDENTITÉ ---")

            print(f"Series UID :")
            print(f"  {safe_get(ds, 'SeriesInstanceUID')}")

            print(f"\nReferenced Series UID :")

            ref_uid = "UNKNOWN"

            try:
                ref_uid = (
                    ds.ReferencedSeriesSequence[0]
                    .SeriesInstanceUID
                )
            except:
                pass

            print(f"  {ref_uid}")

            print("\n--- SEGMENTS ---")

            if hasattr(ds, "SegmentSequence"):

                for seg in ds.SegmentSequence:

                    seg_num = safe_get(seg, "SegmentNumber")
                    seg_label = safe_get(seg, "SegmentLabel")
                    seg_desc = safe_get(seg, "SegmentDescription")

                    print(f"\nSegment #{seg_num}")
                    print(f"Label       : {seg_label}")
                    print(f"Description : {seg_desc}")

                    try:
                        cat = seg.SegmentedPropertyCategoryCodeSequence[0]
                        print(f"Category    : {safe_get(cat, 'CodeMeaning')}")
                    except:
                        pass

                    try:
                        typ = seg.SegmentedPropertyTypeCodeSequence[0]
                        print(f"Type        : {safe_get(typ, 'CodeMeaning')}")
                    except:
                        pass

            print("\n--- PIXELS / VOLUME ---")

            try:

                arr = ds.pixel_array

                print(f"Shape : {arr.shape}")
                print(f"Dtype : {arr.dtype}")

                unique_vals = np.unique(arr)

                print(f"Valeurs uniques : {unique_vals}")

                nonzero = np.count_nonzero(arr)

                print(f"Nb voxels segmentés : {nonzero}")

            except Exception as e:
                print(f"[ERREUR PIXELS] {e}")

        except Exception as e:

            print(f"\n[ERREUR LECTURE] {f}")
            print(e)

import os
import shutil

def extract_irm_to_nnunet_flat(subjects_dir, nnunet_root, dataset_id=1, irm_suffix="_IRM.nii.gz"):
    """
    Extracts IRM NIfTI images and creates full nnUNet v2 folder structure.

    Parameters:
    - subjects_dir (str): Root directory with subject folders.
    - nnunet_base (str): Base directory where 'nnunetv2/' will be created.
    - dataset_id (int): Dataset number (e.g., 1 → Dataset001).
    - irm_suffix (str): Suffix for IRM files (default: "_IRM.nii.gz").
    """
    dataset_name = f"Dataset{dataset_id:03d}"

    # Create nnunetv2 under the given base path
    nnunet_root = os.path.join(nnunet_root, "nnunetv2")
    os.makedirs(nnunet_root, exist_ok=True)

    # Define full subfolders
    nnunet_raw = os.path.join(nnunet_root, "nnUNet_raw", dataset_name)
    nnunet_preprocessed = os.path.join(nnunet_root, "nnUNet_preprocessed")
    nnunet_results = os.path.join(nnunet_root, "nnUNet_results", dataset_name, "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_0")

    imagesTr_dir = os.path.join(nnunet_raw, "imagesTr")
    labelsTr_dir = os.path.join(nnunet_raw, "labelsTr")
    predTs_dir = os.path.join(nnunet_raw, "imagesTs_pred3dfullres")

    # Create all required folders
    for d in [imagesTr_dir, labelsTr_dir, predTs_dir, nnunet_preprocessed, nnunet_results]:
        os.makedirs(d, exist_ok=True)

    subjects = sorted([s for s in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, s))])
    print(f"Found {len(subjects)} subjects in: {subjects_dir}")

    for subj in subjects:
        subj_path = os.path.join(subjects_dir, subj)
        expected_irm = os.path.join(subj_path, f"{subj}{irm_suffix}")

        if not os.path.isfile(expected_irm):
            print(f"[Skip] IRM file not found for {subj}: {expected_irm}")
            continue

        dest_file = os.path.join(imagesTr_dir, f"{subj}_0000.nii.gz")
        shutil.copy(expected_irm, dest_file)
        print(f"Copied: {expected_irm} → {dest_file}")

    print(f"\nIRM-to-nnUNet conversion complete.")
    print(f"→ nnunetv2 folder created at: {nnunet_root}")
    print(f"→ imagesTr folder: {imagesTr_dir}")

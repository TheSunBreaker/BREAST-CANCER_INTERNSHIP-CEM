import os
import SimpleITK as sitk


def dicom2nifti(dicom_folder, output_dir, prefix="output"):
    """
    Converts a DICOM series to a NIfTI volume using SimpleITK.

    Parameters:
        dicom_folder (str): Path to the folder containing DICOM files.
        output_dir (str): Directory where the NIfTI file will be saved.
        prefix (str): Optional prefix for the output filename.

    Returns:
        str: Path to the saved NIfTI file.
    """
    if not os.path.isdir(dicom_folder):
        raise FileNotFoundError(f"DICOM folder not found: {dicom_folder}")

    os.makedirs(output_dir, exist_ok=True)

    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(dicom_folder)

    if not series_files:
        raise RuntimeError(f"No DICOM series found in: {dicom_folder}")

    reader.SetFileNames(series_files)
    image = reader.Execute()

    # Save the image as NIfTI
    output_path = os.path.join(output_dir, f"{prefix}.nii.gz")
    sitk.WriteImage(image, output_path)
    print(f"Saved NIfTI volume: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a DICOM series to NIfTI format.")
    parser.add_argument("dicom_folder", help="Path to folder containing DICOM files.")
    parser.add_argument("output_dir", help="Directory to save the NIfTI file.")
    parser.add_argument("--prefix", default="output", help="Prefix for output NIfTI filename.")

    args = parser.parse_args()

    try:
        dicom2nifti(args.dicom_folder, args.output_dir, args.prefix)
    except Exception as e:
        print(f"Error: {e}")

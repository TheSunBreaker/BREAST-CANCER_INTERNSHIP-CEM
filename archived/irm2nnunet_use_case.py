# irm_to_nnunet_use_case.py

import argparse
import sys

# Import the function from the module
try:
    from src.irm2nnunet import extract_irm_to_nnunet_flat
except ImportError:
    print("Could not import extract_irm_to_nnunet_flat from src/irm_to_nnunet.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Convert subject-wise IRM NIfTI files to nnUNet format with full directory setup.",
        epilog="""
Example:
  python irm_to_nnunet_use_case.py /path/to/input_root /path/to/nnunet_root --dataset-id 1
        """
    )

    parser.add_argument("subjects_dir", help="Directory with subject folders (each containing *_IRM.nii.gz)")
    parser.add_argument("nnunet_root", help="Root nnUNet folder to create nnUNet_raw, results, etc.")
    parser.add_argument("--dataset-id", type=int, default=1, help="Dataset ID (e.g., 1 → Dataset001)")
    parser.add_argument("--irm-suffix", default="_IRM.nii.gz", help="Suffix to identify IRM files (default: '_IRM.nii.gz')")

    args = parser.parse_args()

    try:
        extract_irm_to_nnunet_flat(
            subjects_dir=args.subjects_dir,
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            irm_suffix=args.irm_suffix
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Use Case Example for DICOM to NIfTI Conversion

This script demonstrates how to use the dcm2nii function to convert DICOM series
to NIfTI format for multiple subjects and modalities.

Author: esalasvilla
Date: 2025
"""
import os
import sys
from pathlib import Path
import argparse
from typing import Tuple, Optional

# Import the dicom2nifti function from the src module
try:
    from src.dcm2nii import dicom2nifti
except ImportError:
    print("Error: Could not import dicom2nifti function from src/dcm2nii.py")
    print("Make sure src/dcm2nii.py exists and is accessible.")
    sys.exit(1)


def process_subjects(input_root: str, output_root: str, modalities: Tuple[str, ...] = ("IRM", "TEP", "TDM")) -> None:
    """
    Navigates the input folder structure and converts available DICOM series to NIfTI.

    Parameters:
        input_root (str): Root input directory containing subject folders.
        output_root (str): Directory where converted NIfTI files will be saved.
        modalities (tuple): List of modality subfolders to check and convert.
    """
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root folder does not exist: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    for subj_id in sorted(os.listdir(input_root)):
        subj_path = os.path.join(input_root, subj_id)
        if not os.path.isdir(subj_path):
            continue  # skip files

        print(f"\nProcessing subject: {subj_id}")
        for modality in modalities:
            modality_path = os.path.join(subj_path, modality)
            if os.path.isdir(modality_path):
                try:
                    output_dir = os.path.join(output_root, subj_id)
                    prefix = f"{subj_id}_{modality}"
                    dicom2nifti(modality_path, output_dir, prefix=prefix)
                except Exception as e:
                    print(f"Failed to convert {modality_path}: {e}")
            else:
                print(f"Skipping {modality} — not found for {subj_id}")



def main():
    """Main function to demonstrate the use case."""
    parser = argparse.ArgumentParser(
        description="DICOM to NIfTI conversion use case example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all subjects with default modalities (IRM, TEP, TDM)
  python use_case_example.py /path/to/input /path/to/output
  
  # Convert with custom modalities
  python use_case_example.py /path/to/input /path/to/output --modalities IRM TEP
  
        """
    )
    
    parser.add_argument("input_root", nargs="?", help="Root input directory containing subject folders")
    parser.add_argument("output_root", nargs="?", help="Directory where converted NIfTI files will be saved")
    parser.add_argument("--modalities", nargs="+", default=["IRM", "TEP", "TDM"],
                       help="List of modality subfolders to check and convert")
    parser.add_argument("--create-sample", metavar="PATH", 
                       help="Create a sample directory structure for testing")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.input_root or not args.output_root:
        parser.print_help()
        return
    
    try:
        # Convert modalities tuple
        modalities = tuple(args.modalities)
        
        print("=" * 60)
        print("DICOM to NIfTI Conversion Use Case")
        print("=" * 60)
        print(f"Input directory: {args.input_root}")
        print(f"Output directory: {args.output_root}")
        print(f"Modalities to process: {modalities}")
        print("=" * 60)
        
        # Process the subjects
        process_subjects(args.input_root, args.output_root, modalities)
        
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 

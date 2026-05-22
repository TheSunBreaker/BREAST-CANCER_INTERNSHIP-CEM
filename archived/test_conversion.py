#!/usr/bin/env python3
"""
Test script for DICOM to NIfTI conversion functionality.

This script provides a simple way to test the installation and functionality
of the breast cancer radiomics tools.

Author: esalasvilla
Date: 2025
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import SimpleITK as sitk
        print("SimpleITK imported successfully")
    except ImportError as e:
        print(f"Failed to import SimpleITK: {e}")
        return False
    
    try:
        from src.dcm2nii import dicom2nifti
        print("dcm2nii module imported successfully")
    except ImportError as e:
        print(f"Failed to import dcm2nii: {e}")
        return False
    
    try:
        from dcm2nii_use_case import process_subjects
        print("dcm2nii_use_case module imported successfully")
    except ImportError as e:
        print(f"Failed to import dcm2nii_use_case: {e}")
        return False
    
    try:
        from src.irm2nnunet import extract_irm_to_nnunet_flat
        print("irm2nnunet module imported successfully")
    except ImportError as e:
        print(f"Failed to import irm2nnunet: {e}")
        return False
    
    try:
        from irm2nnunet_use_case import main as irm2nnunet_main
        print("irm2nnunet_use_case module imported successfully")
    except ImportError as e:
        print(f"Failed to import irm2nnunet_use_case: {e}")
        return False
    
    return True

def test_simpleitk_version():
    """Test SimpleITK version and basic functionality."""
    try:
        import SimpleITK as sitk
        version = sitk.Version()
        print(f"SimpleITK version: {version}")
        
        # Test basic functionality
        image = sitk.Image(10, 10, 10, sitk.sitkUInt8)
        print("SimpleITK basic functionality test passed")
        return True
    except Exception as e:
        print(f"SimpleITK test failed: {e}")
        return False

def create_sample_structure():
    """Create a sample directory structure for testing."""
    print("\nCreating sample directory structure for testing...")
    
    # Create test directories
    test_dirs = [
        "test_data/input/subject_001/IRM",
        "test_data/input/subject_001/TEP",
        "test_data/input/subject_002/IRM",
        "test_data/output"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Create a dummy file to simulate DICOM structure
    for dir_path in test_dirs[:-1]:  # Exclude output directory
        dummy_file = Path(dir_path) / "dummy.txt"
        dummy_file.write_text("This is a dummy file for testing purposes.")
        print(f"✓ Created dummy file: {dummy_file}")
    
    print("✓ Sample directory structure created successfully")
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("Breast Cancer Radiomics - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\nImport tests failed. Please check your installation.")
        sys.exit(1)
    
    # Test SimpleITK
    if not test_simpleitk_version():
        print("\nSimpleITK tests failed. Please check your SimpleITK installation.")
        sys.exit(1)
    
    # Create sample structure
    create_sample_structure()
    
    print("\n" + "=" * 60)
    print("All tests passed! Installation is successful.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Place your DICOM files in the test_data/input/ directories")
    print("2. Run: python dcm2nii_use_case.py test_data/input test_data/output")
    print("3. Check the converted NIfTI files in test_data/output/")
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main() 

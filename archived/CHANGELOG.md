# Changelog

All notable changes to the Breast Cancer Radiomics Project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- DICOM to NIfTI conversion functionality
- Batch processing capabilities
- Comprehensive documentation
- Test scripts and examples

## [1.0.0] - 2025-01-XX

### Added
- Core DICOM to NIfTI conversion module (`src/dcm2nii.py`)
- IRM to nnUNet conversion module (`src/irm2nnunet.py`)
- Batch processing script for multiple subjects (`dcm2nii_use_case.py`)
- IRM to nnUNet conversion script (`irm2nnunet_use_case.py`)
- Comprehensive README documentation
- Requirements.txt with dependency specifications
- Setup.py for package installation
- Test script for functionality verification
- Contributing guidelines
- MIT License
- .gitignore for Python projects
- CHANGELOG.md for version tracking

### Features
- **DICOM Series Reading**: Automatic detection and reading of DICOM series
- **NIfTI Export**: Conversion to compressed NIfTI format (.nii.gz)
- **IRM to nnUNet Conversion**: Preparation of MRI data for deep learning segmentation
- **nnUNet Directory Structure**: Automatic creation of full nnUNet v2 folder structure
- **Batch Processing**: Process multiple subjects and modalities
- **Error Handling**: Robust error handling and validation
- **Command Line Interface**: Easy-to-use CLI for both single and batch operations
- **Cross-platform Support**: Works on Windows, macOS, and Linux

### Technical Details
- Uses SimpleITK for medical image processing
- Supports multiple modalities (IRM, TEP, TDM)
- Creates full nnUNet v2 directory structure
- Automatic output directory creation
- Configurable file naming conventions
- Progress reporting and error logging

### Documentation
- Installation instructions
- Usage examples
- API reference
- Contributing guidelines
- Research context and methodology

## [0.1.0] - 2024-12-XX

### Added
- Initial development version
- Basic DICOM to NIfTI conversion
- Simple command-line interface

---

## Version History

- **1.0.0**: First stable release with comprehensive documentation
- **0.1.0**: Initial development version

## Future Plans

### Planned for v1.1.0
- [ ] Add support for additional medical image formats
- [ ] Implement image preprocessing capabilities
- [ ] Add radiomics feature extraction
- [ ] Include machine learning model integration
- [ ] Add GUI interface option

### Planned for v1.2.0
- [ ] Advanced image registration
- [ ] Quality control automation
- [ ] Integration with clinical databases
- [ ] Performance optimizations
- [ ] Extended testing suite

---

For detailed information about each release, see the [GitHub releases page](https://github.com/your-username/breast-cancer-radiomics/releases). 

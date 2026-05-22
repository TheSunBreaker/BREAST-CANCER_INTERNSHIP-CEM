# Breast Cancer Radiomics Project

A comprehensive set of tools for neoadjuvant immunotherapy response prediction in triple negative breast cancer using radiomics extracted from MRI.

## 📋 Overview

This project provides tools for processing medical imaging data in breast cancer radiomics analysis pipelines, including:

1. **DICOM to NIfTI Conversion**: Essential preprocessing step for medical image analysis
2. **IRM to nnUNet Conversion**: Preparation of MRI data for deep learning segmentation with nnUNet
3. **nnUNet Inference**: Deep learning segmentation for tumor detection
4. **PCR Prediction**: Pathological Complete Response prediction from segmented images

These tools form a complete pipeline from raw DICOM images to PCR prediction, crucial for radiomics analysis pipelines in breast cancer research.

## 🏗️ Project Structure

```
BREAST-CANCER-RADIOMICS/
├── src/                          # Core functional modules
│   ├── dcm2nii.py              # DICOM to NIfTI conversion module
│   ├── irm2nnunet.py           # IRM to nnUNet conversion module
│   ├── nnunet_inference.py     # nnUNet inference module
│   └── pcr_prediction.py       # PCR prediction module
├── dcm2nii_use_case.py          # DICOM to NIfTI batch processing example
├── irm2nnunet_use_case.py       # IRM to nnUNet conversion example
├── nnunet_inference_use_case.py # nnUNet inference example
├── pcr_prediction_use_case.py   # PCR prediction example
├── complete_pipeline.py         # Complete pipeline from DICOM to PCR
├── test_conversion.py           # Installation test script
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation configuration
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contributing guidelines
├── CHANGELOG.md                # Version history
├── .gitignore                  # Git ignore rules
└── GITHUB_SUBMISSION_GUIDE.md # GitHub submission instructions
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager


## 📦 Dependencies

The main dependencies for this project are:

- **SimpleITK**: For medical image processing and DICOM handling
- **pathlib**: For cross-platform path handling (included in Python 3.4+)
- **argparse**: For command-line argument parsing (included in Python standard library)
- **nnunetv2**: For deep learning segmentation
- **numpy**: For numerical computations
- **scikit-learn**: For machine learning models
- **pandas**: For data manipulation
- **matplotlib**: For visualization
- **seaborn**: For statistical visualization

Install from the requirements file:

```bash
pip install -r requirements.txt
```

## 🌐 Environment Setup

### Creating and Activating Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv <myvenv>

# Activate on Windows PowerShell
.\<myvenv>\Scripts\activate

# Activate on Linux/Mac bash
source <myvenv>/bin/activate
```

### Screen Session Management

For long-running processes, use screen sessions to keep them running even if you disconnect:

```bash
# Create a new screen session with a name
screen -S mysession

# List all running screen sessions
screen -ls

# Reattach to a running session
screen -r mysession

# If only one screen running, you can just do:
screen -r

# If the session is "detached but not dead" and you want to force reattach:
screen -D -r mysession
```

**Screen Commands:**
- `Ctrl+A` then `D`: Detach from current session
- `Ctrl+A` then `C`: Create new window
- `Ctrl+A` then `N`: Next window
- `Ctrl+A` then `P`: Previous window
- `Ctrl+A` then `K`: Kill current session

## 🛠️ Usage

### 1. DICOM to NIfTI Conversion

#### Basic DICOM to NIfTI Conversion

```python
from src.dcm2nii import dicom2nifti

# Convert a single DICOM series
dicom2nifti("path/to/dicom/folder", "output/directory", prefix="subject_001")
```

#### Command Line Usage

```bash
# Convert a single DICOM series
python src/dcm2nii.py /path/to/dicom/folder /output/directory --prefix subject_001

# Batch process multiple subjects
python dcm2nii_use_case.py /path/to/input/root /path/to/output/root
```

#### DICOM to NIfTI Input Structure

```
input_root/
├── subject_001/
│   ├── IRM/          # MRI DICOM files
│   ├── TEP/          # PET DICOM files
│   └── TDM/          # CT DICOM files
├── subject_002/
│   ├── IRM/
│   └── TEP/
└── ...
```

#### DICOM to NIfTI Output Structure

```
output_root/
├── subject_001/
│   ├── subject_001_IRM.nii.gz
│   ├── subject_001_TEP.nii.gz
│   └── subject_001_TDM.nii.gz
├── subject_002/
│   ├── subject_002_IRM.nii.gz
│   └── subject_002_TEP.nii.gz
└── ...
```

### 2. IRM to nnUNet format Conversion

#### Basic IRM to nnUNet input format conversion

```python
from src.irm2nnunet import extract_irm_to_nnunet_flat

# Convert IRM files to nnUNet format
extract_irm_to_nnunet_flat(
    subjects_dir="/path/to/subjects",
    nnunet_root="/path/to/nnunet",
    dataset_id=1,
    irm_suffix="_IRM.nii.gz"
)
```

#### Command Line Usage

```bash
# Convert IRM files to nnUNet format
python irm2nnunet_use_case.py /path/to/subjects /path/to/nnunet --dataset-id 1
```

#### IRM to nnUNet Input structure

```
subjects_dir/
├── subject_001/
│   └── subject_001_IRM.nii.gz
├── subject_002/
│   └── subject_002_IRM.nii.gz
└── ...
```

#### IRM to nnUNet Output structure

```
nnunet_root/
└── nnunetv2/
    ├── nnUNet_raw/
    │   └── Dataset001/
    │       ├── imagesTr/
    │       │   ├── subject_001_0000.nii.gz
    │       │   └── subject_002_0000.nii.gz
    │       ├── labelsTr/
    │       └── imagesTs_pred3dfullres/
    ├── nnUNet_preprocessed/
    └── nnUNet_results/
        └── Dataset001/
            └── nnUNetTrainer__nnUNetPlans__3d_fullres/
                └── fold_0/
```

### 3. nnUNet Inference

#### Prerequisites Setup

Before running nnUNet inference, you need to set up the environment and install nnUNet:

```bash
# 0. If not created already, make an environment and activate it
python -m venv nnunet_env
source nnunet_env/bin/activate  # On Windows: nnunet_env\Scripts\activate

# 1. Install nnUNet
pip install nnunetv2

# 2. Set nnUNet folders as environment variables
export nnUNet_raw=~/nnUNet_raw
export nnUNet_preprocessed=~/nnUNet_preprocessed
export nnUNet_results=~/nnUNet_results
```

#### Dataset Preparation

```bash
# 3. Make sure the dataset.json file exists and is pasted in the nnUNet_raw folder
# The dataset.json should be in: ~/nnUNet_raw/Dataset001/dataset.json

# 4. Make sure to have the pretrained model in the nnUNet_results folder 
# with the dataset.json, dataset_fingerprint.json and plans.json files
# Structure should be: ~/nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

#### Running Inference

```bash
# 5. Run Infer command
nnUNetv2_predict \
    -d <DATASET_ID> \
    -i /path/to/imagesTs \
    -o /path/to/output \
    -c 3d_fullres \
    -f 0
```

**Parameters:**
- `-d <DATASET_ID>`: Dataset ID (e.g., 1 for Dataset001)
- `-i /path/to/imagesTs`: Path to the test images directory
- `-o /path/to/output`: Path to the output directory for predictions
- `-c 3d_fullres`: Configuration (3d_fullres for 3D full resolution)
- `-f 0`: Fold number (0 for single fold)

**Example:**
```bash
nnUNetv2_predict \
    -d 1 \
    -i ~/nnUNet_raw/Dataset001/imagesTs \
    -o ~/predictions \
    -c 3d_fullres \
    -f 0
```


### nnUNet Inference Input Structure

```
imagesTs/
├── subject_001_0000.nii.gz
├── subject_002_0000.nii.gz
└── ...
```

### nnUNet Inference Output Structure

```
predictions/
├── subject_001.nii.gz          # Segmentation mask
├── subject_002.nii.gz          # Segmentation mask
└── ...
```

### 4. PCR Prediction

#### Basic PCR Prediction

```python
from src.pcr_prediction import predict_pcr

# Predict PCR from segmented images
prediction = predict_pcr(
    segmented_images_dir="/path/to/segmented/images",
    model_path="/path/to/trained/model.pkl"
)
```

#### Command Line Usage

```bash
# Predict PCR from segmented images
python pcr_prediction_use_case.py /path/to/segmented/images /path/to/model /path/to/output
```

### PCR Prediction Input Structure

```
segmented_images/
├── subject_001/
│   ├── original.nii.gz         # Original MRI image
│   ├── segmentation.nii.gz     # Tumor segmentation mask
│   └── metadata.json           # Patient metadata
├── subject_002/
│   ├── original.nii.gz
│   ├── segmentation.nii.gz
│   └── metadata.json
└── ...
```### PCR Prediction Input Structure

```
segmented_images/
├── subject_001/
│   ├── original.nii.gz         # Original MRI image
│   ├── segmentation.nii.gz     # Tumor segmentation mask
│   └── metadata.json           # Patient metadata
├── subject_002/
│   ├── original.nii.gz
│   ├── segmentation.nii.gz
│   └── metadata.json
└── ...
```

### 5. Complete Pipeline

#### Running the Complete Pipeline

The complete pipeline processes DICOM images all the way to PCR prediction:

```bash
# Run complete pipeline from DICOM to PCR
python complete_pipeline.py \
    --input-dicom /path/to/dicom/root \
    --output-root /path/to/output \
    --dataset-id 1 \
    --nnunet-model-path /path/to/nnunet/model \
    --pcr-model-path /path/to/pcr/model
```

#### Pipeline Steps

1. **DICOM to NIfTI**: Convert DICOM files to NIfTI format
2. **IRM to nnUNet**: Prepare MRI data for nnUNet segmentation
3. **nnUNet Inference**: Perform tumor segmentation
4. **PCR Prediction**: Predict pathological complete response
5. **Results**: Generate comprehensive output with predictions and visualizations


#### Complete Pipeline in Console

Here's how to run the complete pipeline from DICOM to PCR prediction in the console:

```bash
# 1. Create and activate environment
python3 -m venv breast_cancer_env
source breast_cancer_env/bin/activate  # On Windows: .\breast_cancer_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install nnunetv2

# 3. Set up nnUNet environment variables
export nnUNet_raw=~/nnUNet_raw
export nnUNet_preprocessed=~/nnUNet_preprocessed
export nnUNet_results=~/nnUNet_results

# 4. Run complete pipeline
python complete_pipeline.py \
    --input-dicom /path/to/dicom/images \
    --output-root /path/to/results \
    --dataset-id 1 \
    --nnunet-model-path /path/to/nnunet/model \
    --pcr-model-path /path/to/pcr/model
```

### Complete Pipeline Output Structure

```
pipeline_output/
├── intermediate_results/
│   ├── nifti_conversions/      # DICOM to NIfTI outputs
│   ├── nnunet_prepared/        # nnUNet format data
│   └── segmentations/          # nnUNet inference results
├── final_results/
│   ├── pcr_predictions.csv     # PCR predictions for all subjects
│   ├── confidence_scores.csv   # Prediction confidence scores
│   └── visualizations/         # Generated plots and images
├── logs/
│   ├── conversion.log          # Conversion process logs
│   ├── inference.log           # nnUNet inference logs
│   └── prediction.log          # PCR prediction logs
└── summary_report.html         # Comprehensive HTML report
```

## 🧪 Testing

To test the conversion processes:

1. Create a sample directory structure with DICOM files
2. Run the conversion scripts
3. Verify the output files

```bash
# Test DICOM to NIfTI conversion
python dcm2nii_use_case.py ./test_data/input ./test_data/output

# Test IRM to nnUNet conversion
python irm2nnunet_use_case.py ./test_data/output ./test_data/nnunet --dataset-id 1

# Test nnUNet inference
python nnunet_inference_use_case.py 1 ./test_data/nnunet/imagesTs ./test_data/predictions

# Test PCR prediction
python pcr_prediction_use_case.py ./test_data/predictions ./test_data/pcr_model ./test_data/pcr_output

# Test complete pipeline
python complete_pipeline.py \
    --input-dicom ./test_data/input \
    --output-root ./test_data/pipeline_output \
    --dataset-id 1 \
    --nnunet-model-path ./test_data/nnunet_model \
    --pcr-model-path ./test_data/pcr_model
```

### Testing with Sample Data

```bash
# Create test environment
python3 -m venv test_env
source test_env/bin/activate  # On Windows: .\test_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_conversion.py
```


### Step-by-Step Processing

```bash
# Step 1: DICOM to NIfTI
python dcm2nii_use_case.py /path/to/dicom /path/to/nifti

# Step 2: Prepare for nnUNet
python irm2nnunet_use_case.py /path/to/nifti /path/to/nnunet --dataset-id 1

# Step 3: Run nnUNet inference
nnUNetv2_predict -d 1 -i /path/to/nnunet/imagesTs -o /path/to/segments -c 3d_fullres -f 0

# Step 4: PCR prediction
python pcr_prediction_use_case.py /path/to/segments /path/to/pcr_model /path/to/predictions
```

### Batch Processing Multiple Datasets

```bash
# Process multiple datasets
for dataset_id in 1 2 3; do
    echo "Processing Dataset00${dataset_id}"
    python complete_pipeline.py \
        --input-dicom /path/to/dicom/dataset${dataset_id} \
        --output-root /path/to/results/dataset${dataset_id} \
        --dataset-id ${dataset_id} \
        --nnunet-model-path /path/to/nnunet/model \
        --pcr-model-path /path/to/pcr/model
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **esalasvilla** - *Initial work* - [elianasv]

## 🙏 Acknowledgments

- SimpleITK development team for the excellent medical image processing library
- nnUNet development team for the segmentation framework
- The medical imaging community for open-source contributions

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@software{breast_cancer_radiomics_2025,
  author = {esalasvilla},
  title = {Breast Cancer Radiomics Project},
  year = {2025},
  url = {https://github.com/elianasv/BREAST-CANCER-RADIOMICS}
}
```

## 📞 Support

For questions and support, please open an issue on GitHub or contact elianasv1400@gmail.com.

---

**Note**: This project is designed for research purposes. Always ensure compliance with local regulations and institutional review board requirements when working with medical data. 

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="breast-cancer-radiomics",
    version="1.0.0",
    author="esalasvilla",
    author_email="your.email@example.com",
    description="Tools for breast cancer radiomics analysis and DICOM to NIfTI conversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/breast-cancer-radiomics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dcm2nii=src.dcm2nii:main",
            "dcm2nii-batch=dcm2nii_use_case:main",
            "irm2nnunet=irm2nnunet_use_case:main",
        ],
    },
    keywords="medical imaging, radiomics, breast cancer, DICOM, NIfTI, SimpleITK",
    project_urls={
        "Bug Reports": "https://github.com/your-username/breast-cancer-radiomics/issues",
        "Source": "https://github.com/your-username/breast-cancer-radiomics",
        "Documentation": "https://github.com/your-username/breast-cancer-radiomics#readme",
    },
) 

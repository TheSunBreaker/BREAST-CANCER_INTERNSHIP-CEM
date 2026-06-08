#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Analyse de Segmentation nnUNet Suspecte
===============================================================================

OBJECTIF
---------
Fournir des métriques permettant d'évaluer si une prédiction nnUNet
correspond à une lésion crédible ou à un faux positif probable.

Le script calcule :

- Nombre total de voxels segmentés
- Volume physique
- Nombre de composantes connexes
- Taille des composantes
- Bounding box
- SUV moyen / max dans la segmentation
- Comparaison au fond PET
- Indicateurs de suspicion

ENTREES
--------
mask.nii.gz : segmentation prédite
pet.nii.gz  : PET aligné sur le masque

SORTIE
-------
Rapport texte détaillé.
===============================================================================
"""

import nibabel as nib
import numpy as np

from scipy.ndimage import label


# ============================================================================
# PARAMETRES
# ============================================================================

MASK_PATH = "prediction.nii.gz"
PET_PATH = "pet.nii.gz"

# seuil empirique sous lequel on considère la prédiction très petite
SMALL_VOLUME_ML = 0.05

# nombre minimum de voxels considéré comme crédible
MIN_VOXELS = 50


# ============================================================================
# CHARGEMENT
# ============================================================================

mask_nii = nib.load(MASK_PATH)
pet_nii = nib.load(PET_PATH)

mask = mask_nii.get_fdata() > 0
pet = pet_nii.get_fdata()

spacing = mask_nii.header.get_zooms()[:3]

voxel_volume_mm3 = np.prod(spacing)

print("\n==============================")
print("ANALYSE SEGMENTATION")
print("==============================\n")


# ============================================================================
# VOLUME GLOBAL
# ============================================================================

n_voxels = int(mask.sum())

volume_mm3 = n_voxels * voxel_volume_mm3
volume_ml = volume_mm3 / 1000.0

print("----- Volume -----")
print(f"Nb voxels       : {n_voxels}")
print(f"Volume (mm³)    : {volume_mm3:.2f}")
print(f"Volume (mL)     : {volume_ml:.5f}")
print()


# ============================================================================
# COMPOSANTES CONNEXES
# ============================================================================

labeled, n_components = label(mask)

print("----- Composantes connexes -----")
print(f"Nombre : {n_components}")

component_sizes = []

for i in range(1, n_components + 1):
    size = np.sum(labeled == i)
    component_sizes.append(size)

component_sizes = sorted(component_sizes, reverse=True)

for idx, size in enumerate(component_sizes, start=1):
    comp_ml = size * voxel_volume_mm3 / 1000
    print(
        f"Composante {idx:02d} : "
        f"{size} voxels "
        f"({comp_ml:.5f} mL)"
    )

print()


# ============================================================================
# PLUS GRANDE COMPOSANTE
# ============================================================================

if len(component_sizes) > 0:

    largest = component_sizes[0]

    ratio = largest / n_voxels

    print("----- Cohérence -----")
    print(
        f"Part de la plus grosse composante : "
        f"{100*ratio:.2f}%"
    )
    print()


# ============================================================================
# BOUNDING BOX
# ============================================================================

if n_voxels > 0:

    coords = np.argwhere(mask)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    bbox_size_vox = maxs - mins + 1

    bbox_size_mm = bbox_size_vox * spacing

    print("----- Bounding Box -----")

    print(
        f"Dimensions voxels : "
        f"{bbox_size_vox}"
    )

    print(
        f"Dimensions mm : "
        f"{bbox_size_mm.round(2)}"
    )

    print()


# ============================================================================
# ANALYSE PET
# ============================================================================

if n_voxels > 0:

    lesion_values = pet[mask]

    suv_mean = lesion_values.mean()
    suv_max = lesion_values.max()
    suv_std = lesion_values.std()

    print("----- Intensités PET -----")

    print(f"SUV mean : {suv_mean:.3f}")
    print(f"SUV max  : {suv_max:.3f}")
    print(f"SUV std  : {suv_std:.3f}")

    print()

    # fond = reste du volume
    background = pet[~mask]

    bg_mean = background.mean()
    bg_std = background.std()

    print("----- Contraste fond -----")

    print(f"Fond mean : {bg_mean:.3f}")
    print(f"Fond std  : {bg_std:.3f}")

    if bg_std > 0:

        z_score = (
            suv_mean - bg_mean
        ) / bg_std

        print(
            f"Z-score vs fond : "
            f"{z_score:.2f}"
        )

    print()


# ============================================================================
# INTERPRETATION
# ============================================================================

print("----- Evaluation -----")

suspicions = []

if n_voxels < MIN_VOXELS:
    suspicions.append(
        "Très faible nombre de voxels"
    )

if volume_ml < SMALL_VOLUME_ML:
    suspicions.append(
        "Volume extrêmement petit"
    )

if n_components > 3:
    suspicions.append(
        "Nombre élevé de composantes"
    )

if len(component_sizes) > 0:

    largest_ratio = component_sizes[0] / n_voxels

    if largest_ratio < 0.50:
        suspicions.append(
            "Prédiction fragmentée"
        )

if len(suspicions) == 0:

    print(
        "Aucun indicateur majeur "
        "de faux positif."
    )

else:

    print(
        "Indicateurs suggérant un faux positif :"
    )

    for s in suspicions:
        print(f" - {s}")

print("\nAnalyse terminée.\n")

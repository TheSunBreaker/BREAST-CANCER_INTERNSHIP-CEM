#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Analyse de Segmentation nnUNet avec SimpleITK
===============================================================================

OBJECTIF
--------
Évaluer la crédibilité d'une segmentation prédite par nnUNet en calculant :

    • Nombre total de voxels segmentés
    • Volume physique (mm³ et mL)
    • Nombre de composantes connexes
    • Taille des composantes
    • Bounding Box physique
    • SUV moyen / max / écart-type
    • Contraste par rapport au fond PET
    • Indicateurs simples de suspicion de faux positif

ENTRÉES
-------
prediction.nii.gz : masque binaire prédit
pet.nii.gz        : volume PET co-enregistré

SORTIE
------
Rapport texte détaillé dans le terminal.

DÉPENDANCES
-----------
pip install SimpleITK numpy

===============================================================================
"""

import SimpleITK as sitk
import numpy as np


# =============================================================================
# PARAMÈTRES
# =============================================================================

MASK_PATH = "prediction.nii.gz"
PET_PATH = "pet.nii.gz"

# seuil empirique de très petit volume
SMALL_VOLUME_ML = 0.05

# nombre minimal de voxels considéré comme crédible
MIN_VOXELS = 50


# =============================================================================
# CHARGEMENT DES IMAGES
# =============================================================================

print("\n==============================")
print("ANALYSE SEGMENTATION")
print("==============================\n")

mask_img = sitk.ReadImage(MASK_PATH)
pet_img = sitk.ReadImage(PET_PATH)

# Conversion en tableaux NumPy
#
# Remarque :
# SimpleITK retourne les tableaux sous la forme :
#
#     [z, y, x]
#
mask = sitk.GetArrayFromImage(mask_img) > 0
pet = sitk.GetArrayFromImage(pet_img)

# spacing physique (x, y, z)
spacing = np.array(mask_img.GetSpacing())

# volume d'un voxel en mm³
voxel_volume_mm3 = np.prod(spacing)


# =============================================================================
# VOLUME GLOBAL
# =============================================================================

n_voxels = int(mask.sum())

volume_mm3 = n_voxels * voxel_volume_mm3
volume_ml = volume_mm3 / 1000.0

print("----- Volume -----")
print(f"Nb voxels       : {n_voxels}")
print(f"Volume (mm³)    : {volume_mm3:.2f}")
print(f"Volume (mL)     : {volume_ml:.5f}")
print()


# =============================================================================
# COMPOSANTES CONNEXES
# =============================================================================

#
# Création d'une image binaire SITK
#
mask_sitk = sitk.Cast(mask_img > 0, sitk.sitkUInt8)

#
# Étiquetage des composantes connexes
#
cc = sitk.ConnectedComponent(mask_sitk)

#
# Calcul des statistiques de chaque composante
#
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

n_components = stats.GetNumberOfLabels()

print("----- Composantes connexes -----")
print(f"Nombre : {n_components}")

component_sizes = []

for label_id in stats.GetLabels():

    size = stats.GetNumberOfPixels(label_id)

    component_sizes.append(size)

component_sizes.sort(reverse=True)

for idx, size in enumerate(component_sizes, start=1):

    comp_ml = size * voxel_volume_mm3 / 1000.0

    print(
        f"Composante {idx:02d} : "
        f"{size} voxels "
        f"({comp_ml:.5f} mL)"
    )

print()


# =============================================================================
# COHÉRENCE DE LA SEGMENTATION
# =============================================================================

if component_sizes:

    largest = component_sizes[0]

    ratio = largest / n_voxels

    print("----- Cohérence -----")
    print(
        f"Part de la plus grosse composante : "
        f"{100 * ratio:.2f}%"
    )
    print()


# =============================================================================
# BOUNDING BOX
# =============================================================================

if n_components > 0:

    #
    # Bounding box de la plus grande composante
    #
    largest_label = max(
        stats.GetLabels(),
        key=lambda x: stats.GetNumberOfPixels(x)
    )

    bbox = stats.GetBoundingBox(largest_label)

    #
    # Format :
    # (x_min, y_min, z_min, size_x, size_y, size_z)
    #
    bbox_size_vox = np.array(bbox[3:])

    bbox_size_mm = bbox_size_vox * spacing

    print("----- Bounding Box -----")

    print(
        f"Dimensions voxels : "
        f"{bbox_size_vox}"
    )

    print(
        f"Dimensions mm : "
        f"{np.round(bbox_size_mm, 2)}"
    )

    print()


# =============================================================================
# ANALYSE DES INTENSITÉS PET
# =============================================================================

if n_voxels > 0:

    lesion_values = pet[mask]

    suv_mean = float(np.mean(lesion_values))
    suv_max = float(np.max(lesion_values))
    suv_std = float(np.std(lesion_values))

    print("----- Intensités PET -----")

    print(f"SUV mean : {suv_mean:.3f}")
    print(f"SUV max  : {suv_max:.3f}")
    print(f"SUV std  : {suv_std:.3f}")

    print()

    # -------------------------------------------------------------------------
    # Fond PET
    # -------------------------------------------------------------------------

    background = pet[~mask]

    bg_mean = float(np.mean(background))
    bg_std = float(np.std(background))

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


# =============================================================================
# INTERPRÉTATION AUTOMATIQUE
# =============================================================================

print("----- Évaluation -----")

suspicions = []

#
# Très peu de voxels
#
if n_voxels < MIN_VOXELS:

    suspicions.append(
        "Très faible nombre de voxels"
    )

#
# Très petit volume
#
if volume_ml < SMALL_VOLUME_ML:

    suspicions.append(
        "Volume extrêmement petit"
    )

#
# Segmentation dispersée
#
if n_components > 3:

    suspicions.append(
        "Nombre élevé de composantes"
    )

#
# Fragmentation importante
#
if component_sizes:

    largest_ratio = component_sizes[0] / n_voxels

    if largest_ratio < 0.50:

        suspicions.append(
            "Prédiction fortement fragmentée"
        )

#
# Rapport final
#
if not suspicions:

    print(
        "Aucun indicateur majeur "
        "de faux positif détecté."
    )

else:

    print(
        "Indicateurs suggérant un faux positif :"
    )

    for item in suspicions:

        print(f" - {item}")

print("\nAnalyse terminée.\n")

#!/usr/bin/env python3
"""
Module de Standardisation Spatiale (Idempotent).
Vérifie et corrige la géométrie des images médicales (Alignement, Espacement, Origine).
Si les images sont déjà parfaites (ex: données de Challenge), il ne fait rien.
"""

import SimpleITK as sitk
import numpy as np

def check_geometry_match(img_ref: sitk.Image, img_test: sitk.Image, tol: float = 1e-3) -> bool:
    """
    Vérifie si deux images partagent STRICTEMENT la même grille spatiale.
    """
    if img_ref.GetSize() != img_test.GetSize():
        return False
    if not np.allclose(img_ref.GetSpacing(), img_test.GetSpacing(), atol=tol):
        return False
    if not np.allclose(img_ref.GetOrigin(), img_test.GetOrigin(), atol=tol):
        return False
    if not np.allclose(img_ref.GetDirection(), img_test.GetDirection(), atol=tol):
        return False
    return True

def resample_to_reference(
    moving_img: sitk.Image, 
    ref_img: sitk.Image, 
    is_mask: bool = False, 
    pad_value: float = 0.0
) -> sitk.Image:
    """
    Ré-échantillonne une image (moving_img) pour qu'elle calque parfaitement 
    la grille spatiale de l'image de référence (ref_img).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(pad_value)
    
    if is_mask:
        # Interpolation stricte pour les masques (pas de valeurs intermédiaires)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Interpolation fluide (BSpline est généralement meilleur que Linear pour la 3D)
        resampler.SetInterpolator(sitk.sitkBSpline)
        
    return resampler.Execute(moving_img)

def enforce_strict_alignment(
    ref_path: str, 
    moving_path: str, 
    out_path: str, 
    is_mask: bool = False, 
    pad_value: float = 0.0
) -> bool:
    """
    Fonction de haut niveau : Charge, vérifie, corrige si besoin, et sauvegarde.
    Retourne True si un ré-échantillonnage a eu lieu, False si l'image était déjà alignée.
    """
    ref_img = sitk.ReadImage(ref_path)
    # On force le chargement des images en Float32, ou UInt8 pour les masques
    pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    moving_img = sitk.ReadImage(moving_path, pixel_type)
    
    # 1. Vérification
    is_aligned = check_geometry_match(ref_img, moving_img)
    
    # 2. Action conditionnelle (Idempotence)
    if is_aligned:
        # Si c'est déjà parfait, on se contente de copier (ou de sauvegarder tel quel)
        sitk.WriteImage(moving_img, out_path)
        return False # Pas de modification faite
    else:
        # 3. Correction
        aligned_img = resample_to_reference(moving_img, ref_img, is_mask, pad_value)
        
        # Sécurité supplémentaire pour les masques : forcer la binarisation
        if is_mask:
            aligned_img = sitk.Cast(aligned_img > 0, sitk.sitkUInt8)
            
        sitk.WriteImage(aligned_img, out_path)
        return True # Une correction a été appliquée

def clean_and_binarize_mask(mask_path: str, out_path: str):
    """
    S'assure qu'un masque ne contient que des 0 et des 1, et est au format UInt8 (Requis par nnU-Net).
    """
    mask = sitk.ReadImage(mask_path)
    clean_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    sitk.WriteImage(clean_mask, out_path)

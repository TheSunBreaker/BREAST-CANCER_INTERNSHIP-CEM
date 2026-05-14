#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
  Évaluateur de Métriques de Segmentation (SimpleITK + MONAI)
===============================================================================
Rôle :
  Calcule le Dice, la distance de Hausdorff (HD) et la distance de 
  Hausdorff à X% (HD95 par défaut) entre deux dossiers de masques NIfTI.
  
  Prend en charge les "masques vides" (pCR / Réponse complète) de manière
  sécurisée sans faire crasher les algorithmes de distance.

Dépendances :
  - SimpleITK
  - numpy
  - torch
  - monai

Structure Attendue :
  Dossier A (Ground Truth) /        Dossier B (Prédictions) /
    ├── DUKE_001.nii.gz               ├── DUKE_001.nii.gz
    ├── DUKE_002.nii.gz               ├── DUKE_002.nii.gz
    └── ...                           └── ...
  *(Les fichiers doivent avoir EXACTEMENT le même nom dans les deux dossiers)*

Sorties Générées :
  - Un affichage console du résumé statistique (Moyenne, Écart-type).
  - (Optionnel) Un fichier CSV détaillé :
    metrics_results.csv
      id,dice,hd,hdp
      DUKE_001,0.89,5.12,2.05
      DUKE_002,0.92,3.00,1.50
      ...
===============================================================================
"""

import os
import csv
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple, Dict, Optional
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def list_pairs(dir_a: str, dir_b: str, exts: Tuple[str, ...] = (".nii.gz", ".nii")) -> List[Tuple[str, str, str]]:
    """
    Identifie les fichiers correspondants entre le dossier GT (A) et Prédiction (B).
    Génère des alertes si des fichiers sont orphelins dans l'un des dossiers.
    """
    A = {f: os.path.join(dir_a, f) for f in os.listdir(dir_a) if f.endswith(exts)}
    B = {f: os.path.join(dir_b, f) for f in os.listdir(dir_b) if f.endswith(exts)}
    
    common = sorted(set(A.keys()) & set(B.keys()))

    if not common:
        raise ValueError("ERREUR FATALE : Aucun nom de fichier commun entre les deux dossiers.")

    missing_in_b = sorted(set(A.keys()) - set(B.keys()))
    missing_in_a = sorted(set(B.keys()) - set(A.keys()))
    
    if missing_in_b:
        print(f"[ALERTE] {len(missing_in_b)} fichier(s) présent(s) uniquement dans Ground Truth (ignorés). Ex: {missing_in_b[:3]}")
    if missing_in_a:
        print(f"[ALERTE] {len(missing_in_a)} fichier(s) présent(s) uniquement dans Prédictions (ignorés). Ex: {missing_in_a[:3]}")

    return [(fn, A[fn], B[fn]) for fn in common]


def load_mask_as_tensor(path: str, binarize_thr: float = 0.0) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """
    Charge un masque NIfTI via SimpleITK et le formate pour MONAI.
    
    Retourne :
        - tenseur PyTorch de dimension (B=1, C=1, Z, Y, X)
        - espacement en millimètres ordonné en (Z, Y, X)
    """
    img = sitk.ReadImage(path)
    
    # sitk.GetArrayFromImage retourne une matrice numpy en (Z, Y, X)
    arr = sitk.GetArrayFromImage(img)
    mask = (arr > binarize_thr).astype(np.float32)
    
    # Formatage requis par MONAI : Batch, Channel, Spatial_Dims
    t = torch.from_numpy(mask)[None, None, ...]  # Devient (1, 1, Z, Y, X)
    
    # ATTENTION PIÈGE SIMPLEITK :
    # L'espacement renvoyé par GetSpacing() est toujours (X, Y, Z).
    # Comme notre matrice numpy est en (Z, Y, X), nous DEVONS inverser l'espacement
    # pour que MONAI calcule les distances physiques correctement.
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    
    return t, spacing_zyx


def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    include_background: bool = False,
    hd_percentile: int = 95
) -> Tuple[float, float, float]:
    """
    Calcule le Dice, HD et HD95 pour une paire d'images.
    Gère explicitement les cas de "Réponse Complète" (masques vides).
    """
    gt_t, spacing_gt = load_mask_as_tensor(gt_path)
    pr_t, spacing_pr = load_mask_as_tensor(pred_path)

    # 1. Vérifications d'intégrité spatiale
    if gt_t.shape != pr_t.shape:
        raise ValueError(f"Incohérence de dimensions : GT {tuple(gt_t.shape)} vs Pred {tuple(pr_t.shape)}")

    if not np.allclose(spacing_gt, spacing_pr, atol=1e-5):
        print(f" [!] Espacement différent détecté pour {os.path.basename(gt_path)}: "
              f"GT {spacing_gt} vs Pred {spacing_pr}. L'espacement de la GT sera utilisé.")

    # 2. GESTION SÉCURISÉE DES MASQUES VIDES (Cas de Réponse Complète - pCR)
    # Si la vérité terrain OU la prédiction est vide, la distance de Hausdorff est infinie
    gt_empty = (gt_t.sum() == 0)
    pr_empty = (pr_t.sum() == 0)

    if gt_empty and pr_empty:
        # Les deux sont vides : L'algorithme a eu parfaitement raison (Vrai Négatif absolu)
        return 1.0, 0.0, 0.0
    
    if gt_empty or pr_empty:
        # L'un est vide mais pas l'autre (Faux Positif ou Faux Négatif majeur)
        # Dice = 0. HD = NaN car la distance entre "quelque chose" et "rien" n'est pas mesurable.
        return 0.0, float('nan'), float('nan')

    # 3. Calculs standards MONAI (si les deux masques contiennent au moins un pixel)
    
    # --- Dice ---
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    dice = dice_metric(pr_t, gt_t).item()

    # --- HD (Hausdorff Maximum = 100ème percentile) ---
    hd_metric = HausdorffDistanceMetric(include_background=include_background, percentile=100)
    hd_val = hd_metric(pr_t, gt_t, spacing=spacing_gt).item()

    # --- HDp (Robust Hausdorff, ex: 95ème percentile) ---
    hd_p_metric = HausdorffDistanceMetric(include_background=include_background, percentile=hd_percentile)
    hd_p_val = hd_p_metric(pr_t, gt_t, spacing=spacing_gt).item()

    return float(dice), float(hd_val), float(hd_p_val)


def mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Calcule la moyenne et l'écart-type d'une liste en ignorant les NaN
    (Les NaN proviennent des masques vides).
    """
    # np.nanmean et np.nanstd ignorent gracieusement les 'nan'
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)] # Retire les nan
    
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
        
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def compute_metrics_batch(
    folder_a: str,
    folder_b: str,
    include_background: bool = False,
    hd_percentile: int = 95,
    save_csv: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Orchestre le calcul pour l'ensemble du dataset.
    """
    pairs = list_pairs(folder_a, folder_b)

    rows = []
    dice_vals, hd_vals, hdp_vals = [], [], []

    print(f"\nDébut de l'évaluation sur {len(pairs)} cas...\n")

    for fn, pa, pb in pairs:
        try:
            d, h, hp = compute_metrics_for_pair(
                gt_path=pa,
                pred_path=pb,
                include_background=include_background,
                hd_percentile=hd_percentile
            )
            case_id = os.path.splitext(os.path.splitext(fn)[0])[0]  # Enlève .nii puis .gz
            
            rows.append({"id": case_id, "dice": d, "hd": h, "hdp": hp})
            dice_vals.append(d)
            hd_vals.append(h)
            hdp_vals.append(hp)
            
            print(f"[{case_id}] Dice: {d:.4f} | HD: {h:.2f} mm | HD{hd_percentile}: {hp:.2f} mm")
            
        except Exception as e:
            print(f" [!] ERREUR sur {fn}: {e}")

    # Résumé Statistique (Les NaN sont ignorés dans le calcul de la moyenne)
    dm, ds = mean_std(dice_vals)
    hm, hs = mean_std(hd_vals)
    hpm, hps = mean_std(hdp_vals)

    summary = {
        "dice_mean": dm, "dice_std": ds,
        "hd_mean": hm, "hd_std": hs,
        "hdp_mean": hpm, "hdp_std": hps,
        "hd_percentile": hd_percentile,
        "include_background": include_background,
        "n": len(rows),
    }

    if save_csv and rows:
        # Création du dossier parent du CSV si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "dice", "hd", "hdp"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n[SUCCÈS] CSV détaillé sauvegardé : {save_csv}")

    return rows, summary


def main():
    ap = argparse.ArgumentParser(
        description="Calcule les métriques de segmentation (Dice, HD, HD95) entre deux dossiers de masques."
    )
    ap.add_argument("folder_a", help="Dossier A : Ground Truth (ex: labelsTs)")
    ap.add_argument("folder_b", help="Dossier B : Prédictions (ex: nnUNet_results)")
    ap.add_argument("--percentile", type=int, default=95, help="Percentile pour le Hausdorff Robuste (Défaut: 95)")
    ap.add_argument("--include-background", action="store_true", help="Inclure la classe fond (Background) dans les métriques")
    ap.add_argument("--csv", default="./metrics_results.csv", help="Chemin du fichier CSV de sortie (Défaut: ./metrics_results.csv)")
    args = ap.parse_args()

    _, summary = compute_metrics_batch(
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        include_background=args.include_background,
        hd_percentile=args.percentile,
        save_csv=args.csv,
    )

    print("\n" + "="*40)
    print("             BILAN GLOBAL             ")
    print("="*40)
    print(f" Cas évalués : {summary['n']}")
    print(f" Dice Moyen  : {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f" HD Max      : {summary['hd_mean']:.2f} mm ± {summary['hd_std']:.2f} mm")
    print(f" HD{summary['hd_percentile']}        : {summary['hdp_mean']:.2f} mm ± {summary['hdp_std']:.2f} mm")
    print("="*40)


if __name__ == "__main__":
    main()

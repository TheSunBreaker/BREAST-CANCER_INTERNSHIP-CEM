#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================================
  Extracteur de Masques DICOM (V5 - Version Ultime, Fusion V3+V4).
===============================================================================

ATTENTION : IL EXISTE UN BOOLEAN VAR GLOBALE QUI SPECIFIE SI ON VEUT DIFFERENCIER DE POTENTIELS GANGLIONS (2) DE TUMEURS (1) OU TOUS
LES CONSIDERER COMME DES (1), QUI EST A 'FAUX' NORMALEMENT. (SI CHANGE, LE PRECISER ICI). 
DE PLUS, SI ON PEUT FUSIONNER LES MASQUES MULTIPLES (DONC SHAPE SIMIAIRES), ON FUSIONNE. SINON, ON LAISSE SEPARES CAR ON PAR DU PRINCIPE
QUE L'UN EST CALE SUR PET, L'AUTRE SUR CT (CAS PET ET CT). ET ALORS, L'ORCHESTRATEUR INTEGRE AU SCRIPT DE MISE EN STRUCTURE PET_CT VERS
NNUNET, PRENDRA LE PREMIER MASQUE DES 2 QU'IL TROUVERA ET PROCEDERA AINSI AU CALAGE DU MASQUE SUR LE PET. MAIS CETTE ACTION SUPPOSE
QUE DEUX MASQUES FUIONNABLES DE SHAPES DIFFERENTES SONT UN MEME MASQUE VERSION SHAPES DE PET ET VERSION SHAPE DE CT

HISTORIQUE DES VERSIONS :
  V3 : Analyse automatique multi-masques SEG (Dice, doublons, full-body).
       Logging détaillé par étapes (ÉTAPE 1/2/3). Rapport global partagé.
  V4 : Support RTSTRUCT via Plastimatch. Fusion sémantique multi-classes.
       Auto-labélisation tumeur (1) / ganglion (2). Tempdir pour les pixels.
  V5 : FUSION V3 + V4.
       - Tous les bugs de V4 corrigés (NameError data_a/data_b, shape check
         absent dans la boucle overlap, masques illisibles non comptés).
       - Logging détaillé V3 réintégré (stats par fichier, en-têtes d'étapes,
         message [DISTINCT] pour les structures à faible Dice).
       - Gestion robuste des erreurs : un masque illisible est explicitement
         compté comme rejeté et tracé dans le rapport.
       - Support RTSTRUCT (V4) conservé intégralement.
       - Fusion sémantique multi-classes (V4) conservée intégralement.
       - Auto-labélisation tumeur/ganglion (V4) conservée intégralement.

PIPELINE GLOBAL :
  1. ingesteur_v6.py      → groupe les DICOMs, génère les NIfTI images,
                            copie les SEG/RTSTRUCT bruts dans dicom_mask_*/
  2. extract_masks_v5.py  → lit les masques bruts, les analyse si multiples,
                            fusionne les lésions distinctes, génère les NIfTI
                            masques typés (classe 1 = tumeur, 2 = ganglion).

STRUCTURE DES DOSSIERS ATTENDUE (générée par l'ingesteur) :
  <project_root>/
    DUKE_001/
      dicom_mask_rm/          ← SEG/RTSTRUCT bruts baseline IRM
        DUKE_001_A1B2C/
          fichier.dcm
      dicom_mask_rm_20230514_1430/  ← suivi longitudinal
        ...
      imgs/                   ← NIfTI images de référence (pour Plastimatch)
        DUKE_001_T1.nii.gz
      imgs_20230514_1430/
        ...
    DUKE_002/
      ...

SORTIES GÉNÉRÉES :
  <patient_dir>/mask/
    <patient>_mask_<uid>_<dcm>.nii.gz      → masque individuel (1 seule lésion)
    <patient>_mask_FUSED.nii.gz            → masque fusionné (lésions multiples)
    a_verifier/                            → masques à inspecter manuellement
  rapport_analyse_masques_v5.txt           → rapport global à la racine

===============================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import glob
import shutil
import tempfile
import subprocess
import argparse
from itertools import combinations
from datetime import datetime

import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk


# ==============================================================================
# SECTION 1 — CONFIGURATION GLOBALE
# ==============================================================================
# Tous les seuils décisionnels sont centralisés ici pour faciliter
# la calibration sans avoir à fouiller dans le code.

# ── Seuil "full-body" ─────────────────────────────────────────────────────────
# Un masque dont le ratio (voxels segmentés / voxels totaux) dépasse ce seuil
# est considéré comme aberrant : une tumeur mammaire ou ganglion ne peut pas
# occuper 50 % d'un volume IRM ou PET entier.
# Ajuster si le champ de vue est très petit (ex : IRM focalisée sein).
FULL_BODY_THRESHOLD = 0.50

# ── Seuil "doublon" (Dice ≥ seuil → masques quasi identiques) ─────────────────
# Au-delà de 0.95, les deux masques couvrent essentiellement les mêmes voxels.
# On ne conserve que le plus petit (heuristique oncologique : segmentation plus
# conservative = moins de tissu sain inclus = plus précise).
DICE_DUPLICATE_THRESHOLD = 0.95

# ── Seuil "overlap suspect" (Dice entre seuil et DUPLICATE) ───────────────────
# Un Dice entre 0.20 et 0.95 indique un chevauchement partiel non négligeable.
# On ne peut pas décider automatiquement : cela peut être de la variabilité
# inter-annotateur, deux lésions partiellement superposées ou une erreur.
# Ces masques sont copiés dans "a_verifier/" pour inspection radiologique.
DICE_OVERLAP_THRESHOLD = 0.20

# --- NOUVEAU : Stratégie de résolution des doublons (Dice ≥ 0.95) ---
# True  : Conserve le masque le plus RÉCENT. On suppose que si le médecin a 
#         généré un nouveau masque identique à 95%, c'est une correction/affinement.
# False : Conserve le masque le plus PETIT. (Approche conservative historique 
#         pour minimiser l'inclusion de tissu sain).
KEEP_NEWEST_DUPLICATE_MASK = True

# ── Chemin vers l'exécutable Plastimatch ──────────────────────────────────────
# Plastimatch est requis UNIQUEMENT pour les fichiers RTSTRUCT.
# Pour les SEG purs, pydicom-seg suffit et Plastimatch n'est pas appelé.
# Adapter ce chemin à votre installation (Linux : "/usr/bin/plastimatch",
# Windows : chemin complet vers plastimatch.exe).
PLASTIMATCH_EXE = r"C:\Users\coul0426\plastimatch_portable\Plastimatch\bin\plastimatch.exe"

# ── Mots-clés pour l'auto-labélisation ganglion ───────────────────────────────
# Comparaison insensible à la casse. Toute description contenant l'un de ces
# termes recevra la classe 2 (ganglion lymphatique). Sinon : classe 1 (tumeur).
NODE_KEYWORDS = ["GANGLION", "NODE", "LN", "LYMPH", "AXIL", "GTV-N", "GTVN"]

# ── Séparation sémantique des ganglions ───────────────────────────────────────
# True  : Tumeur = 1, Ganglion = 2 (Nécessite d'ajouter "lymph_node": 2 dans le dataset.json nnU-Net)
# False : Tumeur et Ganglion sont tous les deux marqués avec la valeur 1 (Lésion globale)
SEPARATE_LYMPH_NODE_CLASS = False

# ==============================================================================
# SECTION 2 — MODULE SÉMANTIQUE : AUTO-LABÉLISATION
# ==============================================================================

def determine_mask_class(dicom_path: str) -> int:
    """
    Analyse les métadonnées DICOM pour déterminer la classe sémantique du masque.

    Stratégie :
        - Pour un SEG  : on lit la SegmentSequence. On ignore les segments de type 
          "Background" pour trouver la vraie description de la lésion.
        - Pour un RTSTRUCT : on lit StructureSetROISequence[0].ROIName
        - On cherche les mots-clés ganglion dans la description (insensible à la casse)

    Retourne :
        1 → Tumeur / Masse principale (classe par défaut si rien ne correspond)
        2 → Ganglion lymphatique / Lymph Node (Si SEPARATE_LYMPH_NODE_CLASS = True)

    Note :
        Cette heuristique est volontairement simple. Pour des datasets avec des
        conventions de nommage non standardisées, on peut étendre NODE_KEYWORDS
        ou implémenter une logique de matching plus sophistiquée.
    """
    try:
        # Lecture rapide des métadonnées uniquement (pas besoin des pixels ici)
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        modality = getattr(ds, "Modality", "")
        desc = ""

        if modality == "SEG":
            # --- CORRECTION DU BUG "BACKGROUND" ---
            # Un SEG peut contenir plusieurs "Segments" internes. Parfois, le logiciel 
            # d'annotation crée un Segment #1 nommé "Background" et un Segment #2 
            # nommé "Tumor". Il faut chercher le premier segment qui N'EST PAS un background.
            if hasattr(ds, "SegmentSequence"):
                for seg in ds.SegmentSequence:
                    seg_label = str(getattr(seg, "SegmentLabel", "")).upper()
                    seg_desc  = str(getattr(seg, "SegmentDescription", "")).upper()
                    
                    if "BACKGROUND" not in seg_label and "BACKGROUND" not in seg_desc:
                        # On concatène pour maximiser les chances de trouver le mot-clé
                        desc = seg_desc + " " + seg_label
                        break # On a trouvé la vraie lésion, on arrête de chercher !

        elif modality == "RTSTRUCT":
            # Pour un RTSTRUCT, le nom de la ROI est dans StructureSetROISequence
            if hasattr(ds, "StructureSetROISequence") and len(ds.StructureSetROISequence) > 0:
                desc = getattr(ds.StructureSetROISequence[0], "ROIName", "")

        # Comparaison insensible à la casse avec les mots-clés ganglion
        desc_upper = str(desc).upper()
        if any(kw in desc_upper for kw in NODE_KEYWORDS):
            # C'est un ganglion. Regardons ce que dictent les paramètres globaux :
            if SEPARATE_LYMPH_NODE_CLASS:
                return 2  # Ganglion lymphatique séparé
            else:
                return 1  # Lésion globale (Ganglion et tumeur partagent la classe 1)

        return 1  # Fallback : tumeur principale

    except Exception:
        # En cas d'erreur de lecture, on fait confiance au fallback tumeur
        return 1

# ==============================================================================
# SECTION 3 — MODULE DE CONVERSION TEMPORAIRE (PIXELS)
# ==============================================================================

def convert_to_temp_nifti(
    dcm_path: str,
    ref_nifti_path: str | None,
    temp_dir: str
) -> str | None:
    """
    Convertit un fichier DICOM masque (SEG ou RTSTRUCT) en NIfTI temporaire
    pour permettre le chargement numpy et le calcul du Dice.

    Pourquoi un NIfTI temporaire ?
        L'analyseur multi-masques a besoin de lire les matrices numpy de TOUS les
        masques avant de prendre les décisions (doublon, overlap, fusion).
        Créer des NIfTI temporaires uniformise le format d'entrée quelle que soit
        la modalité source (SEG ou RTSTRUCT).

    Paramètres :
        dcm_path        : chemin du fichier DICOM masque à convertir
        ref_nifti_path  : chemin d'une image NIfTI de référence (nécessaire
                          pour rastériser un RTSTRUCT via Plastimatch).
                          Peut être None si le masque est un SEG pur.
        temp_dir        : répertoire temporaire où écrire le NIfTI intermédiaire.

    Retourne :
        Chemin du NIfTI temporaire créé, ou None en cas d'échec.

    Binarisation :
        Le NIfTI produit est toujours binarisé (0/1) pour homogénéiser la
        comparaison Dice quel que soit le système de contouring d'origine.
        Les segments explicitement étiquetés "Background" sont ignorés pour 
        éviter les faux-positifs "full-body".
    """
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
        modality = getattr(ds, "Modality", "")

        # Chemin de sortie dans le répertoire temporaire
        # On utilise le basename du DCM pour éviter les collisions de noms
        out_path = os.path.join(temp_dir, os.path.basename(dcm_path) + "_temp.nii.gz")

        # ── Cas 1 : DICOM SEG ─────────────────────────────────────────────────
        if modality == "SEG":
            # pydicom-seg gère le bit-packing et les SEG multi-frames compressés
            reader = pydicom_seg.MultiClassReader()
            result = reader.read(pydicom.dcmread(dcm_path))

            arr = sitk.GetArrayFromImage(result.image)
            
            # --- CORRECTION DU BUG "BACKGROUND" ---
            # On crée une grille vide de la même taille que l'image
            clean_arr = np.zeros_like(arr, dtype=np.uint8)
            
            # On inspecte chaque segment encodé par le logiciel
            # (Certains logiciels comme 3D Slicer encodent un segment entier pour le "fond")
            for seg_val, seg_info in result.segment_infos.items():
                label = str(getattr(seg_info, "SegmentLabel", "")).upper()
                desc  = str(getattr(seg_info, "SegmentDescription", "")).upper()
                
                # Si le segment N'EST PAS du "Background", on le garde et on l'active (1)
                # Cela protège contre les masques qui couvrent 100% de l'image.
                if "BACKGROUND" not in label and "BACKGROUND" not in desc:
                    clean_arr[arr == seg_val] = 1

            clean_img = sitk.GetImageFromArray(clean_arr)
            # On préserve les métadonnées spatiales (origine, spacing, directions)
            # pour que le masque s'aligne correctement sur l'image de référence
            clean_img.CopyInformation(result.image)

            sitk.WriteImage(clean_img, out_path)
            return out_path

        # ── Cas 2 : RTSTRUCT ──────────────────────────────────────────────────
        elif modality == "RTSTRUCT":
            # Un RTSTRUCT encode des contours 2D vectoriels (polygones plan par plan).
            # Pour obtenir un masque 3D voxelisé, on utilise Plastimatch qui
            # rastérise ces polygones sur la grille de l'image de référence.

            if not ref_nifti_path:
                # Sans image de référence, la rastérisation est impossible :
                # Plastimatch ne connaît pas la grille spatiale sur laquelle
                # projeter les contours.
                raise ValueError(
                    "RTSTRUCT orphelin : aucune image de référence disponible "
                    "pour la rastérisation Plastimatch."
                )

            # Appel Plastimatch :
            #   --input       : le fichier RTSTRUCT DICOM
            #   --fixed       : l'image de référence qui définit la grille 3D
            #   --output-img  : le NIfTI de sortie (masque voxelisé)
            #   --output-type : uint8 pour un masque binaire léger
            commande = [
                PLASTIMATCH_EXE, "convert",
                "--input",       dcm_path,
                "--fixed",       ref_nifti_path,
                "--output-img",  out_path,
                "--output-type", "uint8",
            ]
            subprocess.run(
                commande,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            return out_path

        else:
            # Modalité inconnue (RTDOSE, etc.) : on refuse silencieusement
            return None

    except Exception as e:
        print(f"   [ERREUR CONV TEMP] {os.path.basename(dcm_path)} : {e}")
        return None


# ==============================================================================
# SECTION 4 — MODULE DE CHARGEMENT DES PIXELS (STATISTIQUES INDIVIDUELLES)
# ==============================================================================

def load_mask_pixel_data(
    dcm_path: str,
    ref_nifti_path: str | None,
    temp_dir: str
) -> dict:
    """
    Charge un masque DICOM (SEG ou RTSTRUCT) et calcule ses statistiques.

    C'est la fusion de :
        - load_seg_pixel_data() de V3 (stats détaillées, gestion d'erreur explicite)
        - La logique de chargement inline de V4 (support RTSTRUCT via tempdir)

    Le dictionnaire retourné contient TOUJOURS le champ "path" et soit :
        - Un champ "error" si le chargement a échoué (le masque sera rejeté)
        - Tous les champs de statistiques si le chargement a réussi

    Champs retournés en cas de succès :
        path              : chemin du fichier DICOM source
        temp_nii          : chemin du NIfTI temporaire créé
        sitk_img          : objet SimpleITK.Image (pour copier les métadonnées spatiales)
        binary_mask       : np.ndarray booléen (True = voxel segmenté)
        shape             : tuple (z, y, x) de la matrice
        unique_values     : liste des valeurs uniques dans le masque brut
        segmented_voxels  : nombre de voxels à True (int)
        total_voxels      : nombre total de voxels (int)
        ratio             : segmented_voxels / total_voxels (float)
    """
    # Initialisation avec le chemin source dans tous les cas
    result_base = {"path": dcm_path}

    # Étape 1 : conversion en NIfTI temporaire (SEG ou RTSTRUCT)
    tmp_nii = convert_to_temp_nifti(dcm_path, ref_nifti_path, temp_dir)

    if tmp_nii is None:
        # La conversion a échoué : on retourne un dict d'erreur explicite
        return {**result_base, "error": "Échec de la conversion en NIfTI temporaire"}

    # Étape 2 : chargement de l'image SimpleITK depuis le NIfTI temporaire
    try:
        sitk_img = sitk.ReadImage(tmp_nii)
        arr      = sitk.GetArrayFromImage(sitk_img)  # shape (z, y, x)

        binary_mask      = arr > 0
        total_voxels     = int(binary_mask.size)
        segmented_voxels = int(binary_mask.sum())
        ratio = segmented_voxels / total_voxels if total_voxels > 0 else 0.0

        # --- NOUVEAU : Extraction de la date de création absolue du masque ---
        # On lit l'en-tête DICOM du masque pour trouver quand il a été dessiné/sauvegardé.
        # InstanceCreationTime est la balise la plus fiable pour un masque généré a posteriori.
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
            date_str = getattr(ds, 'InstanceCreationDate', getattr(ds, 'ContentDate', getattr(ds, 'SeriesDate', '19000101')))
            time_str = getattr(ds, 'InstanceCreationTime', getattr(ds, 'ContentTime', getattr(ds, 'SeriesTime', '000000')))
            
            date_str = str(date_str).strip() if date_str else '19000101'
            time_str = str(time_str).split('.')[0].strip() if time_str else '000000'
            if len(time_str) < 6: time_str = time_str.ljust(6, '0')
            
            mask_dt = datetime.strptime(f"{date_str}{time_str[:6]}", "%Y%m%d%H%M%S")
        except Exception:
            mask_dt = datetime(1900, 1, 1) # Fallback absolu

        return {
            **result_base,
            "temp_nii"         : tmp_nii,
            "sitk_img"         : sitk_img,
            "binary_mask"      : binary_mask,
            "shape"            : arr.shape,
            "unique_values"    : np.unique(arr).tolist(),
            "segmented_voxels" : segmented_voxels,
            "total_voxels"     : total_voxels,
            "ratio"            : ratio,
            "datetime"         : mask_dt, # <--- NOUVEAU CHAMP AJOUTÉ ICI
        }

    except Exception as e:
        return {**result_base, "error": f"Erreur SimpleITK ReadImage : {e}"}


# ==============================================================================
# SECTION 5 — CALCUL DU COEFFICIENT DE DICE
# ==============================================================================

def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calcule le coefficient de Dice entre deux masques binaires numpy.

    Formule :
        Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)

    Interprétation :
        1.0 → Masques parfaitement superposés (identiques voxel-à-voxel)
        0.0 → Aucun voxel en commun (structures entièrement disjointes)

    Préconditions :
        - Les deux masques doivent avoir la même shape (vérifié par l'appelant).
        - Les masques doivent être de type booléen ou entier avec 0/1.

    Cas dégénéré :
        Si les deux masques sont complètement vides (sum = 0), on retourne 0.0.
        Deux masques vides ne sont pas "identiques" dans un contexte clinique :
        ils signalent tous les deux une absence de segmentation, ce qui est un
        problème à investiguer, pas une concordance.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    volume_a     = mask_a.sum()
    volume_b     = mask_b.sum()

    # Protection division par zéro (deux masques entièrement vides)
    if volume_a + volume_b == 0:
        return 0.0

    return float((2.0 * intersection) / (volume_a + volume_b))


# ==============================================================================
# SECTION 6 — MOTEUR D'ANALYSE MULTI-MASQUES
# ==============================================================================

def analyze_multiple_masks(loaded_data: list[dict], log: list[str]) -> dict:
    """
    Moteur de décision pour un groupe de masques appartenant à la même visite.

    Parcourt trois étapes structurées (réintégrées de V3) :
        ÉTAPE 1 — Analyse individuelle (stats + détection corrompus/full-body)
        ÉTAPE 2 — Comparaisons croisées Dice (doublons, overlap, structures distinctes)
        ÉTAPE 3 — Consolidation et classification finale

    Résultats possibles pour chaque masque :
        "keep"   → Masque sain et seul de son espèce : conversion directe
        "fuse"   → Masque sain parmi plusieurs structures distinctes : sera fusionné
        "reject" → Corrompu, full-body ou doublon à supprimer
        "manual" → Overlap ambigu : copié dans "a_verifier/" pour inspection

    Paramètres :
        loaded_data : liste de dicts produits par load_mask_pixel_data()
                      Chaque dict contient path, binary_mask, shape, etc.
        log         : liste de chaînes partagée avec l'appelant.
                      Toutes les décisions y sont appendées pour le rapport.

    Retourne un dict avec quatre listes :
        keep, fuse, reject, manual  → chacune contient des chemins DICOM (str)

    BUGS CORRIGÉS PAR RAPPORT À V4 :
        - Bug NameError : les boucles utilisaient 'data_a'/'data_b' au lieu de 'a'/'b'
        - Shape check absent dans la boucle overlap → risque de crash numpy
        - Masques en erreur non comptés comme rejetés
        - Logging des stats individuelles absent (réintégré de V3)
        - Message [DISTINCT] absent pour les faibles Dice (réintégré de V3)
    """

    # Dictionnaire de décisions : chemin → décision courante
    # Initialement tout le monde est "keep" ; les étapes suivantes peuvent changer ça.
    decs = {d["path"]: "keep" for d in loaded_data}

    # ──────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 : Analyse individuelle — stats et filtres qualité
    # ──────────────────────────────────────────────────────────────────────────
    log.append("\n  --- ÉTAPE 1 : Analyse individuelle ---")

    for data in loaded_data:
        fname = os.path.basename(data["path"])

        # ── Cas A : Le chargement de ce masque a échoué ───────────────────────
        # (V3 avait cette logique dans load_seg_pixel_data ; V4 l'avait perdue)
        if "error" in data:
            log.append(f"\n  Fichier : {fname}")
            log.append(f"    → [REJET] Erreur de chargement : {data['error']}")
            decs[data["path"]] = "reject"
            continue  # On passe au masque suivant sans tenter d'accéder aux stats

        # ── Logging des statistiques individuelles (réintégré de V3) ─────────
        log.append(f"\n  Fichier : {fname}")
        log.append(f"    Shape           : {data['shape']}")
        log.append(f"    Valeurs uniques : {data['unique_values']}")
        log.append(f"    Voxels segm.    : {data['segmented_voxels']} / {data['total_voxels']}")
        log.append(f"    Ratio           : {data['ratio']:.4f}")

        # ── Cas B : Masque uniforme ou corrompu ───────────────────────────────
        # Un masque avec une seule valeur unique est soit entièrement vide (0)
        # soit entièrement plein (1). Dans les deux cas, il n'est pas exploitable.
        if len(data["unique_values"]) <= 1:
            log.append(
                "    → [REJET] Masque uniforme ou corrompu "
                "(une seule valeur de pixel : 0 ou 1 partout)."
            )
            decs[data["path"]] = "reject"
            continue

        # ── Cas C : Masque full-body / aberrant ───────────────────────────────
        # Si le masque couvre plus de FULL_BODY_THRESHOLD du volume total,
        # il s'agit probablement d'un artefact ou d'un masque corps entier
        # qui n'a rien à faire dans un pipeline de segmentation tumorale.
        if data["ratio"] > FULL_BODY_THRESHOLD:
            log.append(
                f"    → [REJET] Masque suspect : ratio {data['ratio']:.4f} "
                f"dépasse le seuil full-body ({FULL_BODY_THRESHOLD})."
            )
            decs[data["path"]] = "reject"
            continue

        # ── Cas D : Masque valide ─────────────────────────────────────────────
        log.append("    → [VALIDE] Masque exploitable, candidat pour comparaison.")

    # Sous-ensemble des masques non rejetés pour les comparaisons Dice
    active = [d for d in loaded_data if decs[d["path"]] == "keep"]

    # ──────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 : Comparaisons croisées Dice
    # ──────────────────────────────────────────────────────────────────────────
    log.append("\n  --- ÉTAPE 2 : Comparaisons Dice croisées ---")

    if len(active) < 2:
        # Moins de 2 masques valides : rien à comparer
        log.append("  Moins de 2 masques valides → aucune comparaison nécessaire.")
    else:
        # On itère sur toutes les paires possibles parmi les masques actifs
        for a, b in combinations(active, 2):
            name_a = os.path.basename(a["path"])
            name_b = os.path.basename(b["path"])
            log.append(f"\n  Comparaison : {name_a}  VS  {name_b}")

            # ── Pré-condition : shapes identiques ─────────────────────────────
            # Le Dice voxel-à-voxel n'a de sens que si les deux grilles sont
            # identiques. Si elles diffèrent, les masques couvrent forcément
            # des volumes distincts (ou des résolutions différentes).
            # BUG CORRIGÉ V4 : utilisait 'data_a'/'data_b' au lieu de 'a'/'b'
            if a["shape"] != b["shape"]:
                log.append(
                    f"    → [SKIP] Shapes différentes "
                    f"({a['shape']} vs {b['shape']}) : "
                    "grilles incompatibles — structures conservées séparément."
                )
                # Pas de changement de décision : les deux restent "keep"
                continue

            # ── Calcul du Dice ─────────────────────────────────────────────────
            dice = dice_score(a["binary_mask"], b["binary_mask"])
            log.append(f"    Dice : {dice:.4f}")

            # ── Interprétation du score Dice ───────────────────────────────────

            if dice >= DICE_DUPLICATE_THRESHOLD:
                # DOUBLON : les deux masques couvrent quasi-exactement les mêmes voxels.
                # Heuristique clinique : on garde le plus petit (plus conservative,
                # donc généralement plus précis — moins de tissu sain inclus).
                log.append(
                    f"    → [DOUBLON] Dice ≥ {DICE_DUPLICATE_THRESHOLD} : "
                    "masques quasi identiques."
                )
                # --- NOUVEAU : Logique de sélection paramétrable ---
                raison_choix = "" # Initialisation vitale pour éviter le crash
                if KEEP_NEWEST_DUPLICATE_MASK:
                    # Stratégie Clinique (Superviseur) : On suppose que le médecin a 
                    # corrigé/affiné son masque, on garde donc le plus récent.
                    if a["datetime"] >= b["datetime"]:
                        keep_d, reject_d = a, b
                    else:
                        keep_d, reject_d = b, a
                    raison_choix = "le plus récent"
                else:
                    # Stratégie Conservative : On garde le plus petit volume pour 
                    # minimiser le risque d'inclure du tissu sain.
                    if a["segmented_voxels"] <= b["segmented_voxels"]:
                        keep_d, reject_d = a, b
                    else:
                        keep_d, reject_d = b, a
                    raison_choix = "le plus petit volume"

                log.append(
                    f"    → [GARDER]  {os.path.basename(keep_d['path'])} "
                    f"({keep_d['segmented_voxels']} voxels — plus petit)"
                )
                log.append(
                    f"    → [REJETER] {os.path.basename(reject_d['path'])} "
                    f"({reject_d['segmented_voxels']} voxels — plus grand)"
                )
                # On écrase la décision provisoire "keep" par "reject" pour le doublon
                decs[reject_d["path"]] = "reject"

            elif dice >= DICE_OVERLAP_THRESHOLD:
                # OVERLAP PARTIEL : chevauchement significatif mais pas identique.
                # Causes possibles : variabilité inter-annotateur, deux lésions
                # adjacentes, erreur de segmentation.
                # On ne peut pas décider algorithmiquement → inspection manuelle.
                log.append(
                    f"    → [OVERLAP] Dice = {dice:.4f} "
                    f"(entre {DICE_OVERLAP_THRESHOLD} et {DICE_DUPLICATE_THRESHOLD}) : "
                    "chevauchement partiel ambigu."
                )
                log.append(
                    "    → [INSPECTION MANUELLE] Les deux masques sont copiés "
                    "dans 'a_verifier/' pour décision radiologique."
                )
                # BUG CORRIGÉ V4 : la boucle overlap n'avait pas de shape check.
                # On vérifie ici que les deux masques ne sont pas déjà rejetés
                # par une paire précédente avant de les passer en manuel.
                if decs[a["path"]] != "reject":
                    decs[a["path"]] = "manual"
                if decs[b["path"]] != "reject":
                    decs[b["path"]] = "manual"

            else:
                # STRUCTURES DISTINCTES : Dice faible → les masques couvrent
                # des zones très différentes. Il s'agit probablement de deux
                # lésions anatomiquement séparées (ex : tumeur primaire +
                # ganglion axillaire métastatique).
                # On les conserve TOUS LES DEUX pour fusion sémantique.
                # MESSAGE [DISTINCT] RÉINTÉGRÉ DE V3 (absent dans V4)
                log.append(
                    f"    → [DISTINCT] Dice = {dice:.4f} < {DICE_OVERLAP_THRESHOLD} : "
                    "structures probablement différentes (ex : lésions multiples)."
                )
                log.append(
                    "    → [GARDER LES DEUX] Ils seront fusionnés dans un "
                    "volume multi-classes unique (_FUSED.nii.gz)."
                )
                # Pas de changement de décision : les deux restent "keep"
                # La décision de fusion interviendra à l'étape 3

    # ──────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 : Consolidation — promotion "keep" → "fuse" si multiples distincts
    # ──────────────────────────────────────────────────────────────────────────
    log.append("\n  --- ÉTAPE 3 : Décisions finales ---")

    # On relève les masques encore en "keep" après les étapes précédentes
    final_keeps  = [d for d in loaded_data if decs[d["path"]] == "keep"]
    final_manual = [d for d in loaded_data if decs[d["path"]] == "manual"]
    final_reject = [d for d in loaded_data if decs[d["path"]] == "reject"]

    # Si plusieurs masques sont toujours en "keep" ET qu'aucun n'a été mis en
    # "manual", c'est qu'ils sont tous distincts (Dice < OVERLAP_THRESHOLD).
    # Dans ce cas, on les groupe pour une FUSION sémantique multi-classes, si possible !
    has_manual = len(final_manual) > 0

    # --- CORRECTION DE LA FUSION : VÉRIFICATION DES SHAPES ---
    # On ne peut fusionner que si TOUS les masques cibles ont EXACTEMENT la même shape.
    shapes_uniques = set(d["shape"] for d in final_keeps)
    can_fuse = (len(shapes_uniques) == 1)

    if not has_manual and len(final_keeps) > 1 and can_fuse:
        # Promotion "keep" → "fuse" pour déclencher la fusion sémantique
        log.append(
            f"    → [FUSION PROGRAMMÉE] {len(final_keeps)} structures distinctes "
            "seront réunies dans un seul NIfTI multi-classes."
        )
        for d in final_keeps:
            decs[d["path"]] = "fuse"
        fuse_paths = [d["path"] for d in final_keeps]
        keep_paths = []
    else:
        # Cas simples :
        # - 1 seul masque valide
        # - Des masques en "manuel" bloquent la fusion
        # - NOUVEAU : Des masques valides mais avec des grilles (shapes) incompatibles
        if not has_manual and len(final_keeps) > 1 and not can_fuse:
            log.append(
                "    → [FUSION ANNULÉE] Les masques n'ont pas la même dimension (shape). "
                "Ils seront convertis séparément."
            )
            
        fuse_paths = []
        keep_paths = [d["path"] for d in final_keeps]

    # ── Log récapitulatif de l'étape 3 ────────────────────────────────────────
    reject_paths = [d["path"] for d in final_reject]
    manual_paths = [d["path"] for d in final_manual]

    for path in keep_paths:
        log.append(f"  [KEEP]    {os.path.basename(path)}")
    for path in fuse_paths:
        log.append(f"  [FUSE]    {os.path.basename(path)}")
    for path in reject_paths:
        log.append(f"  [REJECT]  {os.path.basename(path)}")
    for path in manual_paths:
        log.append(f"  [MANUAL]  {os.path.basename(path)}")

    log.append(
        f"\n  Résumé : {len(keep_paths)} keep, {len(fuse_paths)} à fusionner, "
        f"{len(reject_paths)} rejeté(s), {len(manual_paths)} en inspection manuelle."
    )

    return {
        "keep"   : keep_paths,
        "fuse"   : fuse_paths,
        "reject" : reject_paths,
        "manual" : manual_paths,
    }


# ==============================================================================
# SECTION 7 — ORCHESTRATEUR PRINCIPAL PAR PASSE
# ==============================================================================

def process_patient_masks(
    project_root    : str,
    mask_prefix     : str,
    global_log_lines: list[str],
    global_manual_cases: list[str] = None
) -> int:
    """
    Parcourt une racine de projet, traite tous les dossiers de masques correspondant
    au préfixe donné, et génère les NIfTI masques finaux.

    Pour chaque visite d'un patient :
        1. Collecte tous les fichiers SEG/RTSTRUCT dans les sous-dossiers de séries.
        2. Charge les pixels dans un espace temporaire commun.
        3. Si 1 seul masque  → conversion directe.
           Si N masques      → analyse automatique (ÉTAPE 1/2/3) puis action.
        4. Actions possibles :
             KEEP   → sauvegarde individuelle avec classe sémantique (1 ou 2)
             FUSE   → fusion multi-classes dans _FUSED.nii.gz
             MANUAL → copie dans "a_verifier/" (aucun NIfTI final généré)
             REJECT → aucune action (le DCM reste en place dans dicom_mask_*)
        5. Toutes les décisions sont accumulées dans global_log_lines.

    Paramètres :
        project_root     : chemin absolu ou relatif de la racine du dataset
                           (ex : "./Base_IRM" ou "./Base_PETCT")
        mask_prefix      : préfixe des dossiers de masques à traiter
                           (ex : "dicom_mask_rm", "dicom_mask_pet")
        global_log_lines : liste partagée pour le rapport global. Chaque passe
                           y ajoute ses lignes ; le fichier est écrit une seule
                           fois à la fin du script par le point d'entrée.

    Retourne :
        Nombre de NIfTI masques générés (int) — pour les statistiques globales.
    """

    # ── Vérification de l'existence de la racine ──────────────────────────────
    if not os.path.exists(project_root):
        msg = f"\n[SKIP] Dossier inexistant, passe ignorée : {project_root}"
        print(msg)
        global_log_lines.append(msg)
        return 0

    # ── En-tête de passe ──────────────────────────────────────────────────────
    header = (
        f"\n{'='*70}\n"
        f"  EXTRACTION MASQUES (V5) | Racine : {project_root} | Préfixe : {mask_prefix}\n"
        f"{'='*70}"
    )
    print(header)
    global_log_lines.append(header)

    # Liste des sous-dossiers patients (triée pour reproductibilité)
    patients = sorted([
        p for p in os.listdir(project_root)
        if os.path.isdir(os.path.join(project_root, p))
    ])

    # Compteurs de bilan pour cette passe
    masques_generes = 0
    masques_manuels = 0
    masques_rejetes = 0

    # ── Espace de travail temporaire partagé pour toute la passe ──────────────
    # Un seul tempdir pour toute la passe évite la multiplication des dossiers
    # et garantit leur nettoyage automatique même en cas d'exception.
    with tempfile.TemporaryDirectory() as temp_workspace:

        for patient_id in patients:
            patient_dir = os.path.join(project_root, patient_id)

            # ── Détection dynamique des dossiers de masques de ce patient ─────
            # On cherche TOUS les dossiers qui commencent par mask_prefix :
            #   "dicom_mask_rm"               → baseline
            #   "dicom_mask_rm_20230514_1430" → suivi longitudinal
            mask_dirs = sorted([
                d for d in os.listdir(patient_dir)
                if os.path.isdir(os.path.join(patient_dir, d))
                and d.startswith(mask_prefix)
            ])

            if not mask_dirs:
                continue  # Ce patient n'a pas de masques pour ce préfixe

            print(f"\n[PATIENT] {patient_id}")
            global_log_lines.append(f"\n[PATIENT] {patient_id}")

            for source_mask_folder in mask_dirs:
                source_mask_dir = os.path.join(patient_dir, source_mask_folder)

                # ── Routage vers le bon dossier NIfTI cible ───────────────────
                # "dicom_mask_rm"               → "mask"
                # "dicom_mask_rm_20230514_1430" → "mask_20230514_1430"
                suffixe_temporel = source_mask_folder.replace(mask_prefix, "")
                dest_mask_name   = f"mask{suffixe_temporel}"
                dest_mask_dir    = os.path.join(patient_dir, dest_mask_name)

                # ── Recherche de l'image NIfTI de référence ───────────────────
                # Indispensable pour rastériser les RTSTRUCT via Plastimatch.
                # On cherche dans le dossier "imgs/" ou "imgs_<suffixe>/"
                # correspondant à la même visite que le masque.
                ref_img_folder = f"imgs{suffixe_temporel}"
                ref_img_dir    = os.path.join(patient_dir, ref_img_folder)
                ref_nifti_path = None

                if os.path.exists(ref_img_dir):
                    niftis = glob.glob(os.path.join(ref_img_dir, "*.nii.gz"))
                    if niftis:
                        # On prend le premier NIfTI trouvé comme repère spatial.
                        # Pour les IRM multi-séquences, tous partagent la même grille.
                        ref_nifti_path = niftis[0]

                visit_log = (
                    f"\n  [VISITE] {source_mask_folder} → {dest_mask_name}/ "
                    f"| Ref NIfTI : {os.path.basename(ref_nifti_path) if ref_nifti_path else 'AUCUNE'}"
                )
                print(visit_log)
                global_log_lines.append(visit_log)

                # ── Collecte des fichiers DICOM masques de cette visite ────────
                # Structure attendue par l'ingesteur :
                #   source_mask_dir/
                #       DUKE_001_A1B2C/    ← sous-dossier de série (uid suffix)
                #           fichier.dcm   ← le SEG ou RTSTRUCT lui-même
                #
                # On indexe uid_suffix_filename → chemin_complet pour les clés uniques.
                seg_candidates = {}

                for sub_dir in sorted(os.listdir(source_mask_dir)):
                    series_path = os.path.join(source_mask_dir, sub_dir)
                    if not os.path.isdir(series_path):
                        continue

                    # Extraction du suffixe UID de la série (dernier segment "_")
                    uid_suffix = sub_dir.split("_")[-1] if "_" in sub_dir else sub_dir

                    for file_name in os.listdir(series_path):
                        dcm_path = os.path.join(series_path, file_name)

                        # Pré-vérification légère : est-ce un SEG ou RTSTRUCT ?
                        # On lit uniquement les métadonnées (stop_before_pixels)
                        # pour éviter de charger inutilement les données pixel.
                        try:
                            ds_check = pydicom.dcmread(
                                dcm_path, stop_before_pixels=True, force=True
                            )
                            modality = getattr(ds_check, "Modality", "")
                            if modality in ["SEG", "RTSTRUCT"]:
                                seg_candidates[f"{uid_suffix}_{file_name}"] = dcm_path
                        except Exception:
                            continue  # Fichier DICOM illisible ou non-DICOM, on ignore

                nb_candidats = len(seg_candidates)
                global_log_lines.append(
                    f"  → {nb_candidats} masque(s) DICOM (SEG/RTSTRUCT) détecté(s) "
                    f"dans '{source_mask_folder}'"
                )

                if nb_candidats == 0:
                    global_log_lines.append(
                        "  → Aucun masque DICOM valide trouvé dans ce dossier, passe ignorée."
                    )
                    continue

                # ── Alerte RTSTRUCT orphelin ───────────────────────────────────
                # Un RTSTRUCT sans image de référence ne peut pas être rastérisé.
                # On avertit et on saute cette visite si c'est le seul type présent.
                all_dcm_paths = list(seg_candidates.values())
                has_rtstruct  = any(
                    getattr(
                        pydicom.dcmread(p, stop_before_pixels=True, force=True),
                        "Modality", ""
                    ) == "RTSTRUCT"
                    for p in all_dcm_paths
                )

                if has_rtstruct and not ref_nifti_path:
                    err_msg = (
                        "  [ALERTE] Masque(s) RTSTRUCT orphelin(s) ignoré(s) : "
                        "aucune image de référence NIfTI disponible pour rastérisation."
                    )
                    print(err_msg)
                    global_log_lines.append(err_msg)
                    # On continue quand même si des SEG existent dans le lot
                    # (le convert_to_temp_nifti retournera None pour les RTSTRUCT seuls)

                # =============================================================
                # CHARGEMENT EN MÉMOIRE DE TOUS LES MASQUES CANDIDATS
                # On passe par le tempdir partagé pour les conversions NIfTI
                # =============================================================
                loaded_data = []

                for dcm_key, dcm_path in seg_candidates.items():
                    data = load_mask_pixel_data(dcm_path, ref_nifti_path, temp_workspace)

                    if "error" in data:
                        # Masque illisible : on le logue ET on incrémente le compteur
                        # de rejetés (comportement V3, absent dans V4)
                        err_msg = (
                            f"  [ERREUR CHARGEMENT] {os.path.basename(dcm_path)} : "
                            f"{data['error']}"
                        )
                        print(err_msg)
                        global_log_lines.append(err_msg)
                        masques_rejetes += 1
                        # On ajoute quand même à loaded_data pour que
                        # l'analyseur puisse le tracer explicitement
                        loaded_data.append(data)
                    else:
                        loaded_data.append(data)

                # Si tous les masques ont échoué au chargement, on saute la visite
                valid_count = sum(1 for d in loaded_data if "error" not in d)
                if valid_count == 0:
                    global_log_lines.append(
                        "  → Tous les masques ont échoué au chargement. "
                        "Cette visite est ignorée."
                    )
                    continue

                # =============================================================
                # ANALYSE ET DÉCISION (simple ou multi-masques)
                # =============================================================
                if len(loaded_data) == 1 and "error" not in loaded_data[0]:
                    # Un seul masque valide : conversion directe, pas d'analyse
                    global_log_lines.append(
                        "  → Un seul masque valide : conversion directe (pas d'analyse Dice)."
                    )
                    paths_to_convert = {
                        "keep"   : [loaded_data[0]["path"]],
                        "fuse"   : [],
                        "manual" : [],
                        "reject" : [],
                    }
                else:
                    # Plusieurs masques (ou 1 masque + des erreurs de chargement) :
                    # on lance l'analyseur automatique complet
                    multi_header = (
                        f"\n  *** {len(loaded_data)} masque(s) détecté(s) → "
                        "Analyse automatique multi-masques ***"
                    )
                    print(multi_header)
                    global_log_lines.append(multi_header)

                    # L'analyseur utilise la liste partagée 'global_log_lines'
                    # pour accumuler ses propres lignes de log
                    paths_to_convert = analyze_multiple_masks(loaded_data, global_log_lines)

                    # Mise à jour des compteurs globaux depuis les résultats de l'analyse
                    masques_rejetes += len(paths_to_convert["reject"])
                    masques_manuels += len(paths_to_convert["manual"])

                # =============================================================
                # ACTION 1 : FUSION SÉMANTIQUE MULTI-CLASSES
                # Cas : plusieurs structures distinctes (Dice bas) détectées
                # → Un seul NIfTI avec classe 1 = tumeur, classe 2 = ganglion
                # =============================================================
                if paths_to_convert["fuse"]:
                    os.makedirs(dest_mask_dir, exist_ok=True)

                    # On récupère les données chargées du premier masque à fusionner
                    # pour initialiser la matrice résultat à la bonne shape
                    fuse_data_list = [
                        d for d in loaded_data
                        if d["path"] in paths_to_convert["fuse"] and "error" not in d
                    ]

                    if fuse_data_list:
                        ref_data   = fuse_data_list[0]
                        # Matrice de sortie : initialement tout à 0 (fond)
                        fused_arr  = np.zeros(ref_data["shape"], dtype=np.uint8)

                        global_log_lines.append(f"\n  Fusion de {len(fuse_data_list)} masques :")

                        for fuse_data in fuse_data_list:
                            # Détermination de la classe sémantique (1 ou 2)
                            class_label = determine_mask_class(fuse_data["path"])
                            fname_fuse  = os.path.basename(fuse_data["path"])

                            msg_peinture = (
                                f"      + Peinture de '{fname_fuse}' → "
                                f"Classe {class_label} "
                                f"({'Tumeur' if class_label == 1 else 'Ganglion'})"
                            )
                            print(msg_peinture)
                            global_log_lines.append(msg_peinture)

                            # Les voxels True de ce masque reçoivent la valeur de classe
                            # Note : si deux masques se superposent malgré un Dice bas
                            # (ex : voxels frontière), le dernier peint gagne.
                            fused_arr[fuse_data["binary_mask"]] = class_label

                        # Construction de l'image SimpleITK finale avec les bonnes
                        # métadonnées spatiales (copiées du premier masque de référence)
                        final_img = sitk.GetImageFromArray(fused_arr)
                        final_img.CopyInformation(ref_data["sitk_img"])

                        dest_name = f"{patient_id}_mask_FUSED.nii.gz"
                        dest_path = os.path.join(dest_mask_dir, dest_name)
                        sitk.WriteImage(final_img, dest_path)

                        ok_msg = f"  → [CONVERTI & FUSIONNÉ] {dest_name}"
                        print(ok_msg)
                        global_log_lines.append(ok_msg)
                        masques_generes += 1

                # =============================================================
                # ACTION 2 : SAUVEGARDE INDIVIDUELLE (KEEP)
                # Cas : un seul masque valide, ou une conversion simple
                # → NIfTI avec voxels typés par classe sémantique
                # =============================================================
                for keep_path in paths_to_convert["keep"]:
                    os.makedirs(dest_mask_dir, exist_ok=True)

                    # On récupère les données déjà chargées pour ce masque
                    data = next(
                        (d for d in loaded_data if d["path"] == keep_path and "error" not in d),
                        None
                    )
                    if data is None:
                        global_log_lines.append(
                            f"  → [ECHEC] Données manquantes pour {os.path.basename(keep_path)}"
                        )
                        continue

                    # Suffixe UID de la série (pour nommage traçable)
                    uid_suffix       = os.path.basename(os.path.dirname(keep_path)).split("_")[-1]
                    original_dcm_name = os.path.splitext(os.path.basename(keep_path))[0]

                    # Application de la classe sémantique sur le masque binaire
                    class_label = determine_mask_class(keep_path)
                    arr_typed   = data["binary_mask"].astype(np.uint8) * class_label

                    final_img = sitk.GetImageFromArray(arr_typed)
                    final_img.CopyInformation(data["sitk_img"])

                    dest_name = f"{patient_id}_mask_{uid_suffix}_{original_dcm_name}.nii.gz"
                    dest_path = os.path.join(dest_mask_dir, dest_name)
                    sitk.WriteImage(final_img, dest_path)

                    ok_msg = (
                        f"  → [CONVERTI] {dest_name} "
                        f"(Classe {class_label} : "
                        f"{'Tumeur' if class_label == 1 else 'Ganglion'})"
                    )
                    print(ok_msg)
                    global_log_lines.append(ok_msg)
                    masques_generes += 1

                # =============================================================
                # ACTION 3 : MASQUES EN INSPECTION MANUELLE → "a_verifier/"
                # Ces fichiers DICOM sont copiés hors du flux normal pour ne
                # pas polluer le dataset propre, mais sont conservés pour qu'un
                # radiologue puisse trancher.
                # =============================================================
                if paths_to_convert["manual"]:
                    verif_dir = os.path.join(dest_mask_dir, "a_verifier")
                    os.makedirs(verif_dir, exist_ok=True)

                    for man_path in paths_to_convert["manual"]:
                        dest_verif = os.path.join(verif_dir, os.path.basename(man_path))
                        shutil.copy2(man_path, dest_verif)
                        msg_verif = (
                            f"  → [A VÉRIFIER] Copié dans 'a_verifier/' : "
                            f"{os.path.basename(man_path)}"
                        )
                        print(msg_verif)
                        global_log_lines.append(msg_verif)

                        # --- NOUVEAU : ENREGISTREMENT POUR LE RÉSUMÉ FINAL ---
                        if global_manual_cases is not None:
                            global_manual_cases.append(f" - Patient {patient_id} ({source_mask_folder}) : {os.path.basename(man_path)}")

                # =============================================================
                # ACTION 4 : LOG DES MASQUES REJETÉS
                # On les laisse en place dans dicom_mask_* (non-destructif),
                # on les ignore simplement pour la conversion NIfTI.
                # =============================================================
                for rej_path in paths_to_convert["reject"]:
                    rej_msg = (
                        f"  → [REJETÉ] Non converti (corrompu/full-body/doublon) : "
                        f"{os.path.basename(rej_path)}"
                    )
                    global_log_lines.append(rej_msg)

    # ── Bilan de fin de passe ─────────────────────────────────────────────────
    bilan = (
        f"\n  BILAN PASSE [{mask_prefix}] :\n"
        f"    NIfTI générés/fusionnés : {masques_generes}\n"
        f"    En insp. manuelle       : {masques_manuels}\n"
        f"    Rejetés (auto + erreurs): {masques_rejetes}"
    )
    print(bilan)
    global_log_lines.append(bilan)

    return masques_generes


# ==============================================================================
# SECTION 8 — POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    # ── Parsing des arguments en ligne de commande ────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Extracteur de masques DICOM (SEG & RTSTRUCT) → NIfTI (V5).\n"
            "Analyse automatique multi-masques, fusion sémantique multi-classes,\n"
            "support Plastimatch pour les RTSTRUCT, logging exhaustif."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mri_root",
        default="./Base_IRM",
        help=(
            "Racine des données IRM. Doit contenir les sous-dossiers patients "
            "avec les dossiers 'dicom_mask_rm*' et 'imgs*/'. "
            "(Défaut : ./Base_IRM)"
        )
    )
    parser.add_argument(
        "--petct_root",
        default="./Base_PETCT",
        help=(
            "Racine des données PET/CT. Doit contenir les sous-dossiers patients "
            "avec les dossiers 'dicom_mask_pet*' et 'imgs*/'. "
            "(Défaut : ./Base_PETCT)"
        )
    )
    parser.add_argument(
        "--log_root",
        default=".",
        help=(
            "Dossier où écrire le rapport d'analyse (rapport_analyse_masques_v5.txt). "
            "Mettre le même dossier que rapport_ingestion_v6.txt pour centraliser. "
            "(Défaut : répertoire courant)"
        )
    )
    args = parser.parse_args()

    # ── Initialisation du rapport global ─────────────────────────────────────
    # Ce rapport est PARTAGÉ entre toutes les passes (IRM, PET, orphelins).
    # Chaque passe y ajoute ses lignes ; on écrit le fichier une seule fois
    # à la fin pour éviter les I/O répétés.
    global_log_lines = [
        "=" * 70,
        "  RAPPORT D'ANALYSE MASQUES DICOM (SEG & RTSTRUCT) — EXTRACTEUR V5",
        f"  Généré le   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Racine IRM  : {os.path.abspath(args.mri_root)}",
        f"  Racine PET  : {os.path.abspath(args.petct_root)}",
        "=" * 70,
        "",
        "Légende des décisions :",
        "  [VALIDE]           → Masque chargé avec succès, candidat à la comparaison",
        "  [REJET]            → Masque uniforme, full-body, doublon ou illisible",
        "  [DOUBLON]          → Dice ≥ 0.95 avec un autre masque (le plus grand rejeté)",
        "  [DISTINCT]         → Dice < 0.20 (structures différentes, fusion prévue)",
        "  [OVERLAP]          → Dice entre 0.20 et 0.95 (ambigu → inspection manuelle)",
        "  [FUSION PROGRAMMÉE]→ Masques distincts regroupés dans _FUSED.nii.gz",
        "  [CONVERTI]         → NIfTI masque individuel écrit avec succès",
        "  [CONVERTI & FUSIONNÉ] → NIfTI masque multi-classes écrit avec succès",
        "  [A VÉRIFIER]       → Copié dans 'a_verifier/' pour décision radiologique",
        "",
        "Les fichiers 'A VÉRIFIER' se trouvent dans le sous-dossier 'a_verifier/'",
        "adjacent au dossier 'mask/' correspondant.",
        "",
    ]

    global_manual_cases = []

    # ── PASSE 1 : Masques IRM baseline ───────────────────────────────────────
    process_patient_masks(
        project_root     = args.mri_root,
        mask_prefix      = "dicom_mask_rm",
        global_log_lines = global_log_lines,
        global_manual_cases = global_manual_cases
    )

    # ── PASSE 2 : Masques PET/CT baseline ────────────────────────────────────
    process_patient_masks(
        project_root     = args.petct_root,
        mask_prefix      = "dicom_mask_pet",
        global_log_lines = global_log_lines,
        global_manual_cases = global_manual_cases
    )

    # ── PASSE 3 : Masques orphelins IRM ──────────────────────────────────────
    # Masques générés par l'ingesteur quand un SEG/RTSTRUCT ne pouvait pas être
    # lié à une série image précise. Traités séparément pour ne pas contaminer
    # les données propres.
    process_patient_masks(
        project_root     = args.mri_root,
        mask_prefix      = "dicom_mask_orphelins",
        global_log_lines = global_log_lines,
        global_manual_cases = global_manual_cases
    )

    # ── PASSE 4 : Masques orphelins PET/CT ───────────────────────────────────
    process_patient_masks(
        project_root     = args.petct_root,
        mask_prefix      = "dicom_mask_orphelins",
        global_log_lines = global_log_lines,
        global_manual_cases = global_manual_cases
    )

    # --- NOUVEAU : CRÉATION DU TABLEAU DE BORD FINAL ---
    global_log_lines.append("\n" + "=" * 70)
    global_log_lines.append("  TABLEAU DE BORD DES ACTIONS REQUISES (INSPECTION MANUELLE)")
    global_log_lines.append("=" * 70)
    if not global_manual_cases:
        global_log_lines.append("  -> AUCUN MASQUE NE REQUIERT D'INSPECTION MANUELLE. TOUT EST PROPRE.")
    else:
        global_log_lines.append(f"  -> {len(global_manual_cases)} MASQUE(S) NÉCESSITENT VOTRE ATTENTION :\n")
        global_log_lines.extend(global_manual_cases)
    global_log_lines.append("=" * 70 + "\n")

    # ── Écriture du rapport global ────────────────────────────────────────────
    # Tout est centralisé dans un seul fichier texte aux côtés du rapport
    # d'ingestion pour une consultation facile après le pipeline complet.
    os.makedirs(args.log_root, exist_ok=True)
    rapport_path = os.path.join(args.log_root, "rapport_analyse_masques_v5.txt")

    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("\n".join(global_log_lines))

    print(f"\n → Rapport complet écrit ici : {rapport_path}")
    print("\n=== EXTRACTION V5 TERMINÉE AVEC SUCCÈS ===")

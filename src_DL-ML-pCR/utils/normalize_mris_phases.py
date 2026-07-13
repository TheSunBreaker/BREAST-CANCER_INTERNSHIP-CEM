# fichier: normalize_mri.py
import SimpleITK as sitk
import numpy as np
import os

def normalize_dce_patient(chemins_entrees: list, chemins_sorties: list):
    """
    Applique la normalisation Z-Score sur une séquence DCE-MRI (plusieurs phases).
    
    Conformément aux recommandations du challenge MAMA-MIA, cette fonction
    calcule la moyenne et l'écart-type globaux sur l'ensemble des phases du patient,
    puis applique cette transformation à chaque phase individuellement.
    
    :param chemins_entrees: Liste des chemins vers les images d'origine (ex: phase 0, 1, 2...)
    :param chemins_sorties: Liste des chemins où sauvegarder les images formatées pour nnU-Net.
    """
    images_sitk = []
    tableaux_numpy = []
    
    # --- ÉTAPE 1 : LECTURE DE TOUTES LES PHASES ---
    for chemin in chemins_entrees:
        # Chargement en float32 (essentiel pour ne pas perdre de précision avec les divisions)
        img = sitk.ReadImage(chemin, sitk.sitkFloat32)
        images_sitk.append(img)
        
        # Extraction de la matrice de pixels
        arr = sitk.GetArrayFromImage(img)
        tableaux_numpy.append(arr)
        
    # --- ÉTAPE 2 : CALCUL DES STATISTIQUES GLOBALES ---
    # On met tous les pixels de toutes les phases "à plat" dans un immense tableau 1D
    pixels_globaux = np.concatenate([arr.flatten() for arr in tableaux_numpy])
    
    # ASTUCE PRO : On ignore le fond noir (l'air autour du patient) pour les statistiques.
    # Si on inclut l'air (qui vaut 0), la moyenne serait artificiellement rabaissée.
    masque_tissus = pixels_globaux > 1e-4
    
    if np.any(masque_tissus):
        moyenne_globale = np.mean(pixels_globaux[masque_tissus])
        ecart_type_global = np.std(pixels_globaux[masque_tissus])
    else:
        # Sécurité extrême au cas où l'image serait entièrement noire
        moyenne_globale = np.mean(pixels_globaux)
        ecart_type_global = np.std(pixels_globaux)
        
    # Sécurité mathématique : éviter la division par zéro
    if ecart_type_global == 0:
        ecart_type_global = 1e-8
        
    # --- ÉTAPE 3 : APPLICATION DU Z-SCORE ET SAUVEGARDE ---
    for img, arr, chemin_sortie in zip(images_sitk, tableaux_numpy, chemins_sorties):
        # Formule du Z-Score : Z = (Valeur - Moyenne) / Ecart-Type
        arr_normalise = (arr - moyenne_globale) / ecart_type_global
        
        # On remet le fond strictement à zéro pour garder un arrière-plan propre (bruit=0)
        arr_normalise[arr <= 1e-4] = 0.0
        
        # Re-transformation du tableau numpy en image SimpleITK
        img_normalisee = sitk.GetImageFromArray(arr_normalise)
        
        # CRUCIAL : Recopier les métadonnées (coordonnées spatiales, espacement des pixels)
        # Sinon l'image n'est plus alignée avec le masque !
        img_normalisee.CopyInformation(img)
        
        # Sauvegarde sur le disque
        sitk.WriteImage(img_normalisee, chemin_sortie)

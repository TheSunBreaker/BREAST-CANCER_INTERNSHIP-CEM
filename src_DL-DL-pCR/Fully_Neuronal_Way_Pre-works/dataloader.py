#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
DATALOADER PYTORCH MULTIMODAL (IRM / PET-CT / CLINIQUE)
=============================================================================
Ce script est la "porte d'entrée" vers les réseaux de neurones (ResNet/DenseNet/MLP).

[ PRÉREQUIS - PLACE DE CE SCRIPT DANS LE PIPELINE ] :
1. Ingestion DICOM : Terminée (Plastimatch / dcm2niix).
2. Hub nnU-Net     : Terminé (Alignement spatial parfait via SimpleITK).
3. Tenseurs NPY    : Terminés (Générés par le script de Crop 96x96x96).
4. Fichier Clinique: Terminé (One-Hot Encoded, mais contenant des NaN et non-scalé).

[ RÔLES DE CE SCRIPT ] :
1. "ClinicalPreprocessor" : Impute les NaN (médiane) et Standardise (Z-score) 
   strictement sur les données d'entraînement pour éviter le Data Leakage.
2. "BreastMultimodalDataset" : Charge les Tenseurs .npy à la volée.
3. Applique le Clipping et Min-Max Scaling physique pour le PET/CT.
4. Applique la Data Augmentation spatiale (via MONAI) à la volée.
=============================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import joblib # Pour save les statistiques de standardisation du dossier clinique

# MONAI est la référence mondiale pour la Data Augmentation médicale 3D
try:
    import monai.transforms as mt
except ImportError:
    raise ImportError("Veuillez installer MONAI : pip install monai")

# Fonction pour filtrer les partients et ne garder que ceux qui ont leurs modalités au complet
def get_multimodal_intersection(tensor_dir: str, clinical_df: pd.DataFrame) -> list:
    valid_patients = []
    # 1. On liste tous les dossiers d'images
    all_subjects = [d for d in os.listdir(tensor_dir) if os.path.isdir(os.path.join(tensor_dir, d))]
    
    for subj in all_subjects:
        subj_dir = os.path.join(tensor_dir, subj)
        
        # 2. Vérification Clinique
        if subj not in clinical_df.index:
            continue
            
        # 3. Vérification Imagerie (DCE + CT + PET)
        # On s'assure que les 3 phases IRM et les 2 images nucléaires existent
        mri_ok = all(os.path.exists(os.path.join(subj_dir, f"{subj}_MRI_phase{i}.npy")) for i in range(3))
        ct_ok = os.path.exists(os.path.join(subj_dir, f"{subj}_CT.npy"))
        pet_ok = os.path.exists(os.path.join(subj_dir, f"{subj}_PET.npy"))
        
        if mri_ok and ct_ok and pet_ok:
            valid_patients.append(subj)
            
    print(f"[FILTRE] Patientes ayant 100% des modalités : {len(valid_patients)} / {len(clinical_df)}")
    return valid_patients

# =============================================================================
# 1. LE PROCESSEUR CLINIQUE (Anti Data-Leakage)
# =============================================================================
class ClinicalPreprocessor:
    """
    S'occupe des données cliniques. 
    L'astuce de pointe : On 'fit' (apprend) la médiane et l'écart-type UNIQUEMENT 
    sur les patients d'entraînement (train_ids), et on 'transform' tout le monde.
    """
    def __init__(self, target_col="pCR"):
        self.target_col = target_col
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit_transform(self, df: pd.DataFrame, train_ids: list):
        """
        Apprend les statistiques sur le train_set, et transforme le train_set.
        """
        # On exclut la cible (pCR) et l'ID des features à standardiser
        cols_to_exclude = [self.target_col, "ReferenceID", "ACRONYME"]
        self.feature_cols = [c for c in df.columns if c not in cols_to_exclude]
        
        # Isoler les données d'entraînement
        df_train = df.loc[df["ReferenceID"].isin(train_ids)].copy()
        X_train = df_train[self.feature_cols]
        
        # 1. Apprentissage & Imputation des NaN (médiane du train)
        X_train_imp = self.imputer.fit_transform(X_train)
        
        # 2. Apprentissage & Standardisation (Z-score du train)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        
        # On met à jour le DataFrame d'entraînement
        df_train.loc[:, self.feature_cols] = X_train_scaled
        return df_train

    def transform(self, df: pd.DataFrame, test_ids: list):
        """
        Applique les statistiques apprises (sur le train) aux patients de test/validation.
        """
        df_test = df.loc[df["ReferenceID"].isin(test_ids)].copy()
        X_test = df_test[self.feature_cols]
        
        # Transformation uniquement (PAS DE FIT ICI !)
        X_test_imp = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imp)
        
        df_test.loc[:, self.feature_cols] = X_test_scaled
        return df_test


# =============================================================================
# 2. LE DATASET PYTORCH (Imagerie + Clinique)
# =============================================================================
class BreastMultimodalDataset(Dataset):
    def __init__(
        self, 
        tensor_dir: str, 
        processed_clinical_df: pd.DataFrame, 
        patient_ids: list,
        is_training: bool = True
    ):
        """
        Args:
            tensor_dir: Chemin vers le dossier des tenseurs .npy (Crop 96x96x96).
            processed_clinical_df: DataFrame clinique sans NaN et standardisé.
            patient_ids: Liste des "ReferenceID" à charger (ex: la fold courante).
            is_training: Si True, active la Data Augmentation MONAI.
        """
        self.tensor_dir = tensor_dir
        self.is_training = is_training
        
        # On s'assure que l'ID patient est bien l'index du DataFrame pour un accès direct O(1)
        if "ReferenceID" in processed_clinical_df.columns:
            self.clinical_df = processed_clinical_df.set_index("ReferenceID")
        else:
            self.clinical_df = processed_clinical_df

        self.patient_ids = [pid for pid in patient_ids if pid in self.clinical_df.index]

        # --- DATA AUGMENTATION (MONAI) ---
        # Crucial pour éviter le sur-apprentissage sur nos petits jeux de données
        if self.is_training:
            self.spatial_transforms = mt.Compose([
                # 1. Rotations légères (± 15°)
                mt.RandRotate(range_x=0.26, range_y=0.26, range_z=0.26, prob=0.5),
                
                # 2. Flips miroirs (Droite/Gauche et Haut/Bas)
                mt.RandFlip(spatial_axis=0, prob=0.5),
                mt.RandFlip(spatial_axis=1, prob=0.5),
                
                # 3. Déformation élastique (Crucial en oncologie pour simuler la variabilité des formes tumorales)
                # Déforme la tumeur comme si elle était en "caoutchouc"
                mt.Rand3DElastic(
                    sigma_range=(5, 8), 
                    magnitude_range=(100, 200), 
                    prob=0.3,
                    padding_mode="zeros"
                ),
                
                # 4. Zoom aléatoire
                mt.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
                
                # 5. Bruit Gaussien (Simule le bruit des capteurs d'imagerie)
                mt.RandGaussianNoise(prob=0.2, mean=0.0, std=0.05)
            ])
        else:
            self.spatial_transforms = mt.Compose([]) # Rien en test/val

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        subj = self.patient_ids[idx]
        subj_dir = os.path.join(self.tensor_dir, subj)
        
        # =========================================================
        # 1. CHARGEMENT IRM (3 Phases)
        # =========================================================
        data_mri = []
            # L'IRM est DEJA normalisée globalement (MAMA-MIA), on charge juste les 3 phases
            # Ordre attendu : Phase0, Phase1, Phase2
          for i in range(3):
              phase_path = os.path.join(subj_dir, f"{subj}_MRI_phase{i}.npy")
              # On ajoute une dimension [1, Z, Y, X] pour permettre la concaténation
              data_mri.append(np.expand_dims(np.load(phase_path), axis=0)) 
            
            # Concaténation finale : [3, 96, 96, 96]
            mri_array = np.concatenate(data_mri, axis=0) 
            
        # =========================================================
        # 2. CHARGEMENT PET/CT (Clipping & Scaling)
        # =========================================================
        # PET/CT ne sont PAS normalisés. Ils contiennent les vrais SUV et HU physiques.
        ct_array = np.load(os.path.join(subj_dir, f"{subj}_CT.npy"))
        pet_array = np.load(os.path.join(subj_dir, f"{subj}_PET.npy"))
            
        # --- Clipping & Scaling Spécifique ---
        # CT : Focalisation sur les tissus mous (Sein/Tumeur). On ignore l'air profond et les os.
        ct_clipped = np.clip(ct_array, -150.0, 250.0)
        # Min-Max Scaling -> ramène entre 0.0 et 1.0 ( (val - min) / (max - min) )
        ct_scaled = (ct_clipped + 150.0) / 400.0
        # PET : Clipping strict pour éviter qu'un point très chaud (ex: vessie à 50 SUV) écrase le contraste
        pet_clipped = np.clip(pet_array, 0.0, 15.0)
        # Min-Max Scaling -> ramène entre 0.0 et 1.0
        pet_scaled = pet_clipped / 15.0
        
        # Empilement : [2, 96, 96, 96] (Canal 0 = CT, Canal 1 = PET)
        petct_array = np.stack([ct_scaled, pet_scaled], axis=0)

        # =========================================================
        # 3. DATA AUGMENTATION SYNCHRONISÉE
        # =========================================================
        # On fusionne temporairement pour que MONAI applique la MÊME rotation aux deux
        combined_array = np.concatenate([mri_array, petct_array], axis=0) # Shape: (5, 96, 96, 96)
        combined_tensor = torch.FloatTensor(combined_array)
        
        if self.is_training:
            combined_tensor = self.spatial_transforms(combined_tensor)
            
        # On dépile après l'augmentation !
        img_mri_tensor = combined_tensor[:3, :, :, :]    # Les 3 premiers canaux
        img_petct_tensor = combined_tensor[3:, :, :, :]  # Les 2 derniers canaux

        # =========================================================
        # CHARGEMENT CLINIQUE & CIBLE
        # =========================================================
        # On extrait toutes les colonnes SAUF la cible pCR
        cols_features = [c for c in self.clinical_df.columns if c != "pCR"]
        clin_row = self.clinical_df.loc[subj, cols_features].values
        clin_tensor = torch.tensor(clin_row.astype(np.float32))
        
        # Le label (La pCR à prédire)
        label = float(self.clinical_df.loc[subj, 'pCR'])
        label_tensor = torch.tensor(label, dtype=torch.float32) # Pour Binary Cross Entropy avec Logits

        # On retourne un dictionnaire très pratique
        return {
            "id": subj,
            "mri": img_mri_tensor,       # (3, 96, 96, 96) -> Vers le ResNet + LSTM
            "petct": img_petct_tensor,   # (2, 96, 96, 96) -> Vers le DenseNet
            "clinical": clin_tensor,     # (N_features)    -> Vers le MLP
            "label": label_tensor
        }

# =============================================================================
# EXEMPLE D'UTILISATION (Comment l'appeler dans le script d'entraînement)
# =============================================================================
if __name__ == "__main__":
    
    # 1. Fichiers sources
    TENSOR_DIR = "./img_tensors_96"
    CLINICAL_CSV = "./ready_steady_clinicals.xlsx" # Généré par le script clinique
    
    df_clinical_brut = pd.read_excel(CLINICAL_CSV)

    valid_ids = get_multimodal_intersection(TENSOR_DIR, df_clinical_brut)
    
    # Simulation d'un Split Cross-Validation (K-Fold)
    # all_patients = df_clinical_brut["ReferenceID"].tolist()
    all_patients = valid_ids
    train_patients = all_patients[:int(len(all_patients)*0.8)] # 80%
    val_patients = all_patients[int(len(all_patients)*0.8):]   # 20%
    
    # 2. Imputation et Standardisation Anti-Leakage
    processor = ClinicalPreprocessor(target_col="pCR")
    
    # Apprend sur Train, Transforme Train
    df_train_clean = processor.fit_transform(df_clinical_brut, train_patients)

    # On save les stats de standardisation :
    joblib.dump(processor, "clinical_preprocessor_fold1.pkl")

    # Le jour de l'inférence dans X mois/sems :
    # processor = joblib.load("clinical_preprocessor_fold1.pkl")
    # df_new_patient = processor.transform(df_new_patient_brut)

    # Transforme Val avec les stats du Train !
    df_val_clean = processor.transform(df_clinical_brut, val_patients)
    
    # 3. Création des Datasets PyTorch
    train_dataset = BreastMultimodalDataset(
        tensor_dir=TENSOR_DIR,
        processed_clinical_df=df_train_clean,
        patient_ids=train_patients,
        is_training=True # Active MONAI
    )
    
    val_dataset = BreastMultimodalDataset(
        tensor_dir=TENSOR_DIR,
        processed_clinical_df=df_val_clean,
        patient_ids=val_patients,
        is_training=False # Désactive MONAI (Test pur)
    )
    
    # 4. DataLoaders (Prêts à nourrir le modèle)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Test d'affichage
    batch = next(iter(train_loader))
    print("\n--- TEST DATALOADER ---")
    print(f"Shape du Batch IRM (B, C, Z, Y, X) : {batch['mri'].shape}")
    print(f"Shape du Batch PET/CT (B, C, Z, Y, X) : {batch['petct'].shape}")
    print(f"Shape du Batch Clinique (B, Features) : {batch['clinical'].shape}")
    print(f"Labels pCR attendus : {batch['label']}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
MODÈLE MULTIMODAL DE PRÉDICTION pCR (ONCOLOGIE)
=============================================================================
Conçu selon l'architecture :
1. Branche IRM (Cinétique) : Time-Distributed ResNet3D -> LSTM -> Vecteur (Emb_MRI)
2. Branche PET/CT (Densité/Métabolisme) : DenseNet3D -> Vecteur (Emb_PETCT)
3. Branche Clinique : MLP -> Vecteur (Emb_Clin)
4. Tête de Classification : Concaténation (Emb_MRI, Emb_PETCT, Emb_Clin) -> MLP final -> pCR
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# On utilise les modèles 3D pré-entraînés (ou à initialiser) de MONAI
try:
    from monai.networks.nets import resnet10, DenseNet121
except ImportError:
    raise ImportError("Installez MONAI pour accéder aux backbones 3D : pip install monai")

# =============================================================================
# 1. LA FONCTION DE PERTE : FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    """
    Idéale pour les jeux de données déséquilibrés en oncologie.
    Applique un poids dynamique (gamma) qui réduit la pénalité pour les cas 
    faciles à prédire, forçant le modèle à se concentrer sur les cas limites.
    
    Formule : FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # BCEWithLogitsLoss combine une Sigmoid et la BCE classique de manière numériquement stable
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # Calcul de la BCE classique
        bce_loss = self.bce(logits, targets)
        # Transformation des logits en probabilités via Sigmoid
        probs = torch.sigmoid(logits)
        
        # p_t est la probabilité de la vraie classe
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Poids de la Focal Loss
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# =============================================================================
# 2. ARCHITECTURE DU MODÈLE
# =============================================================================
class Weirdly_Busty_Cerberus(nn.Module):
    def __init__(self, num_clinical_features, mri_hidden_dim=128, petct_hidden_dim=128, clin_hidden_dim=32):
        super(Weirdly_Busty_Cerberus, self).__init__()
        
        # ---------------------------------------------------------
        # BRANCHE 1 : IRM (DCE Séquentiel)
        # Objectif : Extraire la cinétique de rehaussement
        # ---------------------------------------------------------
        # On utilise un ResNet10 3D léger (pour éviter l'explosion de la VRAM).
        # in_channels=1 car on passe chaque phase séquentiellement.
        self.mri_backbone = resnet10(spatial_dims=3, n_input_channels=1, num_classes=mri_hidden_dim)
        
        # Le LSTM prendra les features du ResNet pour chaque phase (T=3)
        # batch_first=True signifie que nos tenseurs seront de taille (Batch, Seq_len, Features)
        self.mri_lstm = nn.LSTM(
            input_size=mri_hidden_dim, 
            hidden_size=mri_hidden_dim, 
            num_layers=1, 
            batch_first=True,
            dropout=0.0
        )
        
        # ---------------------------------------------------------
        # BRANCHE 2 : PET / CT (Multimodalité Statique)
        # Objectif : Extraire la morphologie et le métabolisme
        # ---------------------------------------------------------
        # in_channels=2 (Canal 0: CT, Canal 1: PET). DenseNet est excellent pour
        # fusionner les caractéristiques à plusieurs échelles.
        # On coupe la tête de classification classique pour extraire les features.
        self.petct_backbone = DenseNet121(spatial_dims=3, in_channels=2, out_channels=petct_hidden_dim)
        
        # ---------------------------------------------------------
        # BRANCHE 3 : CLINIQUE
        # Objectif : Encoder le statut hormonal, âge, stade...
        # ---------------------------------------------------------
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, clin_hidden_dim),
            nn.ReLU()
        )
        
        # ---------------------------------------------------------
        # TÊTE DE FUSION (CLASSIFICATION FINALE)
        # ---------------------------------------------------------
        fusion_dim = mri_hidden_dim + petct_hidden_dim + clin_hidden_dim
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4), # Fort dropout final pour éviter le sur-apprentissage
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 1 seul logit de sortie pour la pCR (Binary Classification)
        )

    def forward(self, mri_tensor, petct_tensor, clinical_tensor):
        """
        Passage avant du modèle.
        mri_tensor : (B, 3, 96, 96, 96)
        petct_tensor: (B, 2, 96, 96, 96)
        clinical_tensor: (B, N_features)
        """
        B = mri_tensor.size(0)
        
        # --- 1. TRAITEMENT DE L'IRM (Time-Distributed) ---
        # mri_tensor a la forme (B, T=3, Z, Y, X)
        T = mri_tensor.size(1)
        Z, Y, X = mri_tensor.size(2), mri_tensor.size(3), mri_tensor.size(4)
        
        # On aplatit le Batch et le Temps pour passer tout d'un coup dans le ResNet 3D
        # Forme : (B*3, 1, Z, Y, X)
        mri_reshaped = mri_tensor.view(B * T, 1, Z, Y, X)
        
        # Extraction spatiale : (B*3, mri_hidden_dim)
        mri_features = self.mri_backbone(mri_reshaped)
        
        # On restaure la dimension temporelle pour le LSTM : (B, T, mri_hidden_dim)
        mri_sequence = mri_features.view(B, T, -1)
        
        # Extraction temporelle : hn contient le dernier état caché
        lstm_out, (hn, cn) = self.mri_lstm(mri_sequence)
        emb_mri = hn[-1] # On prend la dernière sortie (fin de la phase d'injection) -> (B, mri_hidden_dim)
        
        # --- 2. TRAITEMENT PET/CT ---
        # Le DenseNet prend directement les 2 canaux en entrée
        emb_petct = self.petct_backbone(petct_tensor) # (B, petct_hidden_dim)
        
        # --- 3. TRAITEMENT CLINIQUE ---
        emb_clin = self.clinical_mlp(clinical_tensor) # (B, clin_hidden_dim)
        
        # --- 4. FUSION ---
        # On concatène les 3 embeddings sur l'axe des features (dim=1)
        fused_features = torch.cat([emb_mri, emb_petct, emb_clin], dim=1) # (B, fusion_dim)
        
        # Prédiction finale
        logits = self.fusion_classifier(fused_features) # (B, 1)
        
        # On retourne un tenseur aplati (B) au lieu de (B, 1) pour correspondre aux labels
        return logits.squeeze(1) 

# =============================================================================
# EXEMPLE DE TEST (DRY RUN)
# =============================================================================
if __name__ == "__main__":
    # Paramètres simulés
    BATCH_SIZE = 4
    NUM_CLINICAL_FEATURES = 15 # Dépend du nombre de colonnes du DataFrame encodé
    
    # Instanciation du modèle et de la loss
    model = Weirdly_Busty_Cerberus(num_clinical_features=NUM_CLINICAL_FEATURES)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Simulation d'un batch provenant du Dataloader
    # IRM : 3 phases DCE, 96x96x96
    mock_mri = torch.randn(BATCH_SIZE, 3, 96, 96, 96)
    # PETCT : 2 canaux (CT, PET), 96x96x96
    mock_petct = torch.randn(BATCH_SIZE, 2, 96, 96, 96)
    # Clinique : 15 variables
    mock_clinical = torch.randn(BATCH_SIZE, NUM_CLINICAL_FEATURES)
    # Labels : 0 (non-pCR) ou 1 (pCR)
    mock_labels = torch.empty(BATCH_SIZE).random_(2)
    
    # Forward Pass
    print("Passage des tenseurs dans le réseau...")
    logits = model(mock_mri, mock_petct, mock_clinical)
    
    # Calcul de l'erreur
    loss = criterion(logits, mock_labels)
    
    print(f"Shape des logits : {logits.shape}")
    print(f"Valeur de la Focal Loss initiale : {loss.item():.4f}")
    
    # Nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle prêt ! Nombre de paramètres entraînables : {total_params:,}")

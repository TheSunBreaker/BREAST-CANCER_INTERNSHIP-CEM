#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
BOUCLE D'ENTRAÎNEMENT - "WEIRDLY BUSTY CERBERUS"
=============================================================================
Entraîne le modèle multimodal de prédiction de la pCR.

Fonctionnalités avancées incluses :
1. Accumulation de Gradients : Simule un grand Batch Size (ex: 16) tout en 
   gardant un petit Batch Size réel (ex: 2) pour ne pas exploser la VRAM GPU.
2. Checkpointing : Sauvegarde le modèle à chaque époque et garde une copie 
   isolée du "Meilleur Modèle" (Best AUC) sur l'ensemble de validation.
3. Résilience : Capacité à reprendre l'entraînement exactement là où il s'est
   arrêté en cas de crash (Poids, Optimiseur, Scheduler, Epoch).
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib
# Force matplotlib à ne pas chercher d'interface graphique (évite les plantages sur serveur distant)
matplotlib.use('Agg') 

# TODO : Imports
from dataloader import BreastMultimodalDataset, DataLoader
from Weirdly_Busty_Cerberus import Weirdly_Busty_Cerberus, FocalLoss


# =============================================================================
# OUTIL POUR MONITORING
# =============================================================================
class TrainingMonitor:
    """
    Observatoire du Cerbère.
    Traque les métriques, écrit un fichier de log et dessine les courbes en temps réel.
    """
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.out_dir, "training_log.txt")
        self.plot_file = os.path.join(self.out_dir, "training_curves.png")
        
        # Initialisation du fichier de log
        with open(self.log_file, "w") as f:
            f.write("=== LOG D'ENTRAÎNEMENT : WEIRDLY BUSTY CERBERUS ===\n")
            f.write("Epoch\tTime(s)\tLR\tTrainLoss\tValLoss\tValAUC\tValAcc\n")
            
        # Historiques pour les graphiques
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_acc": [],
            "lr": []
        }

    def update(self, epoch, epoch_time, lr, train_loss, val_loss, val_auc, val_acc):
        """Met à jour les historiques et écrit dans le log textuel."""
        # Mise à jour des listes
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_auc"].append(val_auc)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)
        
        # Écriture dans le fichier
        with open(self.log_file, "a") as f:
            f.write(f"{epoch}\t{epoch_time:.1f}\t{lr:.6f}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_auc:.4f}\t{val_acc:.4f}\n")
            
    def draw_plots(self):
        """Génère un tableau de bord PNG de l'entraînement."""
        # Création d'une figure avec 3 sous-graphiques horizontaux
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Tableau de Bord : Weirdly Busty Cerberus', fontsize=16, fontweight='bold')
        
        epochs = self.history["epoch"]
        
        # Graphique 1 : Les Pertes (Loss)
        axes[0].plot(epochs, self.history["train_loss"], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], label='Val Loss', color='red', linewidth=2, linestyle='--')
        axes[0].set_title("Évolution de la Focal Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        # Graphique 2 : Les Performances (AUC & Accuracy)
        axes[1].plot(epochs, self.history["val_auc"], label='Val AUC', color='purple', linewidth=2)
        axes[1].plot(epochs, self.history["val_acc"], label='Val Accuracy', color='green', linewidth=2, linestyle='-.')
        axes[1].set_title("Métriques de Performance")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score (0 à 1)")
        axes[1].set_ylim([0.0, 1.05])
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.6)
        
        # Graphique 3 : Le Taux d'apprentissage (Learning Rate)
        axes[2].plot(epochs, self.history["lr"], label='Learning Rate', color='orange', linewidth=2)
        axes[2].set_title("Scheduler (Cosine Annealing)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        # Échelle logarithmique car le LR varie souvent de 1e-3 à 1e-6
        axes[2].set_yscale('log') 
        axes[2].legend()
        axes[2].grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        # Sauvegarde et libération de la mémoire RAM
        plt.savefig(self.plot_file, dpi=150)
        plt.close(fig)

# =============================================================================
# FONCTIONS DE SAUVEGARDE ET DE REPRISE
# =============================================================================
def save_checkpoint(state, is_best, checkpoint_dir, filename="last_checkpoint.pth"):
    """
    Sauvegarde l'état complet de l'entraînement.
    Si c'est le meilleur modèle historique, crée une copie distincte 'best_model.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, last_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"   => [NOUVEAU RECORD] Modèle sauvegardé avec succès dans {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """
    Restaure les poids du modèle, l'état de l'optimiseur et l'époque de départ.
    """
    if os.path.isfile(checkpoint_path):
        print(f"=> Chargement du checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # CPU first pour la RAM
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint['best_val_auc']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"=> Reprise à l'époque {start_epoch} (Meilleur AUC val: {best_val_auc:.4f})")
        return start_epoch, best_val_auc
    else:
        print(f"=> [ERREUR] Aucun checkpoint trouvé à '{checkpoint_path}'")
        return 0, 0.0

# =============================================================================
# GELEUR-DEGELEUR ENCODEURS
# =============================================================================
def freeze_pretrained_encoders(model):
    """Gèle les poids des backbones pré-entraînés pour protéger leurs filtres."""
    for param in model.mri_backbone.parameters():
        param.requires_grad = False
    for param in model.petct_backbone.parameters():
        param.requires_grad = False
    print(" ❄️ [WARM-UP] Encodeurs (IRM & PET-CT) GELÉS. Seules les têtes apprennent.")

def unfreeze_pretrained_encoders(model):
    """Dégèle les poids pour le fine-tuning final."""
    for param in model.mri_backbone.parameters():
        param.requires_grad = True
    for param in model.petct_backbone.parameters():
        param.requires_grad = True
    print(" 🔥 [FINE-TUNING] Encodeurs DÉGELÉS. Entraînement de bout en bout activé.")

# =============================================================================
# BOUCLE PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================
def train_cerberus(
    model, train_loader, val_loader, 
    criterion, optimizer, scheduler, device, 
    num_epochs=100, 
    accumulation_steps=8, 
    checkpoint_dir="./checkpoints",
    resume_checkpoint=None
):

    # =============================================================================
    #    PREPARATION NECESSAIRES POUR SCHEDULER ET OPTIMISEUR
    # =============================================================================

    # 1. Hyperparamètres
    EPOCHS = 50
    UNFREEZE_EPOCH = 5 # On dégèle à l'époque 5
    
    # 2. Groupes de paramètres (Differential Learning Rates)
    # On sépare les encodeurs (qui ont besoin d'un tout petit LR) des têtes toutes neuves (qui ont besoin d'un gros LR)
    encoder_params = list(model.mri_backbone.parameters()) + list(model.petct_backbone.parameters())
    head_params = list(model.mri_lstm.parameters()) + list(model.clinical_mlp.parameters()) + list(model.fusion_classifier.parameters())
    
    # L'optimiseur AdamW avec deux taux d'apprentissage différents !
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5}, # Très petit LR pour ne pas casser MedicalNet
        {'params': head_params, 'lr': 1e-3}     # LR standard pour les nouvelles couches
    ], weight_decay=1e-4)

    # 3. Scheduler existant (CosineAnnealingLR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. On gèle avant de démarrer
    freeze_pretrained_encoders(model)

    # =============================================================================
    #    TRAIN
    # =============================================================================
    
    model = model.to(device)
    
    start_epoch = 0
    best_val_auc = 0.0
    
    # 1. Reprise sur erreur (si un chemin est fourni)
    if resume_checkpoint:
        start_epoch, best_val_auc = load_checkpoint(resume_checkpoint, model, optimizer, scheduler)

    # Initialisation du Moniteur
    monitor = TrainingMonitor(out_dir=checkpoint_dir)

    print("\n" + "="*50)
    print(f"RÉVEIL DU CERBÈRE : Lancement de l'entraînement")
    print(f"Device : {device} | Accumulation : {accumulation_steps} steps")
    print("="*50 + "\n")

    for epoch in range(start_epoch, num_epochs):
        
        # ---------------------------------------------------------------------
        # PHASE D'ENTRAÎNEMENT
        # ---------------------------------------------------------------------

        # Chronomètre de l'époque
        epoch_start_time = time.time()

        # Le DÉGEL DYNAMIQUE
        if epoch == UNFREEZE_EPOCH:
            unfreeze_pretrained_encoders(model)
       
        model.train() # Active le Dropout et le BatchNorm en mode train
        train_loss = 0.0
        optimizer.zero_grad() # On remet les gradients à zéro au début de l'époque
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] [TRAIN]")
        
        for step, batch in enumerate(train_pbar):
            # On envoie les 3 têtes (entrées) et la cible sur le GPU
            mri     = batch["mri"].to(device)
            petct   = batch["petct"].to(device)
            clin    = batch["clinical"].to(device)
            targets = batch["label"].to(device)
            
            # Forward pass (Prédiction)
            logits = model(mri, petct, clin)
            
            # Calcul de l'erreur (Focal Loss)
            loss = criterion(logits, targets)
            
            # Division de la loss par le nombre d'étapes d'accumulation
            # Indispensable pour que la moyenne mathématique des gradients soit correcte
            loss = loss / accumulation_steps
            
            # Backward pass (Rétropropagation de l'erreur)
            loss.backward()
            
            # Si on a atteint le nombre d'étapes requis (ex: 8 batchs de 2 = 16 images vues)
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # On met à jour les poids du modèle
                optimizer.step()
                # On remet les gradients à zéro pour le prochain "gros" batch
                optimizer.zero_grad()
                
            train_loss += loss.item() * accumulation_steps # Remise à l'échelle pour l'affichage
            
            # Affichage en temps réel
            train_pbar.set_postfix({"Loss": f"{(train_loss / (step + 1)):.4f}"})

            # Le scheduler met à jour le LR
            scheduler.step()
            
        avg_train_loss = train_loss / len(train_loader)

        # ---------------------------------------------------------------------
        # PHASE DE VALIDATION
        # ---------------------------------------------------------------------
        model.eval() # Désactive le Dropout, fixe le BatchNorm
        val_loss = 0.0
        
        all_targets = []
        all_probs = []
        
        # torch.no_grad() désactive le calcul des gradients : économise 50% de RAM et accélère le calcul
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] [VALID]")
            
            for batch in val_pbar:
                mri     = batch["mri"].to(device)
                petct   = batch["petct"].to(device)
                clin    = batch["clinical"].to(device)
                targets = batch["label"].to(device)
                
                logits = model(mri, petct, clin)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
                # Pour les métriques cliniques, on a besoin des probabilités (entre 0 et 1)
                probs = torch.sigmoid(logits)
                
                # On stocke les résultats (ramenés sur le CPU) pour le calcul AUC global
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calcul de l'AUC (Area Under the ROC Curve)
        # Attention: roc_auc_score plante si le batch de validation n'a qu'une seule classe (tout le monde est pCR ou non-pCR). 
        # On sécurise avec un try/except.
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            val_auc = 0.5 # Pire cas si classe unique
            print(" [WARNING] Impossible de calculer l'AUC (Une seule classe présente dans le fold Val).")
            
        # Conversion des probabilités en prédictions binaires (seuil à 0.5 par défaut)
        preds = (np.array(all_probs) >= 0.5).astype(int)
        val_acc = accuracy_score(all_targets, preds)

        # Fin du Chronomètre
        epoch_duration = time.time() - epoch_start_time
        
        # Récupération du Learning Rate actuel (Tête)
        current_lr = optimizer.param_groups[-1]['lr']
        
        print(f" => Bilan Epoch {epoch+1}, terminée en {epoch_duration:.0f}s : Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        # Mise à jour du Moniteur et Dessin du PNG
        monitor.update(
            epoch=epoch+1, 
            epoch_time=epoch_duration, 
            lr=current_lr, 
            train_loss=avg_train_loss, 
            val_loss=avg_val_loss, 
            val_auc=val_auc, 
            val_acc=val_acc
        )
        monitor.draw_plots()

        # ---------------------------------------------------------------------
        # SAUVEGARDE ET SCHEDULER
        # ---------------------------------------------------------------------
        # Mise à jour du taux d'apprentissage (Learning Rate)
        if scheduler:
            scheduler.step()
            
        # Est-ce le nouveau meilleur modèle ?
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            
        # Création du "paquet" de sauvegarde
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_auc': best_val_auc,
            'val_loss': avg_val_loss
        }
        
        save_checkpoint(checkpoint_state, is_best, checkpoint_dir)

    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    print(f"Meilleur score AUC sur la validation : {best_val_auc:.4f}")

# =============================================================================
# BLOC D'EXÉCUTION PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # 1. Configuration Matérielle
    # Utilise le GPU si disponible, sinon tombe sur le CPU (très lent pour la 3D)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Hyperparamètres (À adapter selon la VRAM)
    BATCH_SIZE_REEL = 2
    BATCH_SIZE_CIBLE = 16
    ACCUMULATION_STEPS = BATCH_SIZE_CIBLE // BATCH_SIZE_REEL # = 8
    
    NUM_CLINICAL_FEATURES = 15
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    
    # 3. Initialisation de "Weirdly Busty Cerberus" (Supposé importé)
    # model = Weirdly_Busty_Cerberus(num_clinical_features=NUM_CLINICAL_FEATURES)
    
    # 4. Fonction de Perte et Optimiseur
    # Focal Loss pour gérer le déséquilibre pCR vs non-pCR
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # AdamW est supérieur à Adam standard car il gère mieux la régularisation du poids (Weight Decay)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Le Scheduler réduit doucement le taux d'apprentissage au fil des époques en forme de courbe Cosinus
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Lancement de l'arène !
    # train_cerberus(
    #    model=model, train_loader=train_loader, val_loader=val_loader,
    #    criterion=criterion, optimizer=optimizer, scheduler=scheduler,
    #    device=device, num_epochs=EPOCHS, accumulation_steps=ACCUMULATION_STEPS,
    #    resume_checkpoint=None # Mettre "./checkpoints/last_checkpoint.pth" si plantage
    # )
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnunet_manager.py
Script couteau-suisse pour gérer tout le cycle de vie d'un modèle nnU-Net V2.
Permet d'automatiser le preprocessing, l'entraînement séquentiel (pour éviter les OOM GPU),
l'inférence avec ensembling automatique, le fine-tuning, la reprise sur sauvegarde,
et le monitoring dynamique des courbes d'apprentissage.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import threading
import re
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------
# CONFIGURATION DES CHEMINS 
# ---------------------------------------------------------
BASE_DIR = Path(os.path.abspath("./nnunet_data"))
NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

def setup_env():
    """Injecte les chemins vitaux dans l'environnement système de Python."""
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

def run_command(cmd_list):
    """Wrapper sécurisé pour exécuter les commandes bash."""
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] Lancement de la commande :\n{cmd_str}\n" + "-"*40)
    
    try:
        subprocess.run(cmd_list, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR CRITIQUE] La commande nnU-Net a échoué avec le code retour {e.returncode}.")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERREUR CRITIQUE] L'exécutable nnU-Net est introuvable sur le système. Avez-vous activé votre environnement virtuel ?")
        sys.exit(1)

# ---------------------------------------------------------
# OUTILS DE MONITORING (NOUVEAU)
# ---------------------------------------------------------
def monitor_training_log(run_output_dir: Path, dataset_id: str, fold: str):
    """
    Tourne en tâche de fond. Cherche le fichier log le plus récent,
    le lit en temps réel et envoie les métriques à TensorBoard.
    """
    # 1. Configuration TensorBoard
    tb_dir = NNUNET_RESULTS / "tensorboard_logs" / f"{dataset_id}_fold_{fold}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    print(f"\n[MONITORING] En attente de la création du log par nnU-Net dans : {run_output_dir.name}...")
    
    # 2. Attente active : on cherche le fichier texte le plus récent
    log_file_path = None
    while log_file_path is None:
        time.sleep(2)
        # Cherche tous les fichiers qui commencent par "training_log"
        logs = list(run_output_dir.glob("training_log_*.txt"))
        if logs:
            # S'il y en a plusieurs (ex: reprise sur sauvegarde), on prend le dernier modifié
            log_file_path = max(logs, key=os.path.getmtime)
            
    print(f"[MONITORING] Log détecté ({log_file_path.name}) ! Lancement de TensorBoard.")
    
    current_epoch = 0
    
    # 3. Lecture en direct (Tail)
    with open(log_file_path, "r") as f:
        f.seek(0, 2) # On se place à la fin
        
        while getattr(threading.current_thread(), "do_run", True):
            line = f.readline()
            if not line:
                time.sleep(2)
                continue
            
            line = line.strip()
            
            if line.startswith("Epoch"):
                match = re.search(r"Epoch (\d+)", line)
                if match:
                    current_epoch = int(match.group(1))
            
            elif "train_loss" in line:
                match = re.search(r"train_loss\s*[:=]\s*([\d\.]+)", line)
                if match:
                    writer.add_scalar("Loss/Train", float(match.group(1)), current_epoch)
                    
            elif "val_loss" in line:
                match = re.search(r"val_loss\s*[:=]\s*([\d\.]+)", line)
                if match:
                    writer.add_scalar("Loss/Validation", float(match.group(1)), current_epoch)
                    
            elif "Pseudo dice" in line or "pseudo dice" in line.lower():
                floats = re.findall(r"[\d\.]+", line.split(":")[-1])
                if floats:
                    tumor_dice = float(floats[-1])
                    writer.add_scalar("Metrics/Pseudo_Dice_Tumor", tumor_dice, current_epoch)
                    print(f" 📈 [Époque {current_epoch}] TB Mis à jour -> Dice Tumeur: {tumor_dice:.4f}")

# ---------------------------------------------------------
# FONCTIONS MÉTIERS 
# ---------------------------------------------------------

def do_preprocess(dataset_id: str):
    print(f"--- DÉMARRAGE PREPROCESSING (Dataset {dataset_id}) ---")
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"]
    run_command(cmd)

def do_train(dataset_id: str, config: str, fold: str, resume: bool, pretrained_weights: str, trainer: str):
    print(f"--- DÉMARRAGE ENTRAÎNEMENT (Dataset {dataset_id} | Config: {config} | Fold: {fold} | Trainer: {trainer}) ---")

    folds = ["0", "1", "2", "3", "4"] if fold == "all" else [fold]
    print(f"[INFO] Folds qui vont être entraînés séquentiellement : {folds}")

    for f in folds:

        # NOUVEAU : On ajoute le flag -tr suivi du nom de la classe du Trainer
        cmd = ["nnUNetv2_train", dataset_id, config, f, "-tr", trainer]
        
        # --- NOUVEAU : REPRISE SUR SAUVEGARDE ---
        if resume:
            print("[INFO] Option --resume activée. Reprise de l'entraînement à partir du dernier checkpoint.")
            cmd.append("--c")
            
        # --- NOUVEAU : FINE TUNING (TRANSFER LEARNING) ---
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                print(f"[ERREUR] Le fichier de poids {pretrained_weights} n'existe pas.")
                sys.exit(1)
            print(f"[INFO] Fine-Tuning activé à partir de : {pretrained_weights}")
            cmd.extend(["-pretrained_weights", pretrained_weights])

        # --- DÉMARRAGE DU MONITORING EN TÂCHE DE FOND ---
        # Attention : Le nom du dossier de sortie change si on utilise un Custom Trainer !
        # Format nnU-Net : nnUNetTrainer__3d_fullres ou nnUNetTrainer_500epochs__3d_fullres
        run_output_dir = NNUNET_RESULTS / dataset_id / f"{trainer}__{config}" / f"fold_{f}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        monitor_thread = threading.Thread(
            target=monitor_training_log, 
            args=(run_output_dir, dataset_id, f)
        )
        monitor_thread.do_run = True # Permet de tuer le thread proprement plus tard
        monitor_thread.start()
        
        # --- LANCEMENT DE L'ENTRAÎNEMENT ---
        run_command(cmd)
        
        # Quand l'entraînement est fini, on dit au thread de s'arrêter
        monitor_thread.do_run = False
        monitor_thread.join()

def do_predict(dataset_id: str, config: str, fold: str, input_folder: str, output_folder: str, trainer: str):
    print(f"--- DÉMARRAGE INFÉRENCE (Dataset {dataset_id} | Config: {config} | Fold: {fold} | Trainer: {trainer}) ---")
    
    in_path = Path(input_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists() or not any(in_path.iterdir()):
        print(f"[ERREUR] Le dossier d'entrée {in_path} est vide ou n'existe pas.")
        sys.exit(1)

    folds = ["0", "1", "2", "3", "4"] if fold == "all" else [fold]
    print(f"[INFO] Modèles utilisés pour la prédiction (Ensembling) : {folds}")
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_path),
        "-o", str(out_path),
        "-d", dataset_id,
        "-c", config,
        "-f"
    ] + folds + [
        "-tr", trainer,  # NOUVEAU : Inférence avec le même Trainer !
        "-save_probabilities"
    ]

    run_command(cmd)

def do_evaluate(ground_truth_folder: str, prediction_folder: str):
    print(f"--- DÉMARRAGE ÉVALUATION ---")
    gt_path = Path(ground_truth_folder)
    pred_path = Path(prediction_folder)
    
    if not gt_path.exists() or not pred_path.exists():
        print("[ERREUR] Les dossiers de vérité terrain ou de prédiction sont introuvables.")
        sys.exit(1)

    cmd = [
        "nnUNetv2_evaluate_folder",
        "-g", str(gt_path),
        "-p", str(pred_path),
        "-djfile", str(pred_path / "evaluation_summary.json"),
        "-pfile", str(pred_path / "evaluation_summary.csv")
    ]
    run_command(cmd)

# ---------------------------------------------------------
# PARSER ARGUMENTS TERMINAL
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Couteau Suisse pour orchestrer nnU-Net V2 proprement")
    
    parser.add_argument("action", choices=["preprocess", "train", "predict", "evaluate"], 
                        help="L'action principale à exécuter")
    
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="ID numérique ou nom du dataset (ex: '001' ou 'Dataset001_Breast')")
    
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", 
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        help="Topologie du U-Net. Par défaut: 3d_fullres")
    
    parser.add_argument("-f", "--fold", type=str, default="0", 
                        help="Quel fold utiliser (0-4) ou 'all' pour tous les folds.")

    # --- NOUVEL ARGUMENT : LE TRAINER ---
    parser.add_argument("-tr", "--trainer", type=str, default="nnUNetTrainer",
                        help="Nom de la classe du Trainer. Le changer pour utiliser, par exemple, un modèle avec moins d'époques (ex: nnUNetTrainer_250epochs).")
    
    # --- NOUVEAUX ARGUMENTS POUR L'ENTRAÎNEMENT ---
    parser.add_argument("--resume", action="store_true",
                        help="[TRAIN] Reprend l'entraînement là où il s'est arrêté (si crash ou timeout)")
    
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="[TRAIN] Chemin vers un fichier .pth pour faire du Transfer Learning (Fine-Tuning)")

    # Arguments predict
    parser.add_argument("-i", "--input", type=str, 
                        help="[PREDICT] Chemin du dossier contenant les Nifti à segmenter")
    parser.add_argument("-o", "--output", type=str, 
                        help="[PREDICT] Chemin du dossier où sauvegarder les Nifti générés")

    # Arguments evaluate
    parser.add_argument("-g", "--ground_truth", type=str, 
                        help="[EVALUATE] Dossier contenant les masques de vérité terrain")
    parser.add_argument("-p", "--predictions", type=str, 
                        help="[EVALUATE] Dossier contenant les prédictions du modèle")

    args = parser.parse_args()

    setup_env()

    if args.action == "preprocess":
        do_preprocess(args.dataset)
        
    elif args.action == "train":
        do_train(args.dataset, args.config, args.fold, args.resume, args.pretrained_weights, args.trainer)
        
    elif args.action == "predict":
        if not args.input or not args.output:
            parser.error("L'action 'predict' requiert impérativement les drapeaux -i (--input) et -o (--output).")
        do_predict(args.dataset, args.config, args.fold, args.input, args.output)

    elif args.action == "evaluate":
        if not args.ground_truth or not args.predictions:
            parser.error("L'action 'evaluate' requiert les arguments -g (--ground_truth) et -p (--predictions).")
        do_evaluate(args.ground_truth, args.predictions)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
===============================================================================
 Couteau-Suisse nnU-Net V2 - Sécurisé HPC
===============================================================================
Script pour gérer tout le cycle de vie d'un modèle nnU-Net V2.
Permet d'automatiser le preprocessing, l'entraînement séquentiel (anti OOM),
l'inférence avec ensembling automatique, le fine-tuning, et la reprise.

Sécurités intégrées :
  - Environnement HPC (Copie propre de os.environ).
  - Validation CUDA pré-entraînement (Affichage du GPU alloué).
  - Recherche dynamique du dossier de logs (gestion des plans).
  - Threading robuste pour TensorBoard (Lecture historique + Tail + Flush).
===============================================================================

DEPENDANCES REQUISES :

pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install --user graphviz tensorboard hiddenlayer 
Cloner le git nnUNet et suivre les instructions du git.


ATTENTION, SUR LE SERVER SUR LEQUEL A ETE TESTE CE CODE, LA MEMOIRE PARTAGEE DES DOCKERS LIMITEE ENPECHE D'UTILISER PLUSIEURS WORKERS POUR LE DATALOADER ET LA DATA AUGMNTATION. ALORS IL FAUT 0 COMME NOMBRE DE WORKERS
DE LA VARIABLE D'ENVIRONNEMENT 'nnUNet_n_proc_DA'. VOIR JUSTE EN BAS DANS LE CODE. CPEENDANT, LE PREPROCESSING LUI NE SUPPORTE PAS 0. ALORS IL VAUT MIEEUX METTRE A 1 POUR LE PREPROCESSING ET A 0 POUR LE TRAIN
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import threading
import re

# Rendre torch optionnel pour le parsing d'arguments, mais obligatoire pour le train
try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------
# CONFIGURATION DES CHEMINS 
# ---------------------------------------------------------
BASE_DIR = Path(os.path.abspath("./nnunet_data"))
NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

def setup_env():
    """Injecte les chemins vitaux dans l'environnement système."""
    # On fait une vraie copie pour ne pas corrompre l'environnement hôte
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    
    # Force l'utilisation d'un seul thread pour certaines librairies C++ (HPC)
    env.setdefault("OMP_NUM_THREADS", "1")

    # --- AJOUT CRITIQUE POUR ÉVITER LE CRASH SHARED MEMORY ---
    # Réduit drastiquement le nombre de workers (Défaut: 12) à 2 ou 4.
    # L'entraînement sera un tout petit peu plus long, mais il ne crashera plus.
    env["nnUNet_n_proc_DA"] = "0" 
    # ---------------------------------------------------------

    # --- NOUVELLE SÉCURITÉ ANTI-CRASH COMPILATEUR (Triton / gcc) ---
    # Désactive torch.compile pour éviter l'erreur "Failed to find C compiler"
    env["nnUNet_compile"] = "F"
    env["TORCH_COMPILE_DISABLE"] = "1"
    # ---------------------------------------------------------------

    # --- SOLUTION LOGIQUE POUR LE RECOURS AU FLAG --USER DANS LE SERVEUR ---
    # On ajoute le répertoire des binaires locaux de l'utilisateur (~/.local/bin)
    # au début du PATH pour que subprocess trouve nnUNetv2_plan_and_preprocess
    local_bin_path = str(Path.home() / ".local" / "bin")
    if "PATH" in env:
        env["PATH"] = local_bin_path + os.path.pathsep + env["PATH"]
    else:
        env["PATH"] = local_bin_path
    # --------------------------------------------------------
    
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)
    
    return env

def run_command(cmd_list, env_dict):
    """Wrapper sécurisé pour exécuter les commandes bash."""
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] Lancement de la commande :\n{cmd_str}\n" + "-"*40)
    
    try:
        # On passe directement la sortie console, SLURM s'en chargera
        subprocess.run(cmd_list, check=True, env=env_dict)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERREUR CRITIQUE] nnU-Net a échoué (Code retour: {e.returncode}).")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERREUR CRITIQUE] Exécutable nnU-Net introuvable (Env virtuel actif ?).")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTION] Arrêt manuel demandé (Ctrl+C).")
        sys.exit(0)

# ---------------------------------------------------------
# OUTILS DE MONITORING INTELLIGENT
# ---------------------------------------------------------

def find_latest_run_dir(dataset_id: str, trainer: str, config: str, fold: str) -> Path:
    """
    Cherche dynamiquement le dossier le plus récent qui matche les critères.
    """
    # Si plusieurs datasets commencent par cet ID, on prend le plus récent
    ds_dirs = sorted(list(NNUNET_RESULTS.glob(f"Dataset{dataset_id}_*")), key=os.path.getmtime, reverse=True)
    if not ds_dirs:
        return None
        
    ds_dir = ds_dirs[0] 
    
    possible_runs = [
        d for d in ds_dir.iterdir() 
        if d.is_dir() and trainer in d.name and config in d.name
    ]
    
    if not possible_runs:
        return None
        
    latest_run = max(possible_runs, key=os.path.getmtime)
    fold_dir = latest_run / f"fold_{fold}"
    
    return fold_dir if fold_dir.exists() else None

def monitor_training_log(dataset_id: str, trainer: str, config: str, fold: str, stop_event: threading.Event):
    """
    Tourne en tâche de fond. Trouve le vrai dossier de log,
    lit l'historique puis écoute en temps réel, et envoie à TensorBoard.
    """
    if not TORCH_AVAILABLE:
        return

    tb_dir = NNUNET_RESULTS / "tensorboard_logs" / f"DS{dataset_id}_{trainer}_{config}_fold{fold}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    print(f"\n[MONITORING] En attente de la création des dossiers de run par nnU-Net...")
    
    try:
        run_output_dir = None
        log_file_path = None
        
        # 1. Attente de l'arborescence (Tolérance de 30 secondes)
        attempts = 0
        while run_output_dir is None and not stop_event.is_set():
            run_output_dir = find_latest_run_dir(dataset_id, trainer, config, fold)
            if run_output_dir is None:
                time.sleep(2)
                attempts += 1
                if attempts > 15:
                    print("[MONITORING] Timeout. Abandon du monitoring TensorBoard.")
                    return

        # Sécurité anti-crash au cas où stop_event est déclenché pendant l'attente
        if run_output_dir is None:
            return

        # 2. Attente du fichier .txt
        while log_file_path is None and not stop_event.is_set():
            logs = list(run_output_dir.glob("training_log_*.txt"))
            if logs:
                log_file_path = max(logs, key=os.path.getmtime)
            else:
                time.sleep(2)
                
        if stop_event.is_set() or log_file_path is None:
            return
            
        print(f"[MONITORING] Connecté au fichier : {log_file_path.name}. Lancement TensorBoard.")
                
        current_epoch = None

        # 3. Lecture historique + Tailing robuste
        with open(log_file_path, "r", encoding="utf-8") as f:
            # Lecture depuis le début pour gérer resume + arrivée tardive
            while not stop_event.is_set():
                line = f.readline()
        
                if not line:
                    time.sleep(1)
                    continue
        
                line = line.strip()
        
                if not line:
                    continue
        
                line_lower = line.lower()
        
                # --- Détection robuste de l'epoch ---
                epoch_match = re.search(r"epoch\s+(\d+)", line_lower)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    continue
        
                # Tant qu'aucune epoch valide n'a été vue,
                # on ignore les métriques pour éviter epoch=0 fantôme
                if current_epoch is None:
                    continue
        
                # --- Train loss ---
                if "train_loss" in line_lower:
                    match = re.search(r"train_loss\s*[:=]\s*([-\d\.]+)", line_lower)
                    if match:
                        writer.add_scalar(
                            "Loss/Train",
                            float(match.group(1)),
                            current_epoch
                        )
        
                # --- Validation loss ---
                elif "val_loss" in line_lower:
                    match = re.search(r"val_loss\s*[:=]\s*([-\d\.]+)", line_lower)
                    if match:
                        writer.add_scalar(
                            "Loss/Validation",
                            float(match.group(1)),
                            current_epoch
                        )
        
                # --- Dice ---
                elif "pseudo dice" in line_lower:
                    floats = re.findall(r"[\d\.]+", line_lower.split(":")[-1])
        
                    if floats:
                        tumor_dice = float(floats[-1])
        
                        writer.add_scalar(
                            "Metrics/Pseudo_Dice",
                            tumor_dice,
                            current_epoch
                        )
        
                        # Affichage console allégé
                        if current_epoch % 10 == 0:
                            print(
                                f" 📈 [Époque {current_epoch}] "
                                f"TB Mis à jour -> Dice Pseudo: {tumor_dice:.4f}"
                            )

    finally:
        # GARANTIE ABSOLUE : Fermeture propre de TensorBoard
        writer.close()
        print(f"[MONITORING] TensorBoard fermé proprement pour le fold {fold}.")

# ---------------------------------------------------------
# FONCTIONS MÉTIERS 
# ---------------------------------------------------------

def do_preprocess(dataset_id: str, env_dict: dict):
    print(f"--- DÉMARRAGE PREPROCESSING (Dataset {dataset_id}) ---")
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"]
    run_command(cmd, env_dict)

def do_train(dataset_id: str, config: str, fold: str, resume: bool, pretrained_weights: str, trainer: str, env_dict: dict):
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("[ERREUR CRITIQUE] CUDA ou PyTorch est indisponible sur ce système.")
        print("L'entraînement nnU-Net requiert impérativement un GPU actif.")
        sys.exit(1)

    # Affichage du hardware alloué par le cluster / Colab
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n[HARDWARE] Entraînement lancé sur GPU : {gpu_name}")
    print(f"--- DÉMARRAGE ENTRAÎNEMENT (DS {dataset_id} | Config: {config} | Fold: {fold} | Trainer: {trainer}) ---")

    folds = ["0", "1", "2", "3", "4"] if fold == "all" else [fold]
    print(f"[INFO] Folds qui vont être entraînés séquentiellement : {folds}")

    for f in folds:
        cmd = ["nnUNetv2_train", dataset_id, config, f, "-tr", trainer]
        
        if resume:
            print(f"[INFO] Reprise sur sauvegarde activée (--c) pour le fold {f}.")
            cmd.append("--c")
            
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                print(f"[ERREUR] Le fichier de poids {pretrained_weights} n'existe pas.")
                sys.exit(1)
            print(f"[INFO] Fine-Tuning activé : {pretrained_weights}")
            cmd.extend(["-pretrained_weights", pretrained_weights])

        # Initialisation du Threading sécurisé (Daemon)
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_training_log, 
            args=(dataset_id, trainer, config, f, stop_event),
            daemon=True 
        )
        monitor_thread.start()
        
        try:
            run_command(cmd, env_dict)
        finally:
            stop_event.set()
            monitor_thread.join(timeout=3)

def do_predict(dataset_id: str, config: str, fold: str, input_folder: str, output_folder: str, trainer: str, env_dict: dict):
    print(f"--- DÉMARRAGE INFÉRENCE (DS {dataset_id} | Config: {config} | Fold: {fold} | Trainer: {trainer}) ---")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"[HARDWARE] Inférence accélérée sur GPU : {torch.cuda.get_device_name(0)}")
    
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
        "-tr", trainer,
        "--save_probabilities"
    ]

    run_command(cmd, env_dict)

def do_evaluate(ground_truth_folder: str, prediction_folder: str, env_dict: dict):
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
    run_command(cmd, env_dict)

# ---------------------------------------------------------
# PARSER ARGUMENTS TERMINAL
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Couteau Suisse pour orchestrer nnU-Net V2 proprement sur HPC/Colab")
    
    parser.add_argument("action", choices=["preprocess", "train", "predict", "evaluate"], 
                        help="L'action principale à exécuter")
    
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="ID numérique du dataset (ex: '002')")
    
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", 
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        help="Topologie du U-Net. Par défaut: 3d_fullres")
    
    parser.add_argument("-f", "--fold", type=str, default="0", 
                        help="Quel fold utiliser (0-4) ou 'all' pour tous les folds.")

    parser.add_argument("-tr", "--trainer", type=str, default="nnUNetTrainer",
                        help="Nom de la classe du Trainer (ex: nnUNetTrainer_250epochs).")
    
    # Arguments TRAIN
    parser.add_argument("--resume", action="store_true",
                        help="[TRAIN] Reprend l'entraînement là où il s'est arrêté (si crash ou timeout)")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="[TRAIN] Chemin vers un .pth pour Transfer Learning")

    # Arguments PREDICT
    parser.add_argument("-i", "--input", type=str, 
                        help="[PREDICT] Chemin du dossier contenant les Nifti à segmenter")
    parser.add_argument("-o", "--output", type=str, 
                        help="[PREDICT] Chemin du dossier où sauvegarder les Nifti générés")

    # Arguments EVALUATE
    parser.add_argument("-g", "--ground_truth", type=str, 
                        help="[EVALUATE] Dossier contenant les masques de vérité terrain")
    parser.add_argument("-p", "--predictions", type=str, 
                        help="[EVALUATE] Dossier contenant les prédictions du modèle")

    args = parser.parse_args()
    
    # Configuration sécurisée de l'environnement (HPC)
    env_dict = setup_env()

    try:
        if args.action == "preprocess":
            do_preprocess(args.dataset, env_dict)
            
        elif args.action == "train":
            do_train(args.dataset, args.config, args.fold, args.resume, args.pretrained_weights, args.trainer, env_dict)
            
        elif args.action == "predict":
            if not args.input or not args.output:
                parser.error("L'action 'predict' requiert impérativement -i (--input) et -o (--output).")
            do_predict(args.dataset, args.config, args.fold, args.input, args.output, args.trainer, env_dict)

        elif args.action == "evaluate":
            if not args.ground_truth or not args.predictions:
                parser.error("L'action 'evaluate' requiert -g (--ground_truth) et -p (--predictions).")
            do_evaluate(args.ground_truth, args.predictions, env_dict)
            
    except KeyboardInterrupt:
        print("\n[INFO] Fermeture de l'orchestrateur suite à Ctrl+C.")
        sys.exit(0)

if __name__ == "__main__":
    main()

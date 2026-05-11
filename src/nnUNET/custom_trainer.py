r"""
    Trainer pour nnUNET customisé. On peut y changer le nombre d'époques par défaugt et bien encore. A mettre dans "nnunetv2/training/nnUNetTrainer/variants/custom/".
    Pour l'utiliser, en terminal : python nnunet_manager.py train -d 001 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs
    Changer le nom de la classe pour être plus explic besoin. Préciser le même trainer à l'inférence (pour la bonne localisation des poids)
    nnUnet va scanner son propre code source interne (spécifiquement le dossier nnunetv2/training/nnUNetTrainer/variants/) pour trouver un fichier .py 
    contenant une classe de ce nom.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # C'est ici que la magie opère ! On écrase les 1000 époques par défaut.
        self.num_epochs = 250

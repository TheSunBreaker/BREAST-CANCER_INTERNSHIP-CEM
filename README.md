# 🧬 Prédiction pCR - Cancer du Sein Triple Négatif (Stage CEM)

## 📝 Description du Projet
[cite_start]Ce projet de recherche s'inscrit dans le cadre de mon stage au Centre Eugène Marquis (CEM)[cite: 3]. [cite_start]Il a pour objectif de prédire de manière précoce et non-invasive la **Réponse Pathologique Complète (pCR)** chez les patientes atteintes d'un cancer du sein triple négatif, suite à un traitement néoadjuvant.

Pour y parvenir, le projet exploite une approche IA multimodale intégrant :
* [cite_start]**Images IRM (DCE/T1)** : Analyse spatio-temporelle de la cinétique du contraste.
* [cite_start]**Images TEP et TDM (CT)** : Cartographie de l'activité métabolique et de la densité tissulaire.
* [cite_start]**Dossier Clinique** : Intégration des variables démographiques et biologiques (âge, stade, etc.).

### Stratégie Méthodologique
Le pipeline de traitement est divisé en deux axes de recherche consécutifs :

1. **Phase 1 : Approche Hybride (Deep Learning + Machine Learning)**
   [cite_start]Utilisation de l'architecture nnU-Net (v2) pour la segmentation automatique de la tumeur et des zones d'intérêt[cite: 3]. [cite_start]Cette étape est suivie d'une extraction massive de caractéristiques radiomiques (via PyRadiomics et algorithmes) et d'un pipeline de classification Machine Learning robuste (Nested Cross-Validation, sélection de variables univariée et multivariée avec SelectKBest, ElasticNet, Extra Trees etc.)[cite: 3].

2. **Phase 2 : Approche 100% Neuronale (Modèle WB CERBERUS)**
   *En cours de développement.* Une architecture multimodale de bout en bout qui s'affranchit de l'extraction radiomique manuelle. [cite_start]Elle intègre des encodeurs pré-entraînés (ResNet 3D, DenseNet 3D), des réseaux récurrents (LSTM) pour capturer la dynamique temporelle des phases DCE, et des réseaux perceptrons multicouches (MLP) pour l'encodage des features cliniques, mais aussi pour la prédiction pCR finale[cite: 8].

---

## 📁 Structure du Dépôt

Le projet est organisé autour de deux répertoires principaux reflétant la stratégie méthodologique :
* [cite_start]📂 **`[`Deep_Learning_Machine_Learning/`](./Deep_Learning_Machine_Learning/)`** : Contient tous les scripts relatifs à la Phase 1 (Ingestion DICOM/NIfTI, prétraitements, inférence nnU-Net, extraction radiomique et entraînement des modèles ML classiques)[cite: 3].
* 📂 **`100_Pourcent_Neuronal/`** : Héberge le code relatif à la Phase 2 et au développement du modèle WB CERBERUS[cite: 8].

---

## ⚙️ Prérequis et Installation

L'environnement de travail a été éprouvé sur plusieurs configurations matérielles. Voici les instructions pour installer les dépendances selon votre infrastructure.

### 1. Installation sur Serveur de Calcul (Docker, sans droits root)
Si vous opérez dans un conteneur sur un serveur de calcul sans les permissions d'administration globales, privilégiez l'installation avec l'option `--user`.

Installez d'abord PyTorch (ici compilé pour CUDA 12.4) :
```bash
pip3 install --user torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

Installez ensuite les outils de suivi et de visualisation :

Bash

```
pip3 install --user graphviz tensorboard hiddenlayer synapseclient

```

### 2. Installation sur Machine Locale (Environnement virtuel)

Pour l'exécution des scripts de traitement, d'extraction radiomique et de Machine Learning en local (testé sous Python 3.11.9) :

Bash

```
pip install pydicom==2.3.1 SimpleITK tqdm TotalSegmentator pydicom-seg pandas openpyxl numpy cython versioneer xlsxwriter scikit-learn matplotlib seaborn joblib
pip install pyradiomics --no-build-isolation

```

### 3. Utilisation ponctuelle sur Google Colab

Pour exécuter des tâches spécifiques déportées sur le cloud (comme la segmentation des régions d'intérêt via TotalSegmentator) :

Bash

```
!pip install pydicom SimpleITK tqdm TotalSegmentator

```

### 4. Installation et Configuration de nnU-Net (v2)

**Prérequis indispensable :** PyTorch doit être installé sur votre environnement avant de procéder à l'installation de nnU-Net.

1.  Récupérez le code source officiel :
    
    Bash
    
    ```
    git clone [https://github.com/MIC-DKFZ/nnUNet.git](https://github.com/MIC-DKFZ/nnUNet.git)
    cd nnUNet
    
    ```
    
    _(Alternativement : téléchargez le dépôt en `.zip` et décompressez-le avant de vous placer dans le dossier)._
    
2.  Installez le package en mode éditable pour l'utilisateur courant :
    
    Bash
    
    ```
    pip install -e . --user
    
    ```
    
3.  **Ajout des binaires au `$PATH` (Indispensable sous Linux) :** L'installation avec l'option `--user` place les exécutables (comme `nnUNetv2_predict`) dans le dossier `~/.local/bin`. Pour que votre terminal puisse les trouver, ajoutez ce répertoire à votre variable d'environnement `$PATH` :
    
    Bash
    
    ```
    export PATH=$HOME/.local/bin:$PATH
    
    ```
    
    _(Note : Pour rendre cette modification permanente, ajoutez cette ligne à la fin de votre fichier `~/.bashrc` ou `~/.zshrc`)._

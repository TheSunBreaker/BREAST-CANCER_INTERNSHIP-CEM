# 🧬 Prédiction pCR - Cancer du Sein Triple Négatif (Stage CEM)

## 📝 Description du Projet

Ce projet de recherche s'inscrit dans le cadre de mon projet de stage de Master 1 en Intelligence Artificielle (Université de Rennes), réalisé en collaboration avec le Centre de Lutte Contre le Cancer Eugène Marquis (CEM), l'équipe @OMICs (du laboratoire OSS du CEM) et l'équipe IMPACT (LTSI).
L'objectif principal est de **prédire précocement et de manière non-invasive la Réponse Pathologique Complète (pCR)** chez les patientes atteintes d'un cancer du sein triple négatif (CSTN), suite à un traitement néoadjuvant. 

Pour y parvenir, il exploite une approche IA multimodale intégrant :
**Images IRM (DCE/T1)** : Analyse spatio-temporelle de la cinétique du contraste.
**Images TEP et TDM (CT)** : Cartographie de l'activité métabolique et de la densité tissulaire.
**Dossier Clinique** : Intégration des variables démographiques et biologiques (âge, stade, etc.).

Il est divisé en deux approches méthodologiques, organisées dans deux répertoires principaux :

* [**`src_DL-ML-pCR/`**](./src_DL-ML-pCR) : Phase 1 - Approche Hybride (Deep Learning + Machine Learning).
   Utilisation de l'architecture nnU-Net (v2) pour la segmentation automatique de la tumeur et des zones d'intérêt. Cette étape est suivie d'une extraction massive de caractéristiques radiomiques (via PyRadiomics et algorithmes) et d'un pipeline de classification Machine Learning robuste (Nested Cross-Validation, sélection de variables univariée et multivariée avec SelectKBest, ElasticNet, Extra Trees etc.).
* [**`src_DL-DL-pCR/`**](./src_DL-DL-pCR) : Phase 2 - Approche 100% Neuronale (Modèle WB CERBERUS).
   *En cours de développement.* Une architecture multimodale de bout en bout qui s'affranchit de l'extraction radiomique manuelle. Elle intègre des encodeurs pré-entraînés (ResNet 3D, DenseNet 3D), des réseaux récurrents (LSTM) pour capturer la dynamique temporelle des phases DCE, et des réseaux perceptrons multicouches (MLP) pour l'encodage des features cliniques, mais aussi pour la prédiction pCR finale.

---

## ⚙️ Prérequis et Installation

L'environnement de travail a été éprouvé sur plusieurs configurations matérielles. Voici les instructions pour installer les dépendances selon votre infrastructure.

### 1. Installation sur serveur de calcul (Docker, sans droits root)

Si vous opérez dans un conteneur sur un serveur de calcul sans les permissions d'administration globales (Python 3.11.6 dans mon cas), privilégiez l'installation avec l'option `--user`.

Installez d'abord PyTorch (ici compilé pour CUDA 13.2) :

```bash
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Installez ensuite les outils de suivi et de visualisation :

```bash
pip3 install --user graphviz tensorboard hiddenlayer synapseclient
```

---

### 2. Installation sur machine locale (environnement virtuel)

Pour l'exécution des scripts de traitement, d'extraction radiomique et de Machine Learning en local (testé sous Python 3.11.9) :

```bash
pip install pydicom==2.3.1 SimpleITK tqdm TotalSegmentator pydicom-seg pandas openpyxl numpy cython versioneer xlsxwriter scikit-learn matplotlib seaborn joblib
pip install pyradiomics --no-build-isolation
```

---

### 3. Utilisation ponctuelle sur Google Colab

Pour exécuter des tâches spécifiques déportées sur le cloud (comme la segmentation des régions d'intérêt via TotalSegmentator) :

```bash
!pip install pydicom SimpleITK tqdm TotalSegmentator
```

---

### 4. Installation et configuration de nnU-Net (v2)

> **Prérequis indispensable :** PyTorch doit être installé sur votre environnement avant de procéder à l'installation de nnU-Net.

1. Récupérez le code source officiel :

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
```

*Alternativement : téléchargez le dépôt au format `.zip`, décompressez-le, puis placez-vous dans le dossier.*

2. Installez le package en mode éditable pour l'utilisateur courant :

```bash
pip install -e . --user
```

3. **Ajoutez les binaires au `$PATH` (indispensable sous Linux).**

L'installation avec l'option `--user` place les exécutables (comme `nnUNetv2_predict`) dans le dossier `~/.local/bin`. Pour que votre terminal puisse les trouver, ajoutez ce répertoire à votre variable d'environnement `$PATH` :

```bash
export PATH=$HOME/.local/bin:$PATH
```

> **Remarque :** Pour rendre cette modification permanente, ajoutez cette ligne à la fin de votre fichier `~/.bashrc` ou `~/.zshrc`.

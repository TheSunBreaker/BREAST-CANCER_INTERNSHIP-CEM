# 🧬 Prédiction pCR - Cancer du Sein Triple Négatif (Stage CEM)

## 📝 Description du Projet

Ce projet de recherche s'inscrit dans le cadre de mon projet de stage de Master 1 en Intelligence Artificielle (Université de Rennes), réalisé en collaboration avec le Centre de Lutte Contre le Cancer Eugène Marquis (CEM), l'équipe @OMICs (du laboratoire OSS du CEM) et l'équipe IMPACT (LTSI).
L'objectif principal est de **prédire précocement et de manière non-invasive la Réponse Pathologique Complète (pCR)** chez les patientes atteintes d'un cancer du sein triple négatif (CSTN), suite à un traitement néoadjuvant. 

Pour y parvenir, il exploite une approche IA multimodale intégrant :
**Images IRM (DCE/T1)** : Analyse spatio-temporelle de la cinétique du contraste.
**Images TEP et TDM (CT)** : Cartographie de l'activité métabolique et de la densité tissulaire.
**Dossier Clinique** : Intégration des variables démographiques et biologiques (âge, stade, etc.).

### Stratégie Méthodologique
Le pipeline de traitement est divisé en deux axes de recherche consécutifs :

1. **Phase 1 ([**`src_DL-ML-pCR/`**](./src_DL-ML-pCR)) : Approche Hybride (Deep Learning + Machine Learning)**
   Utilisation de l'architecture nnU-Net (v2) pour la segmentation automatique de la tumeur et des zones d'intérêt. Cette étape est suivie d'une extraction massive de caractéristiques radiomiques (via PyRadiomics et algorithmes) et d'un pipeline de classification Machine Learning robuste (Nested Cross-Validation, sélection de variables univariée et multivariée avec SelectKBest, ElasticNet, Extra Trees etc.).

2. **Phase 2 ([**`src_DL-DL-pCR/`**](./src_DL-DL-pCR)) : Approche 100% Neuronale (Modèle WB CERBERUS)**
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

### 3. Utilisation ponctuelle

Pour exécuter des tâches spécifiques (comme la segmentation des régions d'intérêt via TotalSegmentator) :
* Via NoteBook
```bash
!pip install pydicom SimpleITK tqdm TotalSegmentator
```
* Via terminal classique
```bash
pip install --user pydicom SimpleITK tqdm TotalSegmentator
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

## 🗄️ Jeux de données (Datasets) et Acquisition

*Note : Les données cliniques et d'imagerie privées du Centre Eugène Marquis (CEM) ne sont pas hébergées sur ce dépôt pour des raisons strictes de confidentialité et d'éthique. Cette section détaille la procédure d'acquisition des jeux de données publics utilisés pour l'entraînement et le pré-entraînement des modèles Deep Learning.*

### 1. Dataset QIN-Breast
Ce jeu de données public d'imagerie mammaire est utilisé pour consolider la phase d'entraînement.
* **Accès aux données :** [Collection QIN-Breast sur The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/qin-breast/)

### 2. Dataset MAMA-MIA Challenge
Ces données sont issues du challenge MAMA-MIA. Le téléchargement automatisé de ces cohortes nécessite une authentification préalable.
* **Dépôt officiel du challenge :** [GitHub LidiaGarrucho/MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA)

**Procédure d'acquisition automatisée :**
1. **Création de compte :** Inscrivez-vous sur la plateforme [Synapse](https://www.synapse.org/).
2. **Authentification :** Générez un *Personal Access Token* (Token d'accès) depuis les paramètres de votre compte Synapse.
3. **Exécution du script :** Rendez-vous dans le répertoire [`src_DL-ML-pCR/dataBringer/`](./src_DL-ML-pCR/dataBringer). Ce dossier contient les scripts (utilisant la librairie `synapseclient`) conçus pour se connecter via votre Token et télécharger automatiquement l'ensemble des données du challenge vers votre espace de travail.

## 🛠️ Prétraitement et Ingestion des Données

Le pipeline de prétraitement est divisé en plusieurs étapes majeures, dont le code majoritairement localisé dans les répertoires [`src_DL-ML-pCR/pre_works/`](./src_DL-ML-pCR/pre_works), et [`src_DL-ML-pCR/to_nnUnet_structure/`](./src_DL-ML-pCR/to_nnUnet_structure). Il permet de passer de dossiers DICOM bruts et désorganisés à une base de données NIfTI propre, triée longitudinalement et prête pour l'extraction radiomique ou l'entraînement **nnU-Net**.

---

## 1. Niveau 0 : Anonymisation stricte (`uncle_anonymiser.py`)

Ce script garantit la confidentialité des données cliniques avant toute manipulation complexe. Il remplace les identifiants patients (**IPP**) par des identifiants locaux de recherche tout en préservant les métadonnées indispensables (poids, sexe, temps d'acquisition, géométrie, etc.).

### Entrées requises

* `data_hopital_brut/` : dossier contenant les DICOM originaux.
* `data_clinique.csv` : fichier de correspondance contenant au minimum les colonnes :

  * `IPP`
  * `ID_PAT_LOCAL`

> **Format attendu :** séparateur `;`.

### Sortie générée

* `data_hopital_safe/` : miroir du dossier d'entrée, entièrement anonymisé.

### Exécution

```bash
python src_DL-ML-pCR/pre_works/uncle_anonymiser.py
```

---

## 2. Niveau 1 : Ingesteur et convertisseur NIfTI (`DICOM_ingester_but_even_meaner.py`)

Il s'agit du cœur du pipeline de prétraitement.

Ce script (version **V6**) parcourt la *Safe Zone*, regroupe les séries par examen (`StudyInstanceUID`), filtre automatiquement les reconstructions inutiles, ordonne chronologiquement les séquences **DCE-MRI**, puis convertit les données **DICOM** en **NIfTI**.

Il assure également l'association parfaite entre les masques de segmentation (dessinés par les radiologues) et leurs images sources.

⚠️ **Dépendances externes critiques :** 
> Ce script fait appel à deux outils incontournables qu'il faut télécharger et extraire sur votre machine (les chemins vers les exécutables doivent être mis à jour dans l'en-tête du script Python) :

-   **Plastimatch** (pour la conversion rigoureuse des TEP/TDM) : [Télécharger Plastimatch](https://sourceforge.net/projects/plastimatch/postdownload)
    
-   **dcm2niix** (pour la gestion experte des IRM et du 4D) : [Télécharger dcm2niix](https://github.com/rordenlab/dcm2niix/releases)

### Fonctionnalités principales

* ✅ **Tri temporel automatique**

  * Distinction entre les examens **Baseline** et les visites de **Follow-up**.

* ✅ **Découpage automatique des IRM 4D**

  * Séparation des volumes 4D en phases 3D indépendantes, indispensable pour la compatibilité avec **nnU-Net**.

* ✅ **Audit des examens PET**

  * Vérification de la présence des métadonnées nécessaires au calcul futur des **SUV** (dose injectée, poids, demi-vie du radiotraceur, etc.).

* ✅ **Filtrage des reconstructions**

  * Suppression automatique des séries dérivées (MIP, Subtraction, Scout, etc.).

### Entrée attendue

* `data_hopital_safe/` (généré par l'étape précédente).

### Sorties générées

* `Base_IRM/`

  * Images IRM et masques associés, organisés par patient et par visite (`imgs/` pour la baseline, `imgs_YYYYMMDD/` pour les suivis).

* `Base_PETCT/`

  * Images PET, CT et masques associés, organisés par patient et par visite (`imgs/` pour la baseline, `imgs_YYYYMMDD/` pour les suivis).

* `Base_Autres/`

  * Archive des modalités non exploitées dans ce pipeline (RTDOSE, PR, etc.), organisés par patient et par visite (`imgs/` pour la baseline, `imgs_YYYYMMDD/` pour les suivis).

* `rapport_ingestion_v6.txt`

  * Rapport d'audit complet généré à la racine du projet.

### Exécution

```bash
python src_DL-ML-pCR/pre_works/DICOM_ingester_but_even_meaner.py
```
### 3. Niveau 2 : Extraction et Fusion Intelligente des Masques (`dcm_masks_master.py`)

Les annotations radiologiques (**DICOM SEG** ou **RTSTRUCT**) peuvent être complexes : multi-classes, doublons, chevauchements ou lésions multiples. Ce script (V5) est un moteur d'analyse qui convertit ces contours vectoriels ou segmentés en grilles **NIfTI** parfaitement recalées sur les images de référence.

#### Fonctionnalités clés

* **Support Multi-Formats :**

  * Convertit les `SEG` (via `pydicom-seg`) et rastérise les `RTSTRUCT` en 3D (via l'appel externe à `Plastimatch`).

* **Analyse Sémantique Multi-classes :**

  * Différencie automatiquement la tumeur primaire (Classe 1) des ganglions lymphatiques/nodules (Classe 2) via l'analyse sémantique des métadonnées (recherche de mots-clés comme "GANGLION", "NODE", etc.).

* **Résolution des Conflits (Dice Score) :**

  * **Doublons (Dice ≥ 0.95)** : Conserve automatiquement le masque le plus récent ou le plus restrictif.
  * **Lésions distinctes (Dice < 0.20)** : Les fusionne sémantiquement dans un seul fichier multi-classes (`_FUSED.nii.gz`).
  * **Chevauchements ambigus** : Isole automatiquement les masques dans un dossier `a_verifier/` pour inspection manuelle par un radiologue.

#### Entrées attendues

* Les répertoires `Base_IRM/` et `Base_PETCT/` contenant les dossiers `dicom_mask_*` et les images de référence.

#### Sorties générées

* Dossiers `mask/` (Tumeurs) et `nodule/` (Ganglions) contenant les NIfTI prêts pour le Deep Learning.
* Fichier de suivi global `rapport_analyse_masques_v5.txt`.

#### Exécution

```bash
python src_DL-ML-pCR/pre_works/dcm_masks_master.py --mri_root ./Base_IRM --petct_root ./Base_PETCT
```

---

### 4. Niveau 3 : Normalisation Physique SUVbw pour la TEP (`suv_converter_nii_maker.py`)

Pour que les réseaux de neurones (et les extracteurs radiomiques) puissent analyser quantitativement l'imagerie TEP, les valeurs de pixels brutes (Coups ou Becquerels) doivent être impérativement converties en **SUVbw** (*Standardized Uptake Value based on body weight*).

#### Fonctionnalités clés

* **Extraction DICOM :**

  * Va fouiller dans les en-têtes DICOM bruts de la visite correspondante pour retrouver la dose injectée, le poids du patient, le temps de demi-vie du radiotraceur (ex: FDG) et les heures exactes d'injection et d'acquisition.

* **Fallback CSV :**

  * Permet d'utiliser un fichier clinique de secours si les systèmes PACS de l'hôpital ont effacé le poids du patient des métadonnées.

* **Suivi Longitudinal :**

  * Applique le bon facteur SUV à la Baseline et calcule indépendamment le facteur pour les suivis (Follow-ups).

* **Sécurité Anti-Double Conversion :**

  * Détecte si l'image NIfTI brute est déjà encodée en unité SUV par le constructeur et la copie sans l'altérer.

#### Entrée attendue

* Le dossier `Base_PETCT/` généré aux étapes précédentes, contenant les fichiers `_RAW.nii.gz`.

#### Sorties générées

* Nouveaux fichiers `_SUV.nii.gz` avec les valeurs d'intensité physiquement normalisées.
* Journal de conversion `suv_conversion_log.txt`.

#### Exécution

```bash
python src_DL-ML-pCR/pre_works/suv_converter_nii_maker.py ./Base_PETCT
```

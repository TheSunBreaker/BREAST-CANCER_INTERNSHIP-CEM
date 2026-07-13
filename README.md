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
## 🧠 Formatage pour nnU-Net (Deep Learning)

Une fois les données nettoyées, converties en NIfTI et normalisées physiquement, elles doivent être restructurées selon les spécifications strictes du framework **nnU-Net v2** :

* Renommage des canaux en `_0000`, `_0001`, etc.
* Séparation en `imagesTr` pour l'entraînement et `labelsTr` pour les masques.
* Génération automatique du fichier `dataset.json`.

Les scripts d'orchestration de cette étape se trouvent dans le répertoire [`src_DL-ML-pCR/to_nnUnet_structure/`](./src_DL-ML-pCR/to_nnUnet_structure).

---

# 1. Orchestrateur IRM DCE (`irm2nnunet_v2_PA.py`)

Ce script gère la complexité de l'imagerie dynamique **(4D)**.

Il analyse la cinétique du produit de contraste pour extraire les phases physiologiques pertinentes (**Jump-Anchored Time-Matching**) et réaligne strictement toutes les phases sur la grille spatiale de la **Baseline (T0)**.

## 📥 Entrées attendues (Structure source)

Le script s'attend à une arborescence contenant les images NIfTI (triées chronologiquement) et leurs masques associés, ainsi que le log temporel si utilisé en mode `INGESTEUR`.

```text
Base_IRM/
└── [ID_PATIENT]/
    ├── imgs/
    │   ├── image_01.nii.gz
    │   ├── image_02.nii.gz
    │   ├── ...
    │   └── DCE_temporal_log.txt  <-- Requis en mode INGESTEUR
    └── mask/
        └── masque_FUSED.nii.gz   <-- Requis (sauf si mode --inference)
```

> **Note :** En mode `MAMAMIA`, le fichier CSV externe contenant la timeline et le statut hormonal doit être fourni via le paramètre `--csv`.

---

## 📤 Sorties générées (Format nnU-Net)

Les données sont exportées et renommées avec l'identifiant des canaux requis par **nnU-Net**.

```plaintext
nnunet_data/nnUNet_raw/Dataset001_DCE/
├── dataset.json                  <-- Généré automatiquement
├── imagesTr/                     <-- (Ou imagesTs si --inference)
│   ├── [ID_PATIENT]_0000.nii.gz  (Canal 0 : Baseline)
│   ├── [ID_PATIENT]_0001.nii.gz  (Canal 1 : Wash-in immédiat)
│   ├── [ID_PATIENT]_0002.nii.gz  (Canal 2 : T_inj + 90s)
│   └── [ID_PATIENT]_0003.nii.gz  (Canal 3 : T_inj + 180s)
└── labelsTr/
    └── [ID_PATIENT].nii.gz       (Masque réaligné sur la Baseline)
```

Un rapport global `rapport_extraction_DCE_<mode>.txt` est également généré à la racine de `nnunet_data`.

### ▶️ Commande d'exécution (Mode Entraînement - 4 Canaux)

```bash
python src_DL-ML-pCR/to_nnUnet_structure/irm2nnunet_v2_PA.py \
    --src ./Base_IRM \
    --nnunet ./nnunet_data \
    --num_channels 4 \
    --mode INGESTEUR
```

---

# 2. Orchestrateur TEP / TDM (`pet_and_ct_2_nnunet.py`)

Ce script prépare les données d'imagerie métabolique et anatomique pour un entraînement multimodal.

L'image TEP (convertie en **SUV**) dicte la géométrie spatiale. Le volume TDM (**CT**) et le masque sont strictement ré-échantillonnés et recalés sur cette grille de référence, avec remplissage des zones vides par de l'air à **-1000.0 HU** pour le TDM.

---

## 📥 Entrées attendues (Structure source)

Le script s'attend à trouver les images NIfTI normalisées en **SUVbw** et l'image scanner (**CT**) dans le même dossier.

```plaintext
Base_PETCT/
└── [ID_PATIENT]/
    ├── imgs/
    │   ├── [ID]_TEP_Baseline_A1B2C_SUV.nii.gz  <-- PET normalisé (référence spatiale)
    │   └── [ID]_TDM_A1B2C.nii.gz               <-- CT à aligner
    └── mask/
        └── [ID]_mask_FUSED.nii.gz              <-- Requis (sauf si mode --inference)
```

---

## 📤 Sorties générées (Format nnU-Net)

```plaintext
nnunet_data/nnUNet_raw/Dataset002_BreastPETCT/
├── dataset.json                  <-- Généré automatiquement
├── imagesTr/                     <-- (Ou imagesTs si --inference)
│   ├── [ID_PATIENT]_0000.nii.gz  (Canal 0 : PET copié tel quel)
│   └── [ID_PATIENT]_0001.nii.gz  (Canal 1 : CT aligné sur le PET)
└── labelsTr/
    └── [ID_PATIENT].nii.gz       (Masque aligné sur le PET)
```

### ▶️ Commande d'exécution (Mode Entraînement)

```bash
python src_DL-ML-pCR/to_nnUnet_structure/pet_and_ct_2_nnunet.py \
    --src ./Base_PETCT \
    --nnunet ./nnunet_data
```

---

# 💡 Mode Inférence (Test)

Pour tous les orchestrateurs ci-dessus, l'ajout du flag `--inference` modifie le comportement du script :

* Il ne cherche pas de masques de vérité terrain (**labels**).
* Il n'écrase pas le fichier `dataset.json` de configuration d'entraînement.
* Il exporte les images directement dans le dossier `imagesTs` (**Test Set**).
* Les données sont prêtes à être ingérées par la commande :
```bash
nnUNetv2_predict
```
ou via le Couteau Suisse nnUNet que j'ai conçu pour toutes les interactions avec le modèle.

## 5. Séparation Train / Test Idempotente (`nnUNET_struct_adv_train-test_splitter.py`)

Avant de lancer l'entraînement, il est crucial d'isoler un jeu de données de test (**Test Set**) qui ne sera jamais vu par le modèle. Ce script effectue cette séparation de manière sécurisée et reproductible.

### Fonctionnalités clés

* **Idempotence :**

  * Le script peut être relancé sans risque.
  * S'il détecte que le quota de patients de test est déjà atteint, il s'arrête sans corrompre le dataset.

* **Gestion des Préférences :**

  * Possibilité de forcer l'inclusion de patients spécifiques (via leur ID ou index) dans le set de test.
  * Très utile pour s'assurer que des cas cliniques complexes sont gardés pour l'évaluation finale.

* **Mise à jour automatique :**

  * Ajuste automatiquement la valeur `numTraining` dans le fichier `dataset.json`.

### Action

Déplace un pourcentage défini (ex: **20%**) des fichiers depuis :

```
imagesTr / labelsTr
```

vers :

```
imagesTs / labelsTs
```

### Exemple d'exécution

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_struct_adv_train-test_splitter.py \
    --dataset_dir ./nnunet_data/nnUNet_raw/Dataset001_DCE \
    --ratio 0.20 \
    --pref QIN-BREAST-01-0005 QIN-BREAST-01-0012 0 1
```

---

# 🚀 Entraînement et Inférence : Le Couteau Suisse nnU-Net

Le script `nnUNET_v2_swiss_knife.py` situé dans [`src_DL-ML-pCR/nnUNET/`](./src_DL-ML-pCR/nnUNET/) est l'orchestrateur ultime pour interagir avec le framework **nnU-Net**.

## Pourquoi un Couteau Suisse ?

L'exécution native de nnU-Net sur des serveurs de calcul (via Docker ou Slurm) pose souvent des problèmes critiques liés à la limitation de la mémoire partagée (`/dev/shm`), entraînant des crashs lors de la Data Augmentation ou de l'export des prédictions (OOM).

Ce script encapsule l'exécution de nnU-Net pour patcher ces failles de manière transparente.

---

# Sécurités HPC & Docker implémentées

* **Bascule de la stratégie de partage PyTorch :**

  * Passage sur le système de fichiers (`file_system`) au lieu de la mémoire RAM partagée.

* **Forçage de `nnUNet_n_proc_DA=0` :**

  * Désactivation du parallélisme excessif lors de la Data Augmentation.

* **Désactivation conditionnelle du parallélisme :**

  * Lors de l'export de validation via la variable d'environnement personnalisée :

    ```bash
    NNUNET_DISABLE_PARALLEL_VAL_EXPORT
    ```

* **Monitoring TensorBoard threadé robuste :**

  * Gestion des reprises et de l'historique.

---

# Exemples d'utilisation du Couteau Suisse

## 1. Prétraitement (Plan & Preprocess)

Extraction des empreintes de données (*fingerprints*) et planification des architectures U-Net.

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py preprocess -d 001
```

---

## 2. Entraînement (Train)

Entraînement séquentiel ou ciblé, avec suivi TensorBoard automatique.

### Entraîner uniquement le fold 0

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py train -d 001 -c 3d_fullres -f 0
```

### Entraîner tous les folds (0 à 4) séquentiellement

*(Idéal pour éviter les crashs GPU)*

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py train -d 001 -c 3d_fullres -f all
```

> Options disponibles :
>
> * `--resume` : reprendre un entraînement interrompu.
> * `--pretrained_weights` : effectuer un Fine-Tuning.

---

## 3. Inférence (Predict)

Génération des masques de segmentation sur le **Test Set** avec **Ensembling** (combinaison des prédictions de plusieurs folds).

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py predict \
    -d 001 -c 3d_fullres -f all \
    -i ./nnunet_data/nnUNet_raw/Dataset001_DCE/imagesTs \
    -o ./nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test
```

---

## 4. Évaluation (Evaluate)

Calcul des métriques (**Dice**, **Hausdorff**, etc.) en comparant les prédictions à la vérité terrain.

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py evaluate \
    -g ./nnunet_data/nnUNet_raw/Dataset001_DCE/labelsTs \
    -p ./nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test
```

---

## 5. Export du Modèle (Export)

Package les poids finaux du modèle dans une archive `.zip` pour un déploiement ou un partage facilité.

```bash
python src_DL-ML-pCR/nnUNET/nnUNET_v2_swiss_knife.py export \
    -d 001 \
    --zip mon_modele_dce.zip
```

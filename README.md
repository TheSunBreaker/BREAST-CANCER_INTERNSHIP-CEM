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

### Entrées attendues (Structure source)

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

### Sorties générées (Format nnU-Net)

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

### Entrées attendues (Structure source)

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

### Sorties générées (Format nnU-Net)

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

## 💡 Mode Inférence (Test)

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
---

## 🎭 Segmentation ROI seins, et Raffinement du ROI

Du côté TEP-TDM, une segmentation précise des seins est nécessaire pour isoler le signal métabolique d'intérêt (extraction radiomique). 

Ce pipeline en deux étapes automatise la génération de masques robustes, capables d'exclure les structures anatomiques parasites (**cœur, sternum, côtes, poumons**) qui pourraient fausser l'analyse radiomique ou métabolique.

---

# 1. Segmentation Initiale et Extraction des "Boucliers" (`mrTotalSegmentator.py` : dans [`src_DL-ML-pCR/auto_segmentation_arsenal/`](./src_DL-ML-pCR/auto_segmentation_arsenal/))

Ce script utilise la puissance de **TotalSegmentator** pour deux objectifs :

* Segmenter le volume global des seins.
* Extraire les organes "boucliers" (structures à exclure de la ROI finale).

## Fonctionnalités clés

* **Segmentation Mammaire :**

  * Utilise le modèle spécifique `breasts` pour isoler le tissu mammaire.

* **Extraction Anatomique :**

  * Génère un masque pour :

    * le cœur,
    * le sternum,
    * les côtes,
    * les cartilages intercostaux,
    * les poumons,
    * les vertèbres,
    * les clavicules,
    * les pectoraux.

* **Mode Hybride :**

  * Supporte l'exécution via API Python (pour intégration) ou CLI (pour serveurs distants).

### Entrées attendues

* `Base_PETCT/`
  Dossier contenant les images CT de baseline.

### Sorties générées

* `Base_PETCT_BreastMasks/`

  * Masques mammaires bruts.

* `Base_PETCT_Organs/`

  * Dossier contenant les masques d'organes anatomiques.

## Exécution

```bash id="4p4j8m"
python src_DL-ML-pCR/auto_segmentation_arsenal/mrTotalSegmentator.py \
    --input_root ./Base_PETCT \
    --output_root ./Base_PETCT_BreastMasks \
    --output_organs_root ./Base_PETCT_Organs \
    --muscles_seg
```

---

# 2. Sculpture et Raffinement des ROI (`mrSegmentationGrower.py` : dans [`src_DL-ML-pCR/auto_segmentation_arsenal/`](./src_DL-ML-pCR/auto_segmentation_arsenal/))

Le masque généré par **TotalSegmentator** est un point de départ.

Ce script transforme ce masque brut en une ROI mammaire physiquement cohérente par une approche de **soustraction géométrique déterministe**, supprimant l'instabilité des anciennes méthodes de *Region Growing*.

## Fonctionnalités clés

* **Expansion Frontale Asymétrique :**

  * Dilate le masque vers la peau tout en respectant la limite de l'air ambiant.

* **Soustraction du Super-Bouclier :**

  * Fusionne les organes extraits précédemment (cœur, côtes, pectoraux, etc.) en un bloc massif et impénétrable.
  * Soustrait ensuite ce bloc géométriquement du masque mammaire afin de garantir l'exclusion totale du thorax.

* **Nettoyage Morphologique :**

  * Utilise des opérations d'ouverture et de filtrage par composantes connexes pour éliminer les îlots parasites et les artefacts de connexion entre les seins.

### Entrées attendues

* `Base_PETCT/`

  * Images CT.

* `Base_PETCT_BreastMasks/`

  * Masques mammaires bruts.

* `Base_PETCT_Organs/`

  * Boucliers anatomiques.

### Sortie générée

* `Base_PETCT_BreastMasks_Expanded/`

  * ROI finales, prêtes pour l'extraction de features ou l'entraînement.

## Exécution

```bash id="s7e8yx"
python src_DL-ML-pCR/auto_segmentation_arsenal/mrSegmentationGrower.py \
    --ct_dir ./Base_PETCT \
    --mask_dir ./Base_PETCT_BreastMasks \
    --organs_dir ./Base_PETCT_Organs \
    --output_dir ./Base_PETCT_BreastMasks_Expanded
```

# 3. Segmentation Tumorale Déterministe (`mrAutoSegmentator.py` : dans [`src_DL-ML-pCR/auto_segmentation_arsenal/`](./src_DL-ML-pCR/auto_segmentation_arsenal/))

Ce script, qui a des fins exploratoires (que vaut la méthode déterministe et "à la main" face au Deep Learning ?), implémente une approche par **seuillage métabolique adaptatif** (*Local Peak Segmentation*).

Contrairement au **Deep Learning**, il utilise une logique purement déterministe pour extraire les lésions :

* Segmentation des zones "chaudes" du PET (**SUV**) à l'intérieur de la ROI mammaire.
* Utilisation des données **CT** pour exclure les tissus denses (os/calcifications).

---

### Entrées attendues (Structure source)

Le script nécessite les images traitées par les étapes précédentes ainsi que les ROI mammaires "sculptées" par le script d'expansion.

```plaintext id="x9f4kp"
Base_PETCT/
└── [ID_PATIENT]/
    └── imgs/
        ├── [ID]_TEP_Baseline_XXX_SUV.nii.gz  (PET SUVbw normalisé)
        └── [ID]_TDM_XXX.nii.gz               (CT anatomique)

Base_PETCT_BreastMasks_Expanded/
└── [ID_PATIENT]_breast_roi_V6.nii.gz         (La ROI mammaire sculptée)
```

---

### Sorties générées (Résultat)

```plaintext id="n7m2qs"
Base_PETCT_AutoMasks/
└── [ID_PATIENT]_auto_tumor.nii.gz            (Masque de lésion auto-segmenté)
```

---

## Méthodologie clé

* **Seuillage Adaptatif :**

  * Ne cherche pas un seuil global (fixe).
  * Calcule un seuil local pour chaque cluster détecté, permettant d'isoler des lésions d'intensités variées.

* **Contraintes Anatomiques CT (Optionnel) :**

  * Si activé (`--use-ct`), le script croise les résultats PET avec les unités Hounsfield (**HU**) du scanner.
  * Garantit que la lésion segmentée appartient bien aux tissus mous.
  * Exclut les côtes ou vertèbres adjacentes.

* **Nettoyage Automatique :**

  * Supprime les artefacts par composantes connexes en dessous d'un volume minimal (ex: `< 0.5 mL`).

---

## Exécution

```bash id="u5r8dc"
python src_DL-ML-pCR/auto_segmentation_arsenal/mrAutoSegmentator.py \
    --petct_root ./Base_PETCT \
    --breast_masks ./Base_PETCT_BreastMasks_Expanded \
    --output_root ./Base_PETCT_AutoMasks \
    --use-ct --min-vol 0.5
```
---

## 📊 Évaluation et Contrôle Qualité (QC) des Segmentations

Une fois les prédictions générées par l'orchestrateur **nnU-Net** ou par l'auto-segmentateur déterministe, il est indispensable de mesurer leurs performances et d'en extraire des métriques cliniques fiables.

Ces outils sont regroupés dans le répertoire [`src_DL-ML-pCR/evaluators/`](./src_DL-ML-pCR/evaluators).

---

# 1. Inspecteur Clinique et Multifocalité (`predictions_Inspector.py`)

Ce script est un outil de contrôle qualité (**QC**) orienté oncologie.

Il analyse les masques 3D pour en extraire les biomarqueurs géométriques et lever des alertes sur des cas cliniques complexes (**tumeurs multiples ou absentes**).

## Fonctionnalités clés

* **Filtre Anti-Bruit :**

  * Élimine automatiquement les artefacts de prédiction millimétriques en dessous d'un volume seuil (par défaut `0.005 cm³`).

* **Biomarqueurs Géométriques :**

  * Calcule :

    * le volume exact (en cm³),
    * le diamètre maximal de Feret (en mm),
  * de chaque composante grâce au moteur spatial de **SimpleITK**.

* **Système d'Alerte (Statuts) :**

  * `OK`

    * Une seule lésion primaire détectée.

  * `WARNING_MULTIFOCAL`

    * Détection de plusieurs foyers tumoraux.
    * Nécessite l'attention d'un spécialiste pour identifier la lésion primaire.

  * `WARNING_ZERO_TUMOR`

    * Aucune lésion trouvée.
    * Potentiel cas de pCR ou faux négatif majeur.

---

### Entrée attendue

Un dossier contenant les masques de segmentation (prédictions ou vérité terrain).

```text id="g2v8qk"
nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test/
├── DUKE_001.nii.gz
├── DUKE_002.nii.gz
└── ...
```

---

### Sortie générée

Un fichier texte de rapport détaillé avec le bilan par patient et un résumé statistique global.

```plaintext id="p5n8ws"
evaluations/
└── qc_report.txt
```

---

## ▶️ Exécution

```bash id="e6m3rx"
python src_DL-ML-pCR/evaluators/predictions_Inspector.py \
    -i ./nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test \
    -o ./evaluations/qc_report.txt \
    --min_vol 0.005
```

---

# 2. Le Ring d'Évaluation des Métriques (`segs_fight_ring.py`)

Ce script confronte les prédictions du modèle à la vérité terrain (**Ground Truth**) afin de calculer les scores de performance standards en imagerie médicale.

Il est conçu pour être **"pCR-proof"**, c'est-à-dire qu'il gère les distances mathématiques infinies générées par les masques vides sans faire crasher l'évaluation.

---

## Fonctionnalités clés

* **Métriques MONAI :**

  * Calcule :

    * le coefficient de Dice,
    * la distance de Hausdorff maximale (**HD**),
    * la distance de Hausdorff robuste (**HD95 par défaut**).

* **Gestion Sécurisée des Masques Vides :**

  * Si le patient présente une pCR (**masque GT vide**) et que le modèle prédit également un masque vide :

    * Dice forcé à `1.0`.
    * HD forcée à `0.0`.

  * En cas de discordance (ex: faux positif) :

    * les métriques renvoient `0.0` et `NaN` de manière contrôlée,
    * afin de ne pas corrompre les moyennes globales.

* **Appariement Intelligent :**

  * Compare les fichiers des deux dossiers en se basant sur l'ID du patient, indépendamment des suffixes (`_auto_tumor`, etc.).

---

### Entrées attendues

Deux dossiers contenant les fichiers NIfTI :

* Le dossier de référence (**GT : Ground Truth**).
* Le dossier des prédictions.

```plaintext id="q1d7hc"
nnunet_data/nnUNet_raw/Dataset001_DCE/labelsTs/             <-- (Dossier A : Vérité Terrain)
nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test/ <-- (Dossier B : Prédictions)
```

---

### Sortie générée

Un affichage console des moyennes/écarts-types et un fichier CSV d'analyse fine.

```plaintext id="v4h9pz"
evaluations/
└── metrics_results.csv
```

---

## ▶️ Exécution

```bash id="r8w3fk"
python src_DL-ML-pCR/evaluators/segs_fight_ring.py \
    ./nnunet_data/nnUNet_raw/Dataset001_DCE/labelsTs \
    ./nnunet_data/nnUNet_results/Dataset001_DCE/predictions_test \
    --percentile 95 \
    --csv ./evaluations/metrics_results.csv
```

# 📊 Extraction des Caractéristiques (Radiomique)

Une fois les images et les masques (**vérité terrain** ou **auto-segmentés**) parfaitement alignés et structurés au format **nnU-Net**, cette étape consiste à en extraire des milliers de descripteurs quantitatifs :

* formes,
* intensités,
* textures avancées.

Ces caractéristiques (*features*) alimenteront ensuite les algorithmes de **Machine Learning** pour la prédiction de la réponse **pCR**.

Les scripts d'extraction sont centralisés dans le répertoire [`src_DL-ML-pCR/features_extraction/`](./src_DL-ML-pCR/features_extraction).

---

# 1. Extracteur IRM DCE Multi-Phases & Cinétique (`irm_dce_features_extractor.py`)

Ce script est conçu pour extraire la radiomique des examens **IRM 4D dynamiques** en respectant les normalisations appliquées en amont.

## Fonctionnalités clés

* **Extraction Multi-Zones :**

  * Calcule les caractéristiques sur :

    * la tumeur (**Classe 1**),
    * deux couronnes péritumorales :

      * 0-5 mm,
      * 5-10 mm.
  * Ces couronnes sont construites via une distance Euclidienne exacte (conforme **IBSI**).

* **Pivotement (*Flattening*) :**

  * Condense toutes les phases temporelles d'une patiente :

    * T0,
    * Wash-in,
    * +90s,
    * +180s,
  * sur une seule et unique ligne de données.

* **Delta-Radiomiques Automatiques :**

  * Calcule dynamiquement les deltas absolus et relatifs entre chaque phase post-contraste et la phase native (**Baseline**).
  * Capture ainsi l'évolution de la texture pendant la perfusion du contraste.

* **Isotropisation PyRadiomics :**

  * Force un espacement `[1, 1, 1]` mm en interne.
  * Garantit des matrices de texture robustes et invariantes :

    * GLCM,
    * GLRLM.

---

## 📥 Entrées attendues

Le script lit directement les dossiers d'entraînement préparés pour **nnU-Net**.

* `imagesTr/`

  * Contient les NIfTI de chaque phase :

    * `_0000.nii.gz`,
    * `_0001.nii.gz`,
    * etc.

* `labelsTr/`

  * Contient les masques tumoraux uniques.

---

## 📤 Sorties générées

* `radiomics_results_mri_FLATTENED.csv`

  * Dataset final prêt pour le Machine Learning.

* `radiomics_results_mri_FLATTENED.xlsx`

  * Version Excel du dataset final.

* `run_metadata.json`

  * Fichier de traçabilité des paramètres d'extraction.

---

## ▶️ Exécution

```bash id="z2m7qn"
python src_DL-ML-pCR/features_extraction/irm_dce_features_extractor.py \
    --images_dir ./nnunet_data/nnUNet_raw/Dataset001_DCE/imagesTr \
    --labels_dir ./nnunet_data/nnUNet_raw/Dataset001_DCE/labelsTr \
    --output_dir ./results_radiomics_mri \
    --peri_inner_mm 5.0 \
    --peri_outer_mm 10.0 \
    --bin_width 25.0
```

---

# 2. Extracteur Multimodal TEP/TDM (`pet-ct_features_extractor.py`)

Ce script combine l'imagerie :

* métabolique (**TEP en SUVbw**),
* anatomique (**TDM en unités Hounsfield**),

afin de fournir un tableau de bord quantitatif complet.

---

## Fonctionnalités clés

* **Ré-échantillonnage Isotropique Strict :**

  * Les images sont projetées sur une grille de `2x2x2 mm` en mémoire.
  * Condition requise pour que l'analyse des textures locales sur le PET soit :

    * cliniquement valide,
    * invariante par rotation.

* **Métriques Métaboliques et Anatomiques :**

  * Au-delà des textures PyRadiomics, le script calcule des indices cliniques purs :

    * **SUVpeak** (sur un voisinage `3x3x3`),
    * **MTV** (*Metabolic Tumor Volume*, seuils à `41%` et `2.5`),
    * **TLG** (*Total Lesion Glycolysis*).

* **Asymétrie et Topologie :**

  * Isole :

    * le sein ipsilatéral (malade),
    * le sein controlatéral (sain).
  * Calcule un ratio d'asymétrie d'intensité (**TBR : Tumor to Background Ratio**).

* **Optimisation RAM :**

  * Applique un cropping sur la boîte englobante du masque mammaire.
  * Évite de traiter les voxels vides de l'air ambiant.

---

## 📥 Entrées attendues

* `Dataset002_BreastPETCT/`

  * Dataset nnU-Net contenant :

    * les TEP (`_0000.nii.gz`),
    * les TDM (`_0001.nii.gz`) alignés,
    * la tumeur (`labelsTr`).

* `Base_PETCT_BreastMasks_Expanded/`

  * Dossier contenant les ROI mammaires globales du patient.

---

## 📤 Sorties générées

* `radiomics_features_petct.csv`

  * Dataset final contenant les features TEP et TDM :

    * intratumorales,
    * péritumorales,
    * background.

* `radiomics_features_petct.xlsx`

  * Version Excel du dataset final.

---

## ▶️ Exécution

```bash id="q4w8lc"
python src_DL-ML-pCR/features_extraction/pet-ct_features_extractor.py \
    --nnunet_dir ./nnunet_data/nnUNet_raw/Dataset002_BreastPETCT \
    --breast_dir ./Base_PETCT_BreastMasks_Expanded \
    --output_csv ./results_radiomics_petct/radiomics_features_petct.csv
```

## 🧮 Préparation pour le Machine Learning (ML Pre-works)

Avant de lancer les algorithmes de classification (**Phase 1**), les données radiomiques et cliniques doivent être fusionnées, nettoyées et encodées numériquement.

Les scripts de cette étape sont centralisés dans le répertoire [`src_DL-ML-pCR/M-L_preworks/`](./src_DL-ML-pCR/M-L_preworks).

---

# 1. Nettoyage et Encodage Clinique (`prepare_clinicals.py`)

Les bases de données cliniques brutes (fichiers Excel) contiennent souvent :

* du texte libre,
* des formats de date hétérogènes,
* des conventions de nommage variables.

Ce script agit comme un parseur intelligent pour standardiser ces données et les transformer en variables (*features*) utilisables par les modèles d'IA.

---

## Fonctionnalités clés

* **Ingénierie des caractéristiques (*Feature Engineering*) :**

  * Calcule précisément l'âge au diagnostic (`AgeAtDiagnosis`) à partir :

    * des dates de naissance,
    * de la date du premier diagnostic.

* **Parsing Regex :**

  * Extrait automatiquement les valeurs numériques :

    * des stades TNM (ex: `"T2"` → `2.0`),
    * des grades,
    * du Ki-67 (conversion en pourcentages stricts).

* **Encodage des Biomarqueurs :**

  * Binarise strictement :

    * les récepteurs hormonaux (**ER/PR**).
  * Convertit :

    * les statuts **HER2**,
    * les **nTILs**,
    * en catégories numériques.

* **One-Hot Encoding (OHE) :**

  * Applique un encodage disjonctif (`get_dummies` avec `drop_first=True`) sur les variables catégorielles :

    * Histologie,
    * HER2.
  * Évite le piège de la colinéarité parfaite.
  * Rend les données compatibles avec des modèles sensibles comme :

    * la Régression Logistique,
    * les SVM.

---

### Entrée attendue

* Un fichier Excel ou CSV contenant les données cliniques brutes du patient.

Exemple :

```text
clinicals.xlsx
```

---

### Sortie générée

* `ready_steady_clinicals.xlsx`

Dataset clinique :

* chiffré,
* binarisé,
* prêt pour la concaténation.

---

## ▶️ Exécution

```bash id="k7d3px"
python src_DL-ML-pCR/M-L_preworks/prepare_clinicals.py \
    --input ./data/cliniques_brutes.xlsx \
    --output ./data/ready_steady_clinicals.xlsx
```

---

# 2. Tagger de Vérité Terrain pCR (`PCR_tagger.py`)

Pour que les modèles puissent apprendre (**Apprentissage Supervisé**), chaque ligne de patient dans les fichiers radiomiques doit être étiquetée avec la cible à prédire :

**la réponse pathologique complète (`pcrstatus`).**

Ce script croise de manière sécurisée les fichiers de features avec la base clinique.

---

## Fonctionnalités clés

* **Recherche Robuste (*Sniffer*) :**

  * Détecte automatiquement les séparateurs des fichiers CSV :

    * virgule,
    * point-virgule,
    * tabulation.
  * Prévient les crashs liés aux exports Excel régionaux.

* **Normalisation des Identifiants :**

  * Harmonise les noms de colonnes :

    * `case_id`,
    * `patient_id`,
    * deviennent `subject_id`.
  * Sécurise leur typage en chaînes de caractères (`string`) afin de garantir une jointure (*merge*) parfaite.

* **Exclusion Sécurisée :**

  * Retire automatiquement du dataset d'entraînement les patients dont le statut **pCR** est inconnu ou manquant.

* **Traitement Multimodal :**

  * Peut étiqueter simultanément les fichiers radiomiques issus :

    * de la pipeline IRM,
    * de la pipeline TEP/TDM,
  * en une seule exécution.

---

### Entrées attendues

* `--clinical`

  * Le fichier de référence contenant :

    * les identifiants,
    * le `pcrstatus`.

* `--petct`

  * Les CSV de features radiomiques TEP/TDM générés à l'étape précédente.

* `--mri`

  * Les CSV de features radiomiques IRM générés à l'étape précédente.

---

### Sorties générées

Les fichiers radiomiques mis à jour :

* `*_features_with_pcr.csv`
* `*_features_with_pcr.xlsx`

Chaque patient possède désormais sa cible d'entraînement.

---

## ▶️ Exécution

```bash id="x4q8sm"
python src_DL-ML-pCR/M-L_preworks/PCR_tagger.py \
    --clinical ./data/ready_steady_clinicals.xlsx \
    --mri ./results_radiomics_mri/radiomics_results_mri_FLATTENED.csv \
    --petct ./results_radiomics_petct/radiomics_features_petct.csv
```

# 🤖 Modélisation et Prédiction (Machine Learning)

Cette étape constitue l'aboutissement de la **Phase 1**.

L'objectif est de prédire la réponse pathologique complète (**pCR**) en combinant :

* les données cliniques,
* les données métaboliques (**TEP/TDM**),
* les données dynamiques (**IRM**).

Pour contrer le fléau de la dimensionnalité (**beaucoup de variables radiomiques pour des cohortes souvent réduites**), le pipeline repose sur :

* un filtrage drastique,
* une validation extrêmement stricte.

Les scripts se trouvent dans le répertoire [`src_DL-ML-pCR/M-L/`](./src_DL-ML-pCR/M-L).

---

# 1. Entraînement et Validation Croisée Imbriquée (`NestedCV.py`)

Ce script est le cœur prédictif du projet.

Il teste, compare et valide plusieurs familles d'algorithmes (**linéaires et non-linéaires**) en s'assurant qu'aucune fuite de données (**Data Leakage**) ne vienne fausser les résultats.

---

## Méthodologie "En Entonnoir" (*Feature Selection*)

1. **Pré-filtrage :**

   * Suppression des variables à variance nulle.
   * Élimination de la redondance :

     * filtre de corrélation > `0.95`.

2. **Réduction douce (Filtre Univarié) :**

   * Utilisation de `SelectKBest` (**Mutual Information**) pour isoler les **100 à 300** variables les plus prometteuses.

3. **Sélection Multivariée :**

   * Un dernier filtre algorithmique sélectionne la signature radiomique finale optimale :

     * `ElasticNet` pour les modèles linéaires,
     * `ExtraTrees` / `LightGBM` pour les modèles basés sur les arbres.

---

## Nested Cross-Validation

Une double boucle de validation croisée est utilisée :

* **Outer 3-fold**

  * Évaluation de la généralisation.

* **Inner 3-fold**

  * Optimisation des hyperparamètres avec `GridSearchCV`.

Cette stratégie garantit une estimation robuste et honnête des performances.

---

## Algorithmes comparés

Les modèles évalués sont :

* Régression Logistique (**LR**),
* SVM,
* Random Forest (**RF**),
* ExtraTrees (**ET**),
* HistGradientBoosting (**HGB**),
* KNN,
* Perceptron Multicouche (**MLP**).

---

## 📥 Entrées attendues

* Les fichiers de features générés à l'étape précédente :

  * Clinique,
  * IRM,
  * TEP/TDM.

Le script teste automatiquement toutes les combinaisons possibles :

* modèles unimodaux,
* fusions multimodales.

---

## 📤 Sorties générées

* Dossier `_modality_outputs/`

Contient :

* les modèles finaux entraînés (`.joblib`),
* les listes des variables retenues,
* les métriques détaillées par fold,
* un résumé global :

  * `summary_metrics.csv`.

---

## ▶️ Exécution

```bash id="q7w2nb"
python src_DL-ML-pCR/M-L/NestedCV.py
```

---

# 2. Génération des Visualisations (`results_plotter.py`)

Une fois l'entraînement terminé, ce script lit les tableaux de résultats et produit des graphiques clairs, esthétiques (**style Seaborn professionnel**) et prêts à être intégrés dans des publications ou des présentations.

---

## Visualisations générées

### Comparaison des Modèles

* Barplot global des performances (**ROC-AUC moyen**).
* Permet d'identifier immédiatement la meilleure combinaison de modalités :

Exemple :

* Clinique + IRM + TEP/TDM.

---

### Matrices de Confusion

* Heatmaps cumulées sur les folds pour les **5 meilleurs modèles**.
* Analyse de la répartition :

  * Faux Positifs,
  * Faux Négatifs.

---

### Importance des Variables

* Barplot horizontal identifiant le **Top 15** des caractéristiques les plus influentes :

  * radiomiques,
  * cliniques.

---

## 📥 Entrée attendue

* Le dossier `_modality_outputs/` généré par `NestedCV.py`.

---

## 📤 Sortie générée

* Dossier `_plots/`

Contient les graphiques au format :

* PNG,
* haute résolution (**300 dpi**).

---

## ▶️ Exécution

```bash id="j4p8sm"
python src_DL-ML-pCR/M-L/results_plotter.py \
    -i ./_modality_outputs \
    -o ./_plots
```

---

# 🚀 Phase 2 : Approche 100% Neuronale (Modèle WB CERBERUS)

La radiomique classique (**Phase 1**) repose sur une extraction manuelle et mathématique des caractéristiques (*hand-crafted features*) à partir des segmentations.

Bien que robuste, cette méthode comporte des limites.

> Et si l'on laissait un réseau de neurones découvrir lui-même ses propres descripteurs spatio-temporels directement depuis l'imagerie brute ?

C'est tout l'enjeu de la **Phase 2**, actuellement en cours d'élaboration.

Située dans le répertoire :

```text
src_DL-DL-pCR/
```

cette seconde approche s'affranchit du pipeline d'extraction radiomique pour proposer une architecture **Deep Learning multimodale de bout en bout (End-to-End)** :

## Modèle WB CERBERUS

Une architecture neuronale capable d'apprendre directement les représentations utiles depuis les données d'imagerie.

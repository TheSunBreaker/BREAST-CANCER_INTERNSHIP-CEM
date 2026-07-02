#!/usr/bin/env python3
"""
Niveau 0 : Anonymisation stricte des DICOMs bruts.
Utilise un CSV contenant IPP -> ID_PAT_LOCAL.
Préserve le Poids, la Taille, le Sexe et la Géométrie.
"""

import os
import csv
import pydicom
from pathlib import Path
from tqdm import tqdm

def load_anonymization_mapping(csv_path: str) -> dict:
    """
    Charge le dictionnaire d'anonymisation depuis le fichier CSV clinique.
    Retourne un dict { 'IPP': 'ID_PAT_LOCAL' }
    """
    mapping = {}
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        
        reader = csv.DictReader(f, delimiter=';') 
        for row in reader:
            ipp = str(row.get('IPP', '')).strip()
            id_local = str(row.get('ID_PAT_LOCAL', '')).strip()
            if ipp and id_local:
                mapping[ipp] = id_local
    
    print(f"[INFO] {len(mapping)} correspondances IPP -> ID_LOCAL chargées depuis le CSV.")
    return mapping

def anonymize_dicom_dataset(src_dir: str, dst_dir: str, mapping: dict):
    """
    Parcourt le dossier source, anonymise les fichiers correspondants au mapping,
    et les sauvegarde dans le dossier de destination avec la même arborescence.
    """
    # Lister tous les fichiers pour la barre de progression
    all_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    print(f"\n--- DÉBUT DE L'ANONYMISATION ({len(all_files)} fichiers trouvés) ---")
    
    compteur_succes = 0
    compteur_ignores = 0
    
    for file_path in tqdm(all_files, desc="Anonymisation"):
        try:
            # On charge tout le fichier en mémoire cette fois, car on va le modifier
            ds = pydicom.dcmread(file_path, force=True)
            
            # Vérification : Est-ce un DICOM avec un IPP ?
            if not hasattr(ds, 'PatientID'):
                continue
                
            ipp = str(ds.PatientID).strip()
            
            # Vérification de l'IPP dans notre CSV
            if ipp not in mapping:
                # Sécurité maximale : Si le patient n'est pas dans le CSV, on ne le copie PAS.
                compteur_ignores += 1
                continue
                
            id_local = mapping[ipp]
            
            # ==========================================
            # APPLICATION DES RÈGLES D'ANONYMISATION
            # ==========================================
            
            # 1. Identifiants directs
            ds.PatientID = id_local
            if hasattr(ds, 'PatientName'):
                ds.PatientName = id_local
                
            # 2. Date de naissance (Garder Année+Mois, forcer le jour à 01)
            if hasattr(ds, 'PatientBirthDate') and ds.PatientBirthDate:
                dob = ds.PatientBirthDate
                if len(dob) == 8: # Format standard YYYYMMDD
                    ds.PatientBirthDate = dob[:6] + "01"
                else:
                    ds.PatientBirthDate = "" # En cas de format exotique, on supprime
                    
            # 3. Écrasement des données institutionnelles et personnelles
            tags_a_effacer = [
                'InstitutionName', 'InstitutionAddress', 'ReferringPhysicianName',
                'PerformingPhysicianName', 'StudyID', 'AccessionNumber',
                'OtherPatientIDs', 'PatientAddress', 'PatientTelephoneNumbers'
            ]
            
            for tag in tags_a_effacer:
                if hasattr(ds, tag):
                    setattr(ds, tag, "") # On vide la balise sans la supprimer totalement pour la stabilité
                    
            # REMARQUE : On ne touche PAS à PatientWeight, PatientSex, SeriesTime, etc.
            
            # ==========================================
            # SAUVEGARDE DU NOUVEAU FICHIER
            # ==========================================
            
            # On recrée l'arborescence exacte dans le dossier sécurisé
            rel_path = os.path.relpath(file_path, src_dir)
            out_path = os.path.join(dst_dir, rel_path)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ds.save_as(out_path)
            compteur_succes += 1
            
        except Exception:
            # Si le fichier n'est pas un DICOM valide, on l'ignore silencieusement
            continue

    print("\n" + "="*40)
    print(f"ANONYMISATION TERMINÉE !")
    print(f" -> Fichiers nettoyés et sécurisés : {compteur_succes}")
    print(f" -> Fichiers ignorés (IPP inconnu)  : {compteur_ignores}")
    print("="*40)

if __name__ == "__main__":
    CSV_CLINIQUE = "./data_clinique.csv"
    DOSSIER_ZONE_ROUGE = "./data_hopital_brut"      # Les données contenant les vrais noms
    DOSSIER_ZONE_VERTE = "./data_hopital_safe"      # Les données prêtes pour l'ingestion NIfTI
    
    mapping_anonymisation = load_anonymization_mapping(CSV_CLINIQUE)
    anonymize_dicom_dataset(DOSSIER_ZONE_ROUGE, DOSSIER_ZONE_VERTE, mapping_anonymisation)

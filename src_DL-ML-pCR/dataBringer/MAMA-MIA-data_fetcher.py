import os
import shutil
import synapseclient
import synapseutils
import urllib3

# ==========================================
# PARAMÈTRES DE CONFIGURATION
# ==========================================

# Mon token d'accès Synapse
TOKEN_SYNAPSE = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjE1NzYxNSwiaWF0IjoxNzc2MTU3NjE1LCJqdGkiOiIzNTU2MSIsInN1YiI6IjM1ODQyODQifQ.Ki62gZeEVvsTzIIxJZMk3m3qHROj-Vzc6hdAso-z6SHRfqx4pIEuiMxHQxZdV3twTRAuhi9A5gPjhuAnxn2P_lrONCoBL5ADYN5RMCvgaa9TSLts9d6WdWL2qKr1lE5LVCPPxLhd_I9xX8Ag8LQgil03nRjxCknivsoSRfcSin-Ep9u_ExTvK4R2G7oX7Bb1xb820wFxWjC5dkQr_dfgmavieDm1-a5Yqfgh9yEqs8f4oeX6HTUd6B6363TyoHa81zwtUNeHxV1EaluprkcyApfRSp7c_qZK_cfb8EozunDMY6flrMaIkwOPiQY0dVUBlA7MPopZ0ctnBXn5CJMMww"

# Dossier parent contenant les dossiers de chaque patient
ID_DOSSIER_IMAGES = "syn64871114"  
# Dossier contenant tous les fichiers masques .nii.gz mélangés
ID_DOSSIER_MASQUES = "syn64871175" 

# Destination finale des données formatées
DOSSIER_SORTIE = "./mes_dossiers_nnunet"

# Désactivation SSL pour éviter les erreurs de certificats sur certains réseaux
DESACTIVER_VERIFICATION_SSL = True

def main():
    print("=== DÉBUT DU TÉLÉCHARGEMENT ET FORMATAGE OPTIMISÉ ===")

    # ---------------------------------------------------------
    # 1. INITIALISATION DE LA CONNEXION
    # ---------------------------------------------------------
    if DESACTIVER_VERIFICATION_SSL:
        print("[INFO] Mode SSL non sécurisé activé.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        syn = synapseclient.Synapse(skip_checks=True)
        syn._requests_session.verify = False
    else:
        syn = synapseclient.Synapse()

    print("\nTentative de connexion à Synapse...")
    syn.login(authToken=TOKEN_SYNAPSE)
    print("Authentification réussie.")

    # ---------------------------------------------------------
    # 2. PRÉPARATION DU STOCKAGE TEMPORAIRE
    # ---------------------------------------------------------
    # Je crée un dossier temporaire pour stocker tous les masques d'un seul coup
    # Cela évite de faire une requête réseau par patient pour chercher son masque
    dossier_temp_masques = os.path.join(DOSSIER_SORTIE, "_temp_masks")
    os.makedirs(dossier_temp_masques, exist_ok=True)

    # ---------------------------------------------------------
    # 3. ÉTAPE OPTIMISÉE : TÉLÉCHARGEMENT MASQUES EN BATCH
    # ---------------------------------------------------------
    # Je récupère TOUT le dossier des masques en une seule commande sync
    # C'est beaucoup plus rapide car c'est parallélisé par le client Synapse
    print("\n[ÉTAPE 1/3] Récupération globale des masques en cours...")
    synapseutils.syncFromSynapse(syn, ID_DOSSIER_MASQUES, path=dossier_temp_masques)
    
    # J'indexe les fichiers téléchargés pour vérifier leur présence localement plus tard
    masques_disponibles = os.listdir(dossier_temp_masques)
    print(f"Indexation locale terminée : {len(masques_disponibles)} masques trouvés.")

    # ---------------------------------------------------------
    # 4. RÉCUPÉRATION DE LA STRUCTURE DES PATIENTS
    # ---------------------------------------------------------
    print("\n[ÉTAPE 2/3] Analyse des dossiers patients sur Synapse...")
    dossiers_patients = list(syn.getChildren(parent=ID_DOSSIER_IMAGES, includeTypes=['folder']))
    total_patients = len(dossiers_patients)
    print(f"Total : {total_patients} patients à traiter.")

    compteur_ok = 0
    compteur_skip = 0

    # ---------------------------------------------------------
    # 5. BOUCLE DE TRI ET TÉLÉCHARGEMENT DES IMAGES
    # ---------------------------------------------------------
    print("\n[ÉTAPE 3/3] Organisation des données par patient...")
    
    for i, dossier in enumerate(dossiers_patients):
        id_patient = dossier['name']
        nom_masque_cible = f"{id_patient}.nii.gz"
        chemin_source_masque = os.path.join(dossier_temp_masques, nom_masque_cible)

        print(f"\n--- Patient [{i+1}/{total_patients}] : {id_patient} ---")

        # Je vérifie si le masque existe dans mon dossier temporaire
        if nom_masque_cible not in masques_disponibles:
            print(f"[ALERTE] Masque manquant pour {id_patient}. Passage au suivant.")
            compteur_skip += 1
            continue

        # Je prépare l'arborescence finale demandée par nnU-Net
        chemin_patient = os.path.join(DOSSIER_SORTIE, id_patient)
        chemin_imgs = os.path.join(chemin_patient, "imgs")
        chemin_mask = os.path.join(chemin_patient, "mask")
        
        os.makedirs(chemin_imgs, exist_ok=True)
        os.makedirs(chemin_mask, exist_ok=True)

        # OPTIMISATION : Je déplace le masque localement (opération instantanée)
        # au lieu de le télécharger à nouveau depuis Synapse
        shutil.move(chemin_source_masque, os.path.join(chemin_mask, nom_masque_cible))
        print(f"-> Masque déplacé avec succès.")

        # Je télécharge les images sources (plusieurs volumes IRM) dans le dossier imgs
        print(f"-> Téléchargement des images DICOM/NIfTI...")
        synapseutils.syncFromSynapse(syn, dossier['id'], path=chemin_imgs)
        
        compteur_ok += 1

    # ---------------------------------------------------------
    # 6. NETTOYAGE ET RÉSUMÉ
    # ---------------------------------------------------------
    # Je supprime le dossier temporaire s'il reste des masques inutilisés
    if os.path.exists(dossier_temp_masques):
        print("\nNettoyage des fichiers temporaires...")
        shutil.rmtree(dossier_temp_masques)

    print("\n" + "="*45)
    print("BILAN DU TRAITEMENT")
    print(f"Patients correctement formatés : {compteur_ok}")
    print(f"Patients ignorés (pas de masque) : {compteur_skip}")
    print(f"Dossier final : {os.path.abspath(DOSSIER_SORTIE)}")
    print("="*45)

if __name__ == "__main__":
    main()

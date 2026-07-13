import synapseclient
import urllib3
import synapseutils

# 1. Connexion

# a. On masque les avertissements rouges dans la console liés au contournement SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# b. On initialise Synapse en lui disant de NE PAS tester la connexion tout de suite
syn = synapseclient.Synapse(skip_checks=True)

# c. On ordonne à la session interne de ne plus vérifier les certificats SSL
syn._requests_session.verify = False


syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjE1NzYxNSwiaWF0IjoxNzc2MTU3NjE1LCJqdGkiOiIzNTU2MSIsInN1YiI6IjM1ODQyODQifQ.Ki62gZeEVvsTzIIxJZMk3m3qHROj-Vzc6hdAso-z6SHRfqx4pIEuiMxHQxZdV3twTRAuhi9A5gPjhuAnxn2P_lrONCoBL5ADYN5RMCvgaa9TSLts9d6WdWL2qKr1lE5LVCPPxLhd_I9xX8Ag8LQgil03nRjxCknivsoSRfcSin-Ep9u_ExTvK4R2G7oX7Bb1xb820wFxWjC5dkQr_dfgmavieDm1-a5Yqfgh9yEqs8f4oeX6HTUd6B6363TyoHa81zwtUNeHxV1EaluprkcyApfRSp7c_qZK_cfb8EozunDMY6flrMaIkwOPiQY0dVUBlA7MPopZ0ctnBXn5CJMMww")

# 2. L'identifiant du dossier principal contenant les dossiers patients/images
dossier_parent_id = "syn64871114" 

# 3. Lister uniquement les DOSSIERS cette fois-ci
dossiers_patients = syn.getChildren(parent=dossier_parent_id, includeTypes=['folder'])

# 4. Boucle de téléchargement
nb_dossiers_a_telecharger = 5

print("Début du téléchargement...")

for index, dossier in enumerate(dossiers_patients):
    if index >= nb_dossiers_a_telecharger:
        break
        
    print(f"\n[{index + 1}/{nb_dossiers_a_telecharger}] Traitement du dossier: {dossier['name']} (ID: {dossier['id']})")
    
    # Création dynamique du chemin de destination pour séparer chaque dossier patient
    chemin_destination = f"./mes_{nb_dossiers_a_telecharger}_dossiers_images/{dossier['name']}"
    
    # syncFromSynapse est la fonction magique qui télécharge tout le contenu d'un dossier
    synapseutils.syncFromSynapse(syn, dossier['id'], path=chemin_destination)

print("\nTous les dossiers ont été téléchargés avec succès !")
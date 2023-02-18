# **************************************************************************
# INF7370-Hiver 2023
# Travail pratique 1
# ===========================================================================
# ===========================================================================
# GBEGAN HUGUES
# GBEH24279505
# ===========================================================================
# ===========================================================================


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================
# ===============VARIABLES==================
# ==========================================

# l'emplacement des données sur le disque
# Note: Il faut placer le dossier "donnees"  contenant les 8 fichiers .csv dans le même endroit que les fichiers de code
data_path = "donnees/"

# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================
features = pd.read_csv(data_path + "features.csv", skipinitialspace=True)
print('info', features.info())

# ==========================================
# ==========PRE TRAITEMENT==================
# ==========================================

features.ville = pd.Categorical(features.ville)
features['ville'] = features.ville.cat.codes

features.zone = pd.Categorical(features.zone)
features['zone'] = features.zone.cat.codes

# Remplacement des valeurs manquantes
'''
nb_restaurants_zone
ecart_type_etoiles
nb_avis_favorables
nb_avis_defavorables
ratio_avis_favorables
ratio_avis_defavorables
nb_avis_favorables_mention        
nb_avis_defavorables_mention      
nb_avis_favorables_elites         
nb_avis_defavorables_elites       
nb_conseils                       
nb_conseils_compliment           
nb_conseils_elites                
nb_checkin                        
moyenne_checkin                  
ecart_type_checkin
prix              
'''
# Remplacement des valeurs manquantes par 0
champs_remplacement_0 = ['nb_restaurants_zone', 'nb_avis_favorables', 'nb_avis_defavorables', 'ratio_avis_favorables', 'ratio_avis_defavorables', 'nb_avis_favorables_mention', 'nb_avis_defavorables_mention', 'nb_avis_favorables_elites', 'nb_avis_defavorables_elites', 'nb_conseils', 'nb_conseils_compliment', 'nb_conseils_elites', 'nb_checkin']
features[champs_remplacement_0] = features[champs_remplacement_0].fillna(value=0)

# Remplacement des valeurs manquantes par la moyenne
champs_remplacement_moyenne = ['ecart_type_etoiles', 'moyenne_checkin', 'ecart_type_checkin']
features[champs_remplacement_moyenne] = features[champs_remplacement_moyenne].fillna(value=features[champs_remplacement_moyenne].mean())

# Remplacement des valeurs manquantes des niveaux de prix par inconnu
features[['prix']] = features[['prix']].fillna(value='inconnu')

print('f',features)
print(features.isnull().sum())

# -----------------------------------------------------------
# Sauvegarder l'ensemble de données dans un fichier csv afin d'être utilisé dans l'étape suivante
print('preprocessing file created .......')
features.to_csv("donnees/preprocess.csv", index=False)

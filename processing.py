# **************************************************************************
# INF7370-Hiver 2023
# Travail pratique 1
# ===========================================================================
# ===========================================================================
# Indiquer votre nom ici
# ===========================================================================
# ===========================================================================

# ===========================================================================
# Le but de ce travail est de classifier les restaurants en 2 états (Fermeture définitive / Ouvert)
#
# Ce fichier consiste la deuxième étape du travail -> pré-traitement du dataset issu de la première tache.
# Dans ce fichier code vous devez  traiter l’ensemble de données préparées dans la  première étape afin de
# les rendre prêtes pour la consommation par les modèles d’apprentissage dans l'étape suivante.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres librairies dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque librairie
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# votre code ici:

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================
# ===============VARIABLES==================
# ==========================================

# l'emplacement des données sur le disque
# Note: Il faut placer le dossier "donnees"  contenant les 8 fichiers .csv dans le même endroit que les fichiers de code
data_path = "donnees/"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres variables globales dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque variable
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# votre code ici:

# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# charger en mémoire les features préparées dans la première étape
features = pd.read_csv(data_path + "features.csv")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Remplacez les valeurs manquantes par de propres valeurs
#
# Vous devez identifier tous les features qui manquent de valeurs ou
# qui ont des valeurs erronées dans le fichier "features.csv" préparé dans la première etape,
# puis vous devez remplacez ces valeurs manquantes ou erronées par de propres valeurs.
# Les valeurs manquantes peuvent être remplacées par des 0, ou remplacées par la moyenne ou le mode.
# La méthode choisie doit dépendre de la nature de chaque feature.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 2
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Catégorisation des features: ville et zone
#
# Pour les deux attributs "ville" et "zone" avec des valeurs symboliques,
# il faut effectuer une transformation de ces symboles.
# Vous pouvez utiliser la fonction Categorical (de la librairie Pandas).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:




# -----------------------------------------------------------
# Elimination de la colonne identifiante (ID): restaurant_id
print("------------------------")
print("Elimination de la colonne restaurant_id")
features = features.drop('restaurant_id', axis=1)
print("")

# -----------------------------------------------------------
# Sauvegarder l'ensemble de données pré-traitées dans un fichier csv afin d'être utilisées dans l'étape suivante
features.to_csv("donnees/features_finaux.csv", index=False)

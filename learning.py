# **************************************************************************
# INF7370-Hiver 2023
# Travail pratique 1
# ===========================================================================
# ===========================================================================
# GBEGAN HUGUES
# GBEH24279505
# ===========================================================================
# ===========================================================================

# ===========================================================================
# Le but de ce travail est de classifier les restaurants en 2 états (Fermeture définitive / Ouvert)
#
# Ce fichier consiste la troisième étape du travail -> entrainement des modèles de classification
# Dans ce fichier code, vous devez entrainer 5 modèles de classification sur les données préparées dans l'étape précédente.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd
# la librairie pour normalizer les données par Z-Score
from sklearn.preprocessing import StandardScaler
# la librairie pour diviser les données en deux lots (entrainement et test)
from sklearn.model_selection import train_test_split

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres librairies dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque librairie
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


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

# Votre code ici:


# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# Charger en mémoire les features préparées dans la deuxième étape (pré-traités)
features = pd.read_csv(data_path + "features_finaux.csv")

# ==========================================
# INITIALIZATION DES DONNÉES ET DES ÉTIQUETTES
# ==========================================

# Initialisation des données et des étiquettes
x = features.copy() # "x" contient l'ensemble des données d'entrainement
y = x["ferme"]      # "y" contient les étiquettes des enregistrements dans "x"

# Elimination de la colonne classe (ferme) des features
x = x.drop('ferme', axis=1)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  - Normaliser les données en utilisant Z-score (StandardScaler dans Scikit-learn)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
sc = StandardScaler()
x = sc.fit_transform(x)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 2
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Divisez les données en deux lots (entrainement et test)
# (indiquer dans votre rapport le pourcentage des données de test que vous avez utilisé)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 3
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Entrainez 5 modèles de classification sur l'ensemble de données normalisées (avec tous les features)
#   1 - Arbre de decision
#   2 - Forêt d’arbres décisionnels (Random Forest)
#   3 - Classification bayésienne naïve
#   4 - Bagging
#   5 - AdaBoost
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 4
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Afficher les resultats sur les données test de chaque algorithm entrainé avec tous les features
#   1- Le taux des vrais positifs (TP Rate) – de la classe Restaurants fermés définitivement.
#   2- Le taux des faux positifs (FP Rate) – de la classe Restaurants fermés définitivement.
#   3- F-measure de la classe Restaurants fermés définitivement.
#   4- La surface sous la courbe ROC (AUC).
#   5- La matrice de confusion.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 5
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Selectionnez les tops 10 features
#
# Vous devez identifier les 10 meilleurs features en utilisant la mesure du Gain d’information (Mutual Info dans scikit-learn).
# Afficher les 10 meilleurs features dans un tableau (par ordre croissant selon le score obtenu par le Gain d'information).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 6
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Entrainez 5 modèles de classification sur l'ensemble de données normalisées avec seulement les top 10 features selectionnés.
#   1 - Arbre de decision
#   2 - Forêt d’arbres décisionnels (Random Forest)
#   3 - Classification bayésienne naïve
#   4 - Bagging
#   5 - AdaBoost
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 7
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Afficher les resultats sur les données test de chaque algorithm entrainé avec les top 10 features
#   1- Le taux des vrais positifs (TP Rate) – de la classe Restaurants fermés définitivement.
#   2- Le taux des faux positifs (FP Rate) – de la classe Restaurants fermés définitivement.
#   3- F-measure de la classe Restaurants fermés définitivement.
#   4- La surface sous la courbe ROC (AUC).
#   5- La matrice de confusion.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Votre code ici:


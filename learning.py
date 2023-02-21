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
import numpy as np
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
def ececuter_entrainement(x_test, y_test):
    y_pred_dtc = dtc.predict(x_test)
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    # Matrice de confusion
    cm_dtc = confusion_matrix(y_test, y_pred_dtc)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_dtc).ravel()
    # true positive rate
    TPR = TP/(TP+FN)
    # false positive rate
    FPR = FP/(FP+TN)
    # F-measure
    f1_dtc = f1_score(y_test, y_pred_dtc, average='macro')
    # AUC
    roc_dtc = roc_auc_score(y_test, y_pred_dtc)

    return cm_dtc, TPR, FPR, f1_dtc, roc_dtc


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
# 1 - Arbre de decision
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# 2 - Forêt d’arbres décisionnels (Random Forest)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# 3 - Classification bayésienne naïve
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# 4 - Bagging
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier()
bagging.fit(x_train, y_train)

# 5 - AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(x_train, y_train)



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
# 1 - Arbre de decision
cm_dtc, TPR, FPR, f1_dtc, roc_dtc = ececuter_entrainement(x_test, y_test)
print("1 - Arbre de decision ----- Matrice de confusion,  TP Rate, FP Rate, F-measure, AUC: ", cm_dtc, TPR, FPR, f1_dtc, roc_dtc)

# 2 - Forêt d’arbres décisionnels (Random Forest)
cm_rfc, TPR, FPR, f1_rfc, roc_rfc = ececuter_entrainement(x_test, y_test)
print("2 - Random Forest ------ Matrice de confusion,  TP Rate, FP Rate, F-measure, AUC: ", cm_rfc, TPR, FPR, f1_rfc, roc_rfc)

# 3 - Classification bayésienne naïve
cm_gnb, TPR, FPR, f1_gnb, roc_gnb = ececuter_entrainement(x_test, y_test)
print("3 - Classification bayésienne naïve ------ Matrice de confusion,  TP Rate, FP Rate, F-measure, AUC: ", cm_gnb, TPR, FPR, f1_gnb, roc_gnb)

# 4 - Bagging
cm_bagging, TPR, FPR, f1_bagging, roc_bagging = ececuter_entrainement(x_test, y_test)
print("4 - Bagging ------ Matrice de confusion,  TP Rate, FP Rate, F-measure, AUC: ", cm_bagging, TPR, FPR, f1_bagging, roc_bagging)

# 5 - AdaBoost
cm_adaboost, TPR, FPR, f1_adaboost, roc_adaboost = ececuter_entrainement(x_test, y_test)
print("5 - AdaBoost --------- Matrice de confusion,  TP Rate, FP Rate, F-measure, AUC: ", cm_adaboost, TPR, FPR, f1_adaboost, roc_adaboost)



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


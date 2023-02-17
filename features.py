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
# Ce fichier consiste la première étape du travail -> Calcul des attributs (features engineering)
# Dans ce fichier code vous devez bâtir l'ensemble de données  avec les features nécessaires pour l'entrainement des modèles de classification.
#
# Les données (8 fichiers csv):
# ------------------------------------------------
#  -utilisateurs  : Contient la liste des utilisateurs avec quelques détails.
#  -avis          : Contient les informations qui se rattachent aux avis rédigés pour chaque restaurant avec le nombre d'étoiles accordé.
#  -conseils      : Contient les informations qui se rattachent aux conseils (tips) rédigés par les utilisateurs sur les restaurants.
#  -checkin       : Contient les dates et heures de visite par restaurant.
#  -restaurants   : Contient la liste des restaurants avec quelques détails.
#  -horaires      : Contiens les heures d'ouverture de chaque restaurant pour les sept jours de la semaine.
#  -services      : Contiens la liste des services offerts par chaque restaurant.
#  -categories    : Contiens la liste des catégories des restaurants.
#
# Source des données: https://www.yelp.com/dataset
# Note: Les données étaient pré-traitées pour inclure seulement les informations nécessaires pour ce TP.
# ------------------------------------------------

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import numpy as np
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

# Les jours de la semaine
days = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']


# Fonction pour convertir les horaires au format datetime
def convert_time(time_str):
    if pd.isna(time_str):
        return None
    else:
        return pd.to_datetime(time_str, format='%H:%M', errors='coerce').time()



# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# Lire les 8 tables csv et les remplir dans 8 objets de type Dataframe
# Ces 8 Dataframes doivent être utilisés pour calculer les features
utilisateurs = pd.read_csv(data_path + "utilisateurs.csv", skipinitialspace=True)
avis = pd.read_csv(data_path + "avis.csv", skipinitialspace=True)
conseils = pd.read_csv(data_path + "conseils.csv", skipinitialspace=True)
checkin = pd.read_csv(data_path + "checkin.csv", skipinitialspace=True)
restaurants = pd.read_csv(data_path + "restaurants.csv", skipinitialspace=True)
horaires = pd.read_csv(data_path + "horaires.csv", skipinitialspace=True)
services = pd.read_csv(data_path + "services.csv", skipinitialspace=True)
categories = pd.read_csv(data_path + "categories.csv", skipinitialspace=True)

# Imprimer la taille de chaque table de données
# print("Taille des données:")
# print("------------------")
# print("utilisateurs:\t", len(utilisateurs))
# print("avis:\t\t\t", len(avis))
# print("conseils:\t\t", len(conseils))
# print("checkin:\t\t", len(checkin))
# print("restaurants:\t", len(restaurants))
# print("horaires:\t\t", len(horaires))
# print("services:\t\t", len(services))
# print("categories:\t\t", len(categories))
# print("------------------")

# ==========================================
# ==========CALCUL DES FEATURES=============
# ==========================================

# Initialisation du Dataframe "features" qui va contenir l'ensemble de données d'entrainement
# -----------------------------------------------------
# restaurant_id:  C'est l'identifiant (ID) du restaurant
# ferme: C'est la classe (Fermeture définitive : 1 , ouvert: 0)
# Les 3 premiers features sont déjà chargés: moyenne_etoiles, ville et zone
features = restaurants[['restaurant_id', 'moyenne_etoiles', 'ville', 'zone', 'ferme']].copy()

# Extraire l'année à partir des dates
# La colonne "annee" sera utilisée dans le calcul de certain features
# -----------------------------------------------------
checkin["date"] = pd.to_datetime(checkin["date"], format='%Y-%m-%d')
checkin['annee'] = checkin.date.dt.year

avis["date"] = pd.to_datetime(avis["date"], format='%Y-%m-%d')
avis['annee'] = avis.date.dt.year

conseils["date"] = pd.to_datetime(conseils["date"], format='%Y-%m-%d')
conseils['annee'] = conseils.date.dt.year

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Calculez chacun des 34 features suivants.
# Suivez la description de chaque feature afin de bien l'estimer
# Les features doivent être ajoutés au dataframe "features"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# -----------------------------------------------------------
# 4) nb_restaurants_zone
# Le nombre de restaurants dans la zone associée au restaurant en question.
# -----------------------------------------------------------
nb_restaurants_zone = restaurants[['restaurant_id', 'zone']].groupby('zone').size().reset_index(name='nb_restaurants_zone')
features = features.merge(nb_restaurants_zone, on='zone', how='left')

# -----------------------------------------------------------
# 5) zone_categories_intersection
# Le nombre de restaurants dans la même zone qui partagent au moins une catégorie avec le restaurant en question.
# -----------------------------------------------------------
restaurants_categories_intersection = restaurants[['restaurant_id', 'zone']].merge(categories, on='restaurant_id', how='left')
zone_categories_intersection_nb = restaurants_categories_intersection.groupby(['zone', 'categorie']).size().reset_index(name='zone_categories_intersection')
zone_categories_intersection = pd.merge(restaurants_categories_intersection, zone_categories_intersection_nb, on='zone', how='left')
zone_categories_intersection = zone_categories_intersection.groupby('restaurant_id').sum().reset_index()
features = features.merge(zone_categories_intersection, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 6) ville_categories_intersection
# Le nombre de restaurants dans la même ville qui partagent au moins une catégorie avec le restaurant en question.
# -----------------------------------------------------------
restaurants_categories_intersection = restaurants[['restaurant_id', 'ville']].merge(categories, on='restaurant_id', how='left')
ville_categories_intersection_nb = restaurants_categories_intersection.groupby(['ville', 'categorie']).size().reset_index(name='ville_categories_intersection')
ville_categories_intersection = pd.merge(restaurants_categories_intersection, ville_categories_intersection_nb, on='ville', how='left')
ville_categories_intersection = ville_categories_intersection.groupby('restaurant_id').sum().reset_index()
features = features.merge(ville_categories_intersection, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 7) nb_restaurant_meme_annee
# Le nombre de restaurants qui sont ouverts leurs portes dans la même année que le restaurant en question.
# Ici, on considère que la première année d'un restaurant correspond à l'année de la première publication d'un avis sur ce restaurant sur Yelp.
# -----------------------------------------------------------
annee_ouverture_restaurant = avis[['restaurant_id', 'annee']].groupby('restaurant_id').annee.min().reset_index(name='annee')
nb_restaurant_meme_annee = annee_ouverture_restaurant.groupby('annee').size().reset_index(name='nb_restaurant_meme_annee')
table_fusion = annee_ouverture_restaurant.merge(nb_restaurant_meme_annee, on='annee', how='left')
restaurant_meme_annee = table_fusion[['restaurant_id', 'nb_restaurant_meme_annee']]
features = features.merge(restaurant_meme_annee, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 8) ecart_type_etoiles
# L'écart type de la moyenne des étoiles par année.
# Il faut estimer la moyenne des étoiles par années. Puis, calculer l'écart type sur ces valeurs.
# -----------------------------------------------------------
moyenne_etoiles_annee = avis.groupby(['restaurant_id','annee']).etoiles.mean().reset_index(name='moyenne_etoiles_annee')
ecart_type_etoiles = moyenne_etoiles_annee.groupby('restaurant_id').moyenne_etoiles_annee.std().reset_index(name='ecart_type_etoiles')
features = features.merge(ecart_type_etoiles, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 9) tendance_etoiles
# La différence entre la moyenne des étoiles de la dernière année et la moyenne des étoiles de la première année d'un restaurant.
# Ici, on considère la première année d'un restaurant correspond à l'année de la première publication d'un avis sur ce restaurant sur Yelp.
# -----------------------------------------------------------
premiere_annee = moyenne_etoiles_annee.groupby(['restaurant_id']).min().reset_index()
derniere_annee = moyenne_etoiles_annee.groupby(['restaurant_id']).max().reset_index()
tendance_etoiles = derniere_annee.merge(premiere_annee, on='restaurant_id', how='left')
tendance_etoiles['tendance_etoiles'] = tendance_etoiles.moyenne_etoiles_annee_y - tendance_etoiles.moyenne_etoiles_annee_x
tendance_etoiles = tendance_etoiles[['restaurant_id', 'tendance_etoiles']]
features = features.merge(tendance_etoiles, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 10) nb_avis
# Le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
nb_total_avis = avis.groupby('restaurant_id').size().reset_index(name='nb_total_avis')
features = features.merge(nb_total_avis, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 11) nb_avis_favorables
# Le nombre total d'avis favorables et positifs pour ce restaurant.
# On considère un avis "favorable" si son nombre d'étoiles est >=3.
# -----------------------------------------------------------
nb_avis_favorables = avis[avis.etoiles >= 3].groupby('restaurant_id').size().reset_index(name='nb_avis_favorables')
features = features.merge(nb_avis_favorables, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 12) nb_avis_defavorables
# Le nombre total d'avis défavorables pour ce restaurant.
# On considère un avis comme "défavorable" si son nombre d'étoiles est  < 3.
# -----------------------------------------------------------
nb_avis_defavorables = avis[avis.etoiles < 3].groupby('restaurant_id').size().reset_index(name='nb_avis_defavorables')
features = features.merge(nb_avis_defavorables, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 13) ratio_avis_favorables
#  Le nombre d'avis favorables et positifs sur le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
nb_total_avis = avis.groupby('restaurant_id').size().reset_index(name='nb_total_avis') 
ratio_avis_favorables = nb_avis_favorables.merge(nb_total_avis, on='restaurant_id', how='left')
ratio_avis_favorables['ratio_avis_favorables'] = ratio_avis_favorables.nb_avis_favorables / ratio_avis_favorables.nb_total_avis
ratio_avis_favorables = ratio_avis_favorables[['restaurant_id', 'ratio_avis_favorables']]
features = features.merge(ratio_avis_favorables, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 14) ratio_avis_defavorables
#  Le nombre d'avis défavorables sur le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
ratio_avis_defavorables = nb_avis_defavorables.merge(nb_total_avis, on='restaurant_id', how='left')
ratio_avis_defavorables['ratio_avis_defavorables'] = ratio_avis_defavorables.nb_avis_defavorables / ratio_avis_defavorables.nb_total_avis
ratio_avis_defavorables = ratio_avis_defavorables[['restaurant_id', 'ratio_avis_defavorables']]
features = features.merge(ratio_avis_defavorables, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 15) nb_avis_favorables_mention
# Le nombre total d'avis qui ont reçu au moins une mention "useful" ou "funny" ou "cool" ET le nombre d'étoiles de l'avis est >=3.
# -----------------------------------------------------------
nb_avis_favorables_mention = avis[(avis.etoiles >= 3) & (avis.useful > 0) | (avis.funny > 0) | (avis.cool > 0)].groupby('restaurant_id').size().reset_index(name='nb_avis_favorables_mention')
features = features.merge(nb_avis_favorables_mention, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 16) nb_avis_defavorables_mention
# Le nombre total d'avis qui ont reçu au moins une mention "useful" ou "funny" ou "cool" ET le nombre d'étoiles de l'avis est <3.
# -----------------------------------------------------------
nb_avis_defavorables_mention = avis[(avis.etoiles < 3) & (avis.useful > 0) | (avis.funny > 0) | (avis.cool > 0)].groupby('restaurant_id').size().reset_index(name='nb_avis_defavorables_mention')
features = features.merge(nb_avis_defavorables_mention, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 17) nb_avis_favorables_elites
# Le nombre total d'avis favorables pour un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------
nb_avis_favorables_elites = avis[(avis.etoiles >= 3) & (avis.useful > 0) | (avis.funny > 0) | (avis.cool > 0)].merge(utilisateurs[(utilisateurs.elite == 1) & (utilisateurs.nb_avis >= 100) & (utilisateurs.nb_avis_mention >= 100)], on='utilisateur_id', how='left')
nb_avis_favorables_elites = nb_avis_favorables_elites.groupby('restaurant_id').size().reset_index(name='nb_avis_favorables_elites')
features = features.merge(nb_avis_favorables_elites, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 18) nb_avis_defavorables_elites
# Le nombre total d'avis défavorables pour un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------
nb_avis_defavorables_elites = avis[(avis.etoiles < 3) & (avis.useful > 0) | (avis.funny > 0) | (avis.cool > 0)].merge(utilisateurs[(utilisateurs.elite == 1) & (utilisateurs.nb_avis >= 100) & (utilisateurs.nb_avis_mention >= 100)], on='utilisateur_id', how='left')
nb_avis_defavorables_elites = nb_avis_defavorables_elites.groupby('restaurant_id').size().reset_index(name='nb_avis_defavorables_elites')
features = features.merge(nb_avis_defavorables_elites, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 19) nb_conseils
#  Le nombre total de conseils (tips) associés à un restaurant.
# -----------------------------------------------------------
nb_conseils = conseils.groupby('restaurant_id').size().reset_index(name='nb_conseils')
features = features.merge(nb_conseils, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 20) nb_conseils_compliment
# Le nombre total de conseils qui ont reçu au moins un compliment (voir Table Conseils).
# -----------------------------------------------------------
nb_conseils_compliment = conseils[conseils.nb_compliments > 0].groupby('restaurant_id').size().reset_index(name='nb_conseils_compliment')
features = features.merge(nb_conseils_compliment, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 21) nb_conseils_elites
# Le nombre total de conseils sur un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------
nb_conseils_elites = conseils.merge(utilisateurs[(utilisateurs.elite == 1) & (utilisateurs.nb_avis >= 100) & (utilisateurs.nb_avis_mention >= 100)], on='utilisateur_id', how='left')
nb_conseils_elites = nb_conseils_elites.groupby('restaurant_id').size().reset_index(name='nb_conseils_elites')
features = features.merge(nb_conseils_elites, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 22) nb_checkin
# Le nombre total de visites.
# -----------------------------------------------------------
nb_checkin = checkin.groupby('restaurant_id').size().reset_index(name='nb_checkin')
features = features.merge(nb_checkin, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 23) moyenne_checkin
# La moyenne de visites par année.
# -----------------------------------------------------------
moyenne_checkin = checkin.groupby(['restaurant_id', 'annee']).size().reset_index(name='moyenne_checkin')
moyenne_checkin = moyenne_checkin.groupby('restaurant_id').moyenne_checkin.mean().reset_index(name='moyenne_checkin')
features = features.merge(moyenne_checkin[['restaurant_id', 'moyenne_checkin']], on='restaurant_id', how='left')

# -----------------------------------------------------------
# 24) ecart_type_checkin
# L'écart type de visites par année.
# Ici, on calcul l'écart type pour le total des visites par année.
# -----------------------------------------------------------
checkin_year =  checkin.groupby(['restaurant_id', 'annee']).size().reset_index(name='moyenne_checkin')
ecart_type_checkin = checkin_year.groupby('restaurant_id').moyenne_checkin.std().reset_index(name='ecart_type_checkin')
features = features.merge(ecart_type_checkin, on='restaurant_id', how='left')

# -----------------------------------------------------------
# 25) chaine
#  Prend 0 ou 1. La valeur 1 indique que le restaurant fait parti d'une chaîne (p. ex. McDonald).
#  On considère un restaurant comme il fait partie d'une chaîne, s’il existe un autre restaurant dans la base de données qui a le même nom.
# -----------------------------------------------------------
chaine = restaurants[['restaurant_id', 'nom']].groupby('nom').size().reset_index(name='chaine')
chaine.loc[chaine['chaine'] > 1, 'chaine' ] = 1
restaurants_chaine = restaurants[['restaurant_id', 'nom']].merge(chaine, on='nom', how='left')
features = features.merge(restaurants_chaine[['restaurant_id', 'chaine']], on='restaurant_id', how='left')

# -----------------------------------------------------------
# 26) nb_heures_ouverture_semaine
# Le nombre total d'heures d'ouverture du restaurant par semaine.
# -----------------------------------------------------------
heures_semaine = []
for index, row in horaires.iterrows():
    restaurant_id = row['restaurant_id']
    total_hours = pd.Timedelta(0)
    # Parcourir chaque jour de la semaine
    for day in days:
        # Extraire les horaires d'ouverture et de fermeture
        hours_str = str(row[day])
        if pd.isna(hours_str):
            total_hours += pd.to_timedelta(0)

        if '-' in hours_str:
            opening_hours, closing_hours = hours_str.split('-')
            opening_hours = convert_time(opening_hours)
            closing_hours = convert_time(closing_hours)
        else:
            opening_hours = None
            closing_hours = None
            
        # Calculer la durée d'ouverture pour chaque jour
        if opening_hours is not None and closing_hours is not None:
            opening_time = pd.Timestamp(opening_hours.strftime('%H:%M:%S'))
            closing_time = pd.Timestamp(closing_hours.strftime('%H:%M:%S'))
            total_hours += pd.to_timedelta((closing_time - opening_time).seconds, unit='s')
        else:
            total_hours += pd.to_timedelta(0)

    nb_heures_ouverture_semaine = pd.DataFrame([[restaurant_id, total_hours.total_seconds() / 3600]], columns=['restaurant_id', 'nb_heures_ouverture_semaine'])
    heures_semaine.append(nb_heures_ouverture_semaine)
   
features['nb_heures_ouverture_semaine'] = heures_semaine
print("features", features)

# -----------------------------------------------------------
# 27) ouvert_samedi
# Si le restaurant est ouvert les samedis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

# Votre code ici:


# -----------------------------------------------------------
# 28) ouvert_dimanche
# Si le restaurant est ouvert les dimanches (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

# Votre code ici:


# -----------------------------------------------------------
# 29) ouvert_lundi
# Si le restaurant est ouvert les lundis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

# Votre code ici:


# -----------------------------------------------------------
# 30) ouvert_vendredi
# Si le restaurant est ouvert les vendredis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

# Votre code ici:


# -----------------------------------------------------------
# 31) emporter
# Si le restaurant offre le service à emporter (valeur booléenne : 0 ou 1).
emporter = services.iloc[:, [0, 1]]
features = features.merge(emporter, on='restaurant_id', how='left')

# 32) livraison
# Si le restaurant offre le service de livraison (valeur booléenne : 0 ou 1).
livraison = services.iloc[:, [0, 2]]
features = features.merge(livraison, on='restaurant_id', how='left')

# 33) bon_pour_groupes
# Si le restaurant est approprié pour les groupes (valeur booléenne : 0 ou 1).
bon_pour_groupes = services.iloc[:, [0, 3]]
features = features.merge(bon_pour_groupes, on='restaurant_id', how='left')

# 34) bon_pour_enfants
# Si le restaurant est approprié pour les enfants (valeur booléenne : 0 ou 1).
bon_pour_enfants = services.iloc[:, [0, 4]]
features = features.merge(bon_pour_enfants, on='restaurant_id', how='left')

# 35) reservation
# Si on a besoin de faire une réservation au restaurant (valeur booléenne : 0 ou 1).
reservation = services.iloc[:, [0, 5]]
features = features.merge(reservation, on='restaurant_id', how='left')

# 36) prix
# Le niveau de prix du restaurant. Il existe trois niveaux, 1 (abordable), 2 (moyen) et 3 (coûteux).
prix_niveaux = {0: '', 1: 'abordable', 2: 'moyen', 3: 'coûteux', 4: 'coûteux'}
restaurants_prix = services[['restaurant_id', 'prix']]
restaurants_prix['prix'] = restaurants_prix['prix'].apply(lambda x: prix_niveaux[x])
features = features.merge(restaurants_prix, on='restaurant_id', how='left')

# 37) terrasse
# Si le restaurant a une terrasse (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------
terrasse = services.iloc[:, [0, 7]]
features = features.merge(terrasse, on='restaurant_id', how='left')
# print("features", features)
# -----------------------------------------------------------
# Sauvegarder l'ensemble de données dans un fichier csv afin d'être utilisé dans l'étape suivante
# features.to_csv("donnees/features.csv", index=False)

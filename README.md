# restautant_classification
TP du cour INF7370 UQAM sur la classification des restaurants

Le but de ce travail est d’analyser et de comparer la performance de quelques algorithmes d’apprentissage automatiques : Arbre de décision, Forêt d’arbres décisionnels (Random Forest) et Classification bayésienne naïve. Les algorithmes doivent être entraînés pour
classifier des restaurants en deux situations : fermeture définitive / restaurant encore ouvert.
Nous utiliserons le langage Python avec la librairie scikit-learn pour l'entraînement et l’évaluation des modèles ainsi que la librairie pandas pour l’exploitation des dataframes en fichiers csv. Les données pour le travail seront récupérées de Yelp qui est une application mobile qui sert comme un répertoire en ligne pour la recherche des restaurants par différents critères. (https ://www.yelp.com/dataset)

Il s’agit notamment de :
- utilisateurs : Contient la liste des utilisateurs avec quelques détails.
- avis : Contient les informations qui se rattachent aux avis rédigés pour chaque restaurant avec le nombre d'étoiles accordées.
- conseils : Contient les informations qui se rattachent aux conseils (tips) rédigés par les utilisateurs sur les restaurants.
- checkin : Contient les dates et heures de visite par restaurant.
- restaurants : Contient la liste des restaurants avec quelques détails.
- horaires : Contient les heures d'ouverture de chaque restaurant pour les sept jours de la semaine.
- services : Contient la liste des services offerts par chaque restaurant.
- categories : Contient la liste des catégories des restaurants.

Le travail est divisé en 3 tâches :
- Calcul des attributs (features engineering) a partir des données fournies
- Pré-traitement du dataset issu de la première tâche
- Entraînement des modèles et analyse des résultats.

# Khadijetou_Nourdine_C17879_2026_Machine_learning
# 📑 Rapport de Mini-Projet : Apprentissage Supervisé Linéaire

**Étudiante :** Khadijetou Nourdine Sow (C17879)  
**Niveau :** Master IA  
**Année :** 2026

---

##  Introduction
L’idée du projet est de comprendre comment prédire une valeur numérique continue comme un prix ou un cout à partir de données historiques pour nous permettre d’apprendre à :
* **Préparer** les données.
* **Créer** un modèle mathématique.
* **Mesurer** si le modèle choisi est efficace ou pas.

---

## Partie 1 : Régression Linéaire
J’ai choisi d'utiliser **Medical Insurance Cost** sur Kaggle comme jeu de données de régression. Une fois le modèle trouvé, j’ai téléchargé le fichier `.csv` et l’ai importé dans le notebook avec la bibliothèque **Pandas**.

### 1. Analyser les corrélations via une matrice de chaleur (Heatmap)
J’ai utilisé les bibliothèques `seaborn` et `matplotlib` pour générer une matrice de chaleur (Heatmap) pour pouvoir graphiquement voir si l’une des variables comme l'âge avait une forte influence sur le coût de l’assurance.

**Observations :**
* Avec l'âge, les frais médicaux ont tendance à augmenter (**0,299**).
* L’IMC (**0,198**) mène parfois à des frais médicaux plus élevés.
* Le nombre d'enfants (**0,068**) n’a pas une grande influence sur les frais médicaux.



### 2. Formaliser le modèle (Algorithme ML)
Le but est de prédire $y$ qui représente les frais médicaux (charges) en fonction de $x$ (l'âge, bmi, children). La régression linéaire modélise la relation entre $y$ et $X$ comme une combinaison linéaire selon l'algorithme :
$$ y = β0 + Pβixi + ϵ$$

### 3. Évaluer la performance (R2 et MSE)
On a fait l'entraînement du modèle avec **scikit-learn** en séparant nos données en 2 parties : 
* **Train set :** pour que le modèle apprenne les relations.
* **Test set :** pour vérifier si le modèle peut prédire sur des données jamais vues.

**Résultats obtenus :**
* **R² de 0.78** : indique que 78% de la variabilité des frais est expliquée par le modèle.
* **RMSE de 5796.28 €** : représente l'erreur moyenne de prédiction du modèle en euros.

### 4. Interprétation des coefficients $\beta_i$
Chaque coefficient correspond au "poids" d'une variable. On voit que le facteur le plus déterminant pour les frais médicaux est **d'être fumeur**, suivi par l'IMC, le nombre d'enfants et l'âge. Le sexe a un impact minimal et la région a une influence modérée.

---

## 🧪 Partie 2 : Régression Logistique
Le but est de chercher à classer les fleurs et à transformer ces classes en binaires.

### 1. Préparer les données et normaliser les variables d’entrée
On a chargé le dataset et transformé le problème de 3 classes en un problème binaire (oui/non). On a utilisé le **StandardScaler** car la régression logistique utilise la fonction sigmoïde et des calculs de distance.

### 2. Modéliser la probabilité via la fonction sigmoïde (Algorithme ML)
Afin de transformer un score linéaire en une probabilité entre 0 et 1, on utilise **LogisticRegression** qui gère automatiquement la fonction :
$$P(y = 1|x) = \frac{1}{1+e^{-z}}$$
* Si $P > 0.5$, l'IA prédit la classe 1.
* Si $P < 0.5$, l'IA prédit la classe 0.

### 3. Évaluer le modèle via une Matrice de Confusion
Pour évaluer si mon modèle s’est trompé de fleurs, j’ai généré une matrice de confusion. J’ai obtenu **1.00** partout, ce qui signifie que le modèle ne s’est pas trompé sur le dataset. Cette performance parfaite (Accuracy de 100%) démontre que les classes sont linéairement séparables dans l'espace des caractéristiques après normalisation.

### 4. Calculer l’Accuracy, la Précision et le Rappel (Recall)
L'obtention de scores parfaits (**1.00**) s'explique par la nature du dataset **Iris**. Dans le cadre de cette classification binaire (Classe 0 vs Autres), les caractéristiques physiques de l'espèce Setosa sont radicalement différentes des autres espèces. Géométriquement, il existe une frontière nette que la régression logistique peut tracer sans aucune erreur de classement.

---

## ✅ Conclusion
La régression linéaire est idéale pour prédire une valeur exacte, tandis que la régression logistique excelle pour la classification. La qualité des résultats dépend avant tout de la préparation des données (encodage et normalisation).


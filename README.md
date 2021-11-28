# UE12_evaluation_numerique



Vous allez travaillez dans ce projet sur des données réelles concernant des défauts observés dans des soudures. Ces défauts sont problématiques car ils peuvent occasionner la rupture prématurée d'une pièce, ce qui peut avoir de lourdes conséquences. De nombreux travaux actuels visent donc à caractériser la nocivité des défauts. La morphologie de ces défauts est un paramètre qui influe au premier ordre sur cette nocivité.

Dans ce projet, vous aurez l'occasion de manipuler des données qui caractérisent la morphologie de défauts réels observés dans une soudure volontairement ratée ! Votre mission est de mener une analyse permettant de classer les défauts selon leur forme : en effet, deux défauts avec des morphologies similaires auront une nocivité comparable.



# Évaluation de python-numérique


Ce projet constitue l'évaluation de la partie python-numérique de l'ECUE PE. Le sujet est proposé par `Laurent Lacourt`; le problème et les données sont réels; le code est écirt par `Xindong XU`.
Le sujet se trouve dans le fichier `00-eval-pe.nb.py`. Vous devez compléter ce notebook avec vos implémentations et remarques. 



**Réalisation, condition et date de rendu du projet**  
* le projet doit être réalisé **individuellement**
* il doit être rendu sous la forme d'un notebook (*vous complétez le notebook initial avec votre code et vos explications*)
* vous avez **5 semaines** pour le réaliser: il doit donc être envoyé **avant le `17 décembre minuit`**
* adressez votre `notebook` par mail à `laurent.lacourt@mines-paristech.fr`  
  veuillez utiliser votre adresse `@mines-paristech.fr` comme adresse d'émission
  
  
  
**Des questions ?**  
* les questions concernant le sujet, doivent être posées à Laurent Lacourt (après avoir bien réfléchi à la question)
* les questions techniques, à vos enseignants de Python-numérique (après avoir bien réfléchi à la question)
* pour tout autre problème, contactez valerie.roy@mines-paristech.fr



**Attendus sur votre code**
* le code doit être écrit en anglais
* les commentaires et les explications peuvent rester en français 
* le code doit être propre, lisible et commenté aux endroits adéquats
* vos fonctions doivent contenir une `docstring`
* le code doit s'exécuter sans erreur



## Installation des modules nécessaires
Afin de visualiser une partie des données qui vous sont mises à disposition pour ce projet, vous devez installer un module spécifique. Pour ne pas surcharger votre environnement de base, nous vous conseillons de créer un environnement spécifique à l'évaluation et d'y installer les modules nécessaires: 

```bash
conda create -n eval-pe python=3.9
conda activate eval-pe
pip install -r requirements.txt
jupyter notebook
```

**Dans la suite, vous devrez taper la commande suivante avant de commencer à travailler sur le sujet :**

```
conda activate eval-pe
```

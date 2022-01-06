# -*- coding: utf-8 -*-
# %% [markdown]
# # Évaluation de python-numérique
#
# ***Analyse morphologique de défauts 3D***
#
# ***

# %% [markdown]
# ## Votre identité

# %% [markdown]
# Ne touchez rien, remplacez simplement les ???
#
# Prénom: Xindong
#
# Nom: XU
#
# Langage-avancé (Python ou C++): C++
#
# Adresse mail: xindong.xu@mines-paristech.fr
#
# ***

# %% [markdown]
# ## Quelques éléments de contexte et objectifs
#
# Vous allez travaillez dans ce projet sur des données réelles concernant des défauts observés dans des soudures. Ces défauts sont problématiques car ils peuvent occasionner la rupture prématurée d'une pièce, ce qui peut avoir de lourdes conséquences. De nombreux travaux actuels visent donc à caractériser la **nocivité** des défauts. La morphologie de ces défauts est un paramètre qui influe au premier ordre sur cette nocivité.
#
# Dans ce projet, vous aurez l'occasion de manipuler des données qui caractérisent la morphologie de défauts réels observés dans une soudure volontairement ratée ! Votre mission est de mener une analyse permettant de classer les défauts selon leur forme : en effet, deux défauts avec des morphologies similaires auront une nocivité comparable. 

# %% [markdown]
# ### Import des librairies numériques
#
# Importez les librairies numériques utiles au projet, pour le début du projet, il s'agit de `pandas`, `numpy` et `pyplot` de `matplotlib`.

# %%
# votre code ici
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# On vérifie que l'environnement est bien activé;
print(sys.version)

# %% [markdown]
# ## Lecture des données
#
# Les données se trouvent dans le fichier `defect_data.csv`. Ce fichier contient treize colonnes de données :
# * la première colonne est un identifiant de défaut (attention, il y a 4040 défauts mais les ids varient entre 1 et 4183) ;
# * les neuf colonnes suivantes sont des descripteurs de forme sur lesquels l'étude va se concentrer ;
# * les trois dernières colonnes sont des indicateurs mécaniques auxquels nous n'allons pas nous intéresser dans ce projet.
#
# Lisez les données dans une data-frame en ne gardant que les 9 descripteurs de forme et en indexant vos lignes par les identifiants des défauts. Ces deux opérations doivent être faites au moment de la lecture des données.
#
# Affichez la forme et les deux dernières lignes de la data-frame.

# %%
# votre code ici
# On définit les colonnes propres et l'indice;
col1 = ['radius1', 'lambda1', 'lambda2', 'convexity', 'sphericity', 'varCurv', 'intCurv', 'b1', 'b2', 'id']
col2 = 'id'
df = pd.read_csv('defect_data.csv', index_col = col2 , usecols = col1)

# %%
df.head()

# %%
len(df)

# %%
df[['radius1', 'lambda1', 'lambda2', 'convexity', 'sphericity', 'varCurv', 'intCurv', 'b1', 'b2']].describe()

# %% [markdown]
# ### Parenthèse sur descripteurs morphologiques 
#
# **Note: cette section vous donne du contexte quant à la signification des données que vous manipulez et la façon dont elles ont été acquises. Si certains aspects vous semblent nébuleux, cela ne vous empêchera pas de finir le projet !**
#
# Vous allez manipuler dans ce projet des descripteurs morphologiques. Ces descripteurs sont ici utilisés pour caractériser des défauts, observés par tomographie aux rayons X dans des soudures liant deux pièces métalliques. La tomographie consiste à prendre un jeu de radiographies (comme chez le médecin, avec un rayonnement plus puissant) en faisant tourner la pièce entre chaque prise de vue. En appliquant un algorithme de reconstruction idoine à l'ensemble des clichés, il est possible de remonter à une image 3D des pièces scannées. Plus la zone que l'on traverse est dense plus elle est claire (comme chez le médecin : vos os apparaissent plus clair que vos muscles). Dans notre cas, le constraste entre les défauts constitués d'air et le métal est très marqué : on observe donc les défauts en noir et le métal en gris. Un défaut est donc un amas de voxels (l'équivalent des pixels pour une image 3D) noirs. Sur l'image ci-dessous, les défauts ont été extraits et sont représentés en 3D par les volumes rouges. 
#
# <img src="media/defects_3D.png" width="400px">
#
# Vous voyez qu'ils sont nombreux, de taille et de forme variées. À chaque volume rouge que vous observez correspond une ligne de votre `DataFrame` qui contient les descripteurs morphologiques du-dit défaut. 
#
#
# #### Descripteur $r_1$ (`radius1`)
# En notant $N$ le nombre de voxels constituant le défaut, on obtient le volume du défaut $V=N\times v_0$ (où $v_0$ est le volume d'un voxel).On peut alors définir son *rayon équivalent* comme le rayon de la sphère de même volume soit :
# \begin{equation*}
#  R_{eq} = \left(\frac{3V}{4\pi}\right)^{1/3}
# \end{equation*}
#
# On définit ensuite le *rayon moyen* $R_m$ du défaut comme la moyenne sur tous les voxels de la distance au centre de gravité du défaut. 
#
# $R_{eq}$ et $R_m$ portent une information sur la taille du défaut. En les combinant comme suit:
# \begin{equation*}
#  r_1 = \frac{R_{eq} - R_m}{R_m}
# \end{equation*}
# on la fait disparaître : $r_1$ vaut 1/3 pour une sphère quel que soit son rayon. 
#
# **Note :** vous aurez remarqué que $r_1$ est donc sans dimension.
#
# #### Descripteurs basés sur la matrice d'inertie ($\lambda_1$ et $\lambda_2$) (`lambda1`, `lambda2`)
# La matrice d'inertie de chaque défaut est calculée. Pour ce faire, on remplace tout simplement les intégrales sur le volume présentes dans les formules que vous connaissez par une somme sur les voxels. Par exemple:
# \begin{equation}
#  I_{xy} = -\sum\limits_{v\in\rm{defect}} (x(v)-\bar{x})(y(v)-\bar{y})\qquad \text{avec } \bar{x} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} x(v) \text{ et } \bar{y} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} y(v)
# \end{equation}
# Cette matrice est symétrique réelle, elle peut donc être diagonalisée. Les trois valeurs propres obtenues $I_1 \geq I_2 \geq I_3$ sont les moments d'inertie du défaut dans son repère principal d'inertie. Ces derniers portent de manière intrinsèque une information sur le volume du défaut. Pour la faire disparaître, il suffit de normaliser ces moments comme suit : 
#
# \begin{equation}
#  \lambda_i = \frac{I_i}{I_1+I_2+I_3}
# \end{equation}
#
# On obtient alors trois indicateurs $\lambda_1 \geq \lambda_2 \geq \lambda_3$ vérifiant notamment $\lambda_1 + \lambda_2 + \lambda_3 = 1$ ce qui explique que l'on ne garde que les deux premiers. En utilisant les propriétés des moments d'inertie, on peut montrer que les points obtenus se situent dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$ dans le plan $(\lambda_1, \lambda_2)$. Vous pourrez vérifier cela dans la suite ! 
#
# La position du point dans le triangle renseigne sur sa forme *globale*, comme indiqué par l'image suivante : 
# ∑
# <img src="media/l1_l2.png" width="400px">
#
# #### Convexité (`convexity`)
#
# L'indicateur de convexité utilisé est simplement le rapport entre le volume du défaut et de son convexe englobant. $C = V/V_{CH} \leq 1$. Lorsque qu'un défaut est convexe, il est égal à son convexe englobant et donc $C$ vaut 1.
#
# #### Sphéricité (`sphericity`)
#
# L'indicateur de sphéricité permet de calculer l'écart d'un défaut à une sphère. On utilise ici la caractéristique de la sphère qui minimise la surface extérieure pour un volume donné. La grandeur : 
# \begin{equation*}
# I_S = \frac{6\sqrt{\pi}V}{S^{3/2}}
# \end{equation*}
# où $V$ est le volume du défaut et $S$ sa surface vaut 1 pour une sphère et est inférieur à 1 sinon. 
#
# #### Indicateurs basés sur la mesure de la courbure (`varCurv`, `intCurv`)
#
# Les deux courbures principales $\kappa_1$ et $\kappa_2$ sont calculées en chaque point de la surface des défauts ([ici pour les plus curieux](https://fr.wikipedia.org/wiki/Courbure_principale)). Ces courbures permettent de caractériser la forme locale du défaut. Elle sont combinées pour calculer la courbure moyenne $H = (\kappa_1+\kappa_2)/2$ et la courbure de Gauss $\gamma = \kappa_1\kappa_2$. Pour s'affranchir de l'information sur la taille (pour une sphère de rayon $R$, on a en tout point $\kappa_1 = \kappa_2 = 1/R$), les défauts sont normalisés en volume avant d'en calculer les courbures.
# Les indicateurs retenus sont les suivants:
#
#  - la variance de la courbure de Gauss (colonne `varCurv`) $Var(H)$ ; 
#  - l'intégrale de $\gamma$ sur la surface du défaut(colonne `intCurv`) $\int_S \gamma dS$. 
#  
# Ces indicateurs valent respectivement $0$ et $4\pi$ pour une sphère.
#
# #### Indicateurs sur la boite englobante $(\beta_1, \beta_2)$ (`b1`, `b2`) 
#
# Finalement, c'est une information sur la boite englobante du défaut dans son repère principal d'inertie qui est cachée dans $(\beta_1, \beta_2)$. En notant $B_1, B_2, B_3$ les dimensions (décroissantes) de la boite englobante, on réalise la même normalisation que pour les moments d'inertie : 
# \begin{equation}
#  \beta_i = \frac{B_i}{B_1+B_2+B_3}
# \end{equation}
#
# ***

# %% [markdown]
# ## Visualisation des défauts
#
# Pour que vous saissiez un peu mieux la signification des descripteurs morphologiques, nous avons concocté une petite fonction utilitaire qui vous permettra de visualiser les défauts. Pour que vous puissiez interagir avec `pyplot`, il nous est imposé de changer le backend avec la commande `%matplotlib notebook` et de recharger le module. Pour revenir dans le mode de visualisation précédent, vous devrez évaluer la cellule qui contient la commande `%matplotlib inline` qui arrive un peu plus tard !  
#
# *Nous n'avons malheureusement pas trouvé de solution pour que ce changement soit transparent pour vous... :(*
#
# Amusez vous à chercher des défauts **extrêmes** pour comprendre. Par exemple le défaut qui maximise $\lambda_2$ sera celui qui a la forme la plus *filaire* alors que celui qui minimise aura la forme la plus *plate*. Pourquoi ne pas jeter un coup d'oeil au défaut le moins convexe ? 

# %%
# Ne touchez pas à ça c'est pour la visualisation interactive ! 
# %matplotlib notebook
from importlib import reload
from utilities import plot_defect
reload(plt)
# À partir de maintenant vous pouvez vous amuser ! 

# On récupère un id intéressant
id_to_plot = df.index[df['convexity'].argmin()]
# On affiche à l'écran les valeurs de ses descripteurs
print(df.loc[id_to_plot])
# On l'affiche
plot_defect(id_to_plot)

# %% [markdown]
# N'oubliez pas d'aller rendre visite au défaut avec l'id `4022` qui a une forme rigolote, avec ses petites excroissances. 

# %%
# Le défaut 4022 ! 
print(df.loc[4022])
plot_defect(4022)

# %% [markdown]
# On vous parlait juste avant de défauts de morphologie proche ! Et si une simple distance euclidienne en dimension 9 fonctionnait ? Calculez le défaut le plus proche du défaut `4022` dans l'espace de dimension 9, et tracez-le ! Se ressemblent-ils ?

# %%
# Votre code ici
# On choisit le début de calcul, en supposant que le plus proche défaut est de id 1;
id_proche = 1
# D'abord, on calcul la distance euclidienne entre id 1 et id 4022;
dis_proche = 0
for col in np.array(df.columns):
    dis_proche = dis_proche + (df.loc[4022][col] - df.loc[id_proche][col])**2
    
for i in np.array(df.index):
    dis = 0
    for col in np.array(df.columns):
        dis = dis + (df.loc[4022][col] - df.loc[i][col])**2
    if (dis < dis_proche) and (i != 4022):
        id_proche = i
        dis_proche = dis

# %%
# On affiche des résultats trouvés;
id_proche, dis_proche

# %%
# On affiche à l'écran les valeurs de leurs descripteurs;
print(df.loc[id_proche], '\n')
print(df.loc[4022])

# %%
# On affiche le défaut préliminaire qui est le plus proche de façon morphologique de id 4022;
plot_defect(id_proche)

# %% [markdown]
# **Eh non!** Le défaut le plus proche du défaut `4022` est une patatoïde quelconque. Deux explications sont possibles :
#  * soit la distance euclidienne n'est pas pertinente ici ;
#  * soit le défaut `4022` est le seul avec des petites excroissances... 
# Je vous laisse aller voir le défaut `796` pour trancher entre les deux propositions.. 
#

# %%
print(df.loc[796])
plot_defect(796)
# Évidemment on observe qu'il existe un défaut (id 796) qui est plus proche de id 4022 avec des petites excroissances;
# La distance euclidienne directe ne peut pas nous rendre le meilleur résultat;

# %%
# On revient en mode de visu standard après avoir évalué cette cellule
# %matplotlib inline
reload(plt)


# %% [markdown]
# ## Visualisation des données
#
# Avant de commencer toute analyse à proprement parler de données, il est nécessaire de passer un moment à les observer.
#
# Pour ce faire, nous allons écrire des fonctions utilitaires qui se chargeront de tracer des graphes classiques.

# %% [markdown]
# ### Tracé d'un histogramme
#
# Écrire une fonction `histogram` qui trace l'histogramme d'une série de points. 
#
#
# Par exemple l'appel `histogram(df['radius1'], nbins=10)` devrait renvoyer quelque chose qui ressemble à ceci:
#
# <img src="media/defects-histogram.png" width="400px">

# %%
# votre code ici
def histogram(x, nbins = 10):
    """
    Cette fonction histogram trace l'histogramme d'une série de points;

    Paramètres :
    ----------
    x : (n,)tableau ou séquence de (n,)tableaux, par exemple : pandas.core.series.Series ou liste;
    Dans x, il réserve des donnés à tracer;
    
    nbins : int ou séquence ou str (default : 10);
    Si nbins est un entier, il définit le nombre de cases de largeur égale sur l'axe;
    Si nbins est une séquence, elle définit les bords de la case, 
    y compris le bord gauche de la première case et le bord droit de la dernière case;

    Returns :
    -------
    Un diagramme en barres;
    """
    plt.hist(x, bins = nbins)
    plt.show()


# %%
# pour vérifier
histogram(df['radius1'], nbins=10)

# %%
# On vérifie si le nbins donné est une séquence définie par le min et le max des donnés :
bins = np.linspace(df['radius1'].min(), df['radius1'].max(),11)

# %%
histogram(df['radius1'], nbins = bins)

# %%
# pour vérifier
# c'est bien si votre fonction marche aussi avec une dataframe
histogram(df[['radius1']], nbins=10)


# %% [markdown]
# #### *Bonus* : un histogramme adaptable aux goûts de chacun
#
# Modifier la fonction `histogram` pour que l'utilisateur puisse préciser par exemple: le nom des étiquettes des axes, les couleurs à utiliser pour représenter l'histogramme...
#
# Par exemple en appelant
# ```python
# histogram2(df['radius1'], nbins=10,
#        xlabel='radius1', ylabel='occurrences',
#        histkwargs={'color': 'red'})
# ```
#
# on obtiendrait quelquechose comme ceci
#
# <img src="media/defects-histogram2.png" width="400px">

# %%
# votre code ici
def histogram2(x, nbins, xlabel, ylabel, hist_kwargs):
    """
    Cette fonction histogram2 trace l'histogramme d'une série de points,
    et affiche le nom des étiquettes des axes, les couleurs à utiliser pour représenter l'histogramme... ;

    Paramètres :
    ----------
    x : (n,)tableau ou séquence de (n,)tableaux, par exemple : pandas.core.series.Series ou liste;
    Dans x, il réserve des donnés à tracer;
    
    nbins : int ou séquence ou str (default : 10);
    Si nbins est un entier, il définit le nombre de cases de largeur égale sur l'axe;
    Si nbins est une séquence, elle définit les bords de la case, 
    y compris le bord gauche de la première case et le bord droit de la dernière case;
    
    xlabel, ylabel : str;
    Les noms des étiquettes des axes;
    
    hist_kwargs : dictionnaire;
    Les couleurs à utiliser pour représenter l'histogramme;

    Returns :
    --------
    Un diagramme en barres avec le nom des étiquettes des axes et avec la couleur choisie pour représenter l'histogramme;
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(x, bins = nbins, color = hist_kwargs['color'])
    plt.show()


# %%
# pour vérifier
# seulement si la fonction est définie
if 'histogram2' in globals():
    histogram2(df['radius1'], nbins=10,
           xlabel='radius1', ylabel='occurrences',
           hist_kwargs={'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")


# %% [markdown]
# ### Tracé de nuages de points
#
# Écrire une fonction `correlation_plot` qui permet de tracer le nuage de points entre deux séries de données. 
# L'appel de cette fontion comme suit `correlation_plot(df['lambda1'], df['lambda2'])` devrait donner une image ressemblant à celle-ci :
#
# Ces tracés illustrent le *degré de similarité* des colonnes. Notons, qu'il existe des indices de similarité comme par exemple: la covariance, le coefficient de corrélation de Pearson...
#
#
# <img src="media/defects-correlation.png" width="400px">

# %%
# votre code ici
def correlation_plot(x, y):
    """
    Cette fonction correlation_plot trace une figure illustrant le degré de similarité des colonnes de dataframe;

    Paramètres :
    ----------
    x, y : (n,)tableau ou séquence de (n,)tableaux, par exemple : pandas.core.series.Series ou liste;
    Dans x, y, il réserve des donnés à tracer;

    Returns :
    --------
    Un diagramme en points;
    """
#     lambda1 = []
#     lambda2 = []
#     for i in np.array(df.index):
#         lambda1.append(x[i])
#         lambda2.append(y[i])
#     # On transfert les données de data.séries à numpy.array;
    plt.scatter(x, y, c = 'black', marker = '.')
    plt.show()


# %%
# pour vérifier
correlation_plot(df['lambda1'], df['lambda2'])


# %% [markdown]
# #### *Bonus* les nuages de points pour l'utilisateur casse-pieds (ou daltonien ;) )
# Modifier la fonction `correlation_plot` pour que l'utilisateur puisse préciser des noms pour les axes, et choisir l'aspect des points tracés (couleur, taille, forme, ...). 
#
# par exemple en appelant
# ```python
# correlation_plot2(df['lambda1'], df['lambda2'], 
#                   xlabel='lambda1', ylabel='lambda2',
#                   plot_kwargs={'marker': 'x', 'color': 'red'})
# ```
# on obtiendrait quelque chose comme
#
# <img src="media/defects-correlation2.png" width="400px">

# %%
# votre code ici
def correlation_plot2(x, y, xlabel, ylabel, plot_kwargs):
    """
    Cette fonction correlation_plot2 trace une figure illustrant le degré de similarité des colonnes de dataframe,
    et affiche le nom des étiquettes des axes et l'aspect des points tracés (couleur, taille, forme, ...);
    
    Paramètres :
    ----------
    x, y : (n,)tableau ou séquence de (n,)tableaux, par exemple : pandas.core.series.Series ou liste;
    Dans x, y, il réserve des donnés à tracer;
    
    xlabel, ylabel : str;
    Les noms des étiquettes des axes;
    
    plot_kwargs : dictionnaire;
    Il comprend la couleur à utiliser pour représenter l'histogramme, la taille et la forme de points sur la figure;

    Returns :
    --------
    Un diagramme en points avec le nom des étiquettes des axes 
    et avec la couleur, taille, forme choisies pour représenter l'histogramme;
    """
#     lambda1 = []
#     lambda2 = []
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
    
#     for i in np.array(df.index):
#     #for i in range(1,5):
#         lambda1.append(x[i])
#         lambda2.append(y[i])
    plt.scatter(x, y, c = plot_kwargs['color'], marker = plot_kwargs['marker'], alpha = 0.65)
    plt.show()


# %%
# pour vérifier 

# seulement si la fonction est définie
if 'correlation_plot2' in globals():
    correlation_plot2(df['lambda1'], df['lambda2'], 
                      xlabel='lambda1', ylabel='lambda2',
                      plot_kwargs={'marker': 'x', 'color': 'red'})
else:
    print("vous avez choisi de ne pas faire le bonus")    


# %% [markdown]
# #### *Bonus 2* Tracer le triangle d'inertie en plus
#
# On vous disait plus tôt que les points dans le plan $(\lambda_1, \lambda2)$ sont forcément dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$. 
# Essayez de superposer les données au triangle pour mettre cela en évidence ! 
# Le résultat pourrait ressembler à ceci :
#
# <img src="media/defects-correlation3.png" width="400px">
#
# (Vous pouvez faire ce bonus sans avoir fait le précédent)

# %%
# Votre code ici
def correlation_plot3(x, y, xlabel, ylabel, xlim, ylim, plot_kwargs):
    """
    Cette fonction correlation_plot2 trace une figure illustrant le degré de similarité des colonnes de dataframe,
    qui est encadré par un triangle caractéristique,
    et elle affiche le nom des étiquettes des axes et l'aspect des points tracés (couleur, taille, forme, ...);
    
    Paramètres :
    ----------
    x, y : (n,)tableau ou séquence de (n,)tableaux, par exemple : pandas.core.series.Series ou liste;
    Dans x, y, il réserve des donnés à tracer;
    
    xlabel, ylabel : str;
    Ce sont les noms des étiquettes des axes;
    
    xlim, ylim : tuple de floats;
    En limitant le domaine des axis;
    
    plot_kwargs : dictionnaire;
    Il comprend la couleur à utiliser pour représenter l'histogramme, la taille et la forme de points sur la figure;

    Returns :
    --------
    Un diagramme en points en cadrant des points par un triangle,
    avec le nom des étiquettes des axes et avec la couleur, taille, forme choisies pour représenter l'histogramme;
    """
    lambda1 = []
    lambda2 = []
    plt.figure(figsize = (8, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot([1/3, 1/2, 1/2, 1/3], [1/3, 1/4, 1/2, 1/3], 'r,--')
    
    
#     for i in np.array(df.index):
#         lambda1.append(x[i])
#         lambda2.append(y[i])
#     # On transfert les données de data.séries à numpy.array;
    plt.scatter(x, y, c = plot_kwargs['color'], marker = plot_kwargs['marker'], alpha = 0.5)
    plt.show()

# %%
correlation_plot3(df['lambda1'], df['lambda2'], 
                    xlabel = 'lambda1', ylabel = 'lambda2',
                    xlim = (0.2, 0.55), ylim = (0.2, 0.55),
                    plot_kwargs={'marker': '.', 'color': 'black'})


# %% [markdown]
# ### Affichage de tous les plots des colonnes
#
# Écrire une fonction `plot2D` qui prend en argument une dataframe et qui affiche 
# * les histogrammes des colonnes
# * les plots des corrélations des couples des colonnes  
# n'affichez qu'une seule fois les corrélations par couple de colonnes
#
# avec `plot2D(df[['radius1', 'lambda1', 'lambda2']])` vous devriez obtenir quelque chose comme ceci
#
# <img src="media/defects-plot2d.png" width="200px">

# %%
# votre code ici
def plot2D (df):
    """
    Cette fonction plot2D trace une série de figures, soit en barres, soit en points, 
    illustrant la distribution de valeurs d'une colonne et le degré de similarité entre des colonnes de dataframe,
    et on va afficher le nom des étiquettes des axes et on va fixer l'aspect des points tracés (couleur, taille, forme, ...);
    
    Paramètres :
    ----------
    df : dataframe;
    Il comprend tous les colonnes des donnés à tracer;

    Returns :
    --------
    6 diagrammes soit en points, soit en barres,avec le nom des étiquettes des axes;
    """
    # On utilise directement des fonctions bien construites avant; 
    histogram2(df['radius1'], nbins=50, xlabel='radius1', ylabel='Frequency', hist_kwargs={'color': 'b'})
    histogram2(df['lambda1'], nbins=50, xlabel='lambda1', ylabel='Frequency', hist_kwargs={'color': 'b'})
    histogram2(df['lambda2'], nbins=50, xlabel='lambda2', ylabel='Frequency', hist_kwargs={'color': 'b'})
    
    correlation_plot2(df['lambda1'], df['radius1'], xlabel='lambda1', ylabel='radius1', plot_kwargs={'marker': '.', 'color': 'black'})
    correlation_plot2(df['lambda1'], df['lambda2'], xlabel='lambda1', ylabel='lambda2', plot_kwargs={'marker': '.', 'color': 'black'})
    correlation_plot2(df['lambda2'], df['radius1'], xlabel='lambda2', ylabel='radius1', plot_kwargs={'marker': '.', 'color': 'black'})


# %%
# pour corriger
plot2D(df[['radius1', 'lambda1', 'lambda2']])


# %% [markdown]
# #### *Bonus++* (dataviz expert-level) le tableau des plots des colonnes
#
# Écrire une fonction `scatter_matrix` qui prend une dataframe en argument et affiche un tableau de graphes avec
# * sur la diagonale les histogrammes des colonnes
# * dans les autres positions les plots des corrélations des couples des colonnes
#
# avec `scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'fc': 'g'})` vous devriez obtenir à peu près ceci
#
# <img src="media/defects-matrix.png" width="500px">

# %%
# votre code ici
def scatter_matrix(df, nbins, hist_kwargs):
    """
    Cette fonction scatter_matrix trace une matrice de figures, soit en barres, soit en points, 
    illustrant la distribution de valeurs d'une colonne et le degré de similarité entre des colonnes de dataframe,
    et on va afficher le nom des étiquettes des axes et on va fixer l'aspect des points tracés (couleur, taille, forme, ...);
    
    Paramètres :
    ----------
    df : dataframe;
    Il comprend tous les colonnes des donnés à tracer;
    
    nbins : int ou séquence ou str (default : 10);
    Si nbins est un entier, il définit le nombre de cases de largeur égale sur l'axe;
    Si nbins est une séquence, elle définit les bords de la case, 
    y compris le bord gauche de la première case et le bord droit de la dernière case;
    
    hist_kwargs : dictionnaire;
    Les couleurs à utiliser pour représenter l'histogramme;

    Returns :
    --------
    Une matrice de 9 diagrammes soit en points, soit en barres,avec le nom des étiquettes des axes précisés ;
    """
    
    plt.figure(figsize = (10, 10))
    # On trace la matrice de figures en utilisant la fonction plt.subplot;
    plt.subplot(3, 3, 1)
    # plt.figure(figsize=(2,2))
    plt.hist(df['radius1'], bins = nbins, color = hist_kwargs['fc'])
    
    plt.subplot(3, 3, 2)
    plt.scatter(df['lambda1'], df['radius1'], c = 'black', s = 0.05)
    
    plt.subplot(3, 3, 3)
    plt.scatter(df['b2'], df['radius1'], c = 'black', s = 0.05)
    
    plt.subplot(3, 3, 4)
    plt.scatter(df['radius1'], df['lambda1'], c = 'black', s = 0.02)
    
    plt.subplot(3, 3, 5)
    plt.hist(df['lambda1'], bins = nbins, color = hist_kwargs['fc'])
    
    plt.subplot(3, 3, 6)
    plt.scatter(df['b2'], df['lambda1'], c = 'black', s = 0.02)
    
    plt.subplot(3, 3, 7)
    plt.scatter(df['radius1'], df['b2'], c = 'black', s = 0.02)
    
    plt.subplot(3, 3, 8)
    plt.scatter(df['lambda1'], df['b2'], c = 'black', s = 0.02)
    
    plt.subplot(3, 3, 9)
    plt.hist(df['b2'], bins = nbins, color = hist_kwargs['fc'])
    
    plt.show()


# %%
# pour vérifier 
scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'fc': 'g'})

# %% [markdown]
# ### Corrélations entre les données
#
# Utilisez les fonctions que vous venez d'implémenter pour rechercher (visuellement) les meilleures correlations qui ressortent entre les différentes caractéristiques morphologiques.
#
# Plottez la corrélation qui vous semble la plus frappante (i.e. la plus informative), motivez votre choix.
#

# %%
# votre code ici
# Après quelques essaies, on trouve une relation 'linéaire' pour la corrélation entre lambda2 et b2;
correlation_plot(df['lambda2'], df['b2'])
# Cela signifie que les caractères lambda2 et b2 ont des relations liées, elles ont des informations répétées,
# en fait, b2 représente une information sur la boite englobante du défaut dans son repère principal d'inertie,
# et lambda2 représente une information sur la matrice d'inertie de chaque défaut;

# %% [markdown]
# ## Analyse en composantes principales (ACP)
#
# Les corrélations entre variables mises en évidence précédemment nous indiquent que certaines informations, apportées par les indicateurs, sont parfois redondantes.
#
# L'analyse en composantes principales est une méthode qui va permettre de construire un jeu de *composantes principales* qui sont des combinaisons linéaires des caratéristiques. Ces composantes principales sont indépendantes les unes des autres. Ce type d'analyse est généralement mené pour réduire la dimension d'un problème.
#
# La construction des composantes principales repose sur l'analyse aux valeurs propres d'une matrice indiquant les niveaux de corrélations entre les caractéristiques. En notant $X$ la matrice contenant nos données qui est donc constituée ici de 4040 lignes et 9 colonnes, la matrice de corrélation des caractéristiques demandée ici est $C = X^TX$.

# %% [markdown]
# ### Construction des composantes principales sur les caractéristiques morphologiques
#
# Construisez une matrice des niveaux de corrélation des caractéristiques. Elle doit être carrée de taille 9x9.

# %%
# votre code ici
# # On crée une matrice de forme (4040, 9);
# x = np.ndarray([4040,9])
# # On remplie cette matrice en parcourant des données dans le dataframe;
# for i in range(4040):
#     for j in range(9):
#         x[i][j] = df.iloc[i, j]

# %%
x = df.to_numpy()

# %%
# On construit la matrice des niveaux de corrélation des caractéristiques, et on vérifie sa taille;
c = np.dot(x.transpose(), x)
np.shape(c)

# %%
# On affiche cette matrice caractéristique;
print(c)

# %% [markdown]
# ### Calcul des vecteur propres et valeurs propres de la matrice de corrélation

# %% [markdown]
# Calculez à l'aide du module `numpy.linalg` les valeurs propres et les vecteurs propres de la matrice $C$. Cette dernière est symétrique définie positive par construction, toutes ses valeurs propres sont strictement positives.

# %%
# votre code ici
val, vec = np.linalg.eig(c)

# %%
# On construit les valeurs propres et les vecteurs propres de la matrice C, et on vérifie leurs tailles;
np.shape(val), np.shape(vec.transpose()[0])

# %%
val

# %%
# On vérifie que toutes les valeurs propres sont positives;
val >= 0

# %% [markdown]
# ### Tracé des valeurs propres

# %% [markdown]
# Tracez les différentes valeurs propres calculées en utilisant un axe logarithmique.

# %%
# votre code ici
plt.plot(np.log(val))

# %% [markdown]
# ### Analyse de l'importance relative des composantes principales

# %% [markdown]
# Vous devriez constater que les valeurs propres décroissent vite. Cette décroissance traduit l'importance relative des composantes principales.
#
# Dans le cas où les valeurs propres sont ordonnées de la plus grande à la plus petite ($\forall (i,j) \in \{1, \ldots, N\}^2, i>j
#  \Rightarrow \lambda_i \leq \lambda_j$), tracez l'évolution de la quantité suivante:
# \begin{equation*}
#  \alpha_i = \frac{\sum\limits_{j=1}^{j=i}\lambda_j}{\sum\limits_{j=1}^{j=N}\lambda_j}
# \end{equation*}
#
# $\alpha_i$ peut être interprété comme *la part d'information du jeu de données initial contenu dans les $i$ premières composantes principales*.

# %%
# votre code ici
# On somme d'abord toutes les valeurs propres;
s = 0
somme = sum(val)

alpha = []
for i in range(9):
    s += val[i]
    alpha.append(s/somme)

# %%
plt.plot(alpha)

# %% [markdown]
# ### Quantité d'information contenue par la plus grande composante principale
#
# Affichez la plus grande valeur propre et le vecteur propre correspondant (ça doit correspondre à la première composante principale).
#
# Quelle est la quantité d'information contenue par cette composante ?
#
# Pratiquement toute l'information ! C'est trop beau pour être vrai non ? 
#
# Affichez les coefficients de cette composante principale. Que remarquez vous ? (*hint* cherchez la caractéristique dont le coefficient est le plus important en valeur absolue) 
#
# En observant les données correspondant à cette caractéristique, avez-vous une idée de ce qui s'est passé ? 

# %%
# votre code ici

# %%
val, vec = np.linalg.eig(c)
vec = vec.transpose()

# %%
# On affiche la plus grande valeur propre et le vecteur propre correspondant;
indx = np.argmax(val)
print('Index : ',indx, '\nValeur: ', val[indx], '\nVector: ', vec[indx])

# %%
# On vérifie la relation entre la plus grande valeur propre et le vecteur propre correspondant;
# C.vec = val.vec
np.dot(c, vec[indx]) - val[indx]*vec[indx]

# %%
# On affiche la valeur absolure de ce vecteur propre, qui représente les coefficients de cette composante principale;
abs(vec[indx])

# %%
for i in range(9):
    print('la plus importante caractéristique dans',i,'ième composante principale est',df.columns[np.argmax(np.abs(vec[indx+i]))])

# %% [markdown]
# ## ACP sur les caractéristiques standardisées
#
# Dans la section précédente, la première composante principale ne prenait en compte que la caractéristique de plus grande variance. Un moyen de s'affranchir de ce problème consiste à **standardiser** les données. Pour un échantillon $Y$ de taille $N$, la variable standardisée correspondante est $Y_{std}=(Y-\bar{Y})/\sigma(Y)$ où $\bar{Y}$ est la moyenne empirique de l'échantillon et $\sigma(Y)$ son écart type empirique. 
#
# **Notez que** dans notre cas, il faut réaliser la standardisation **caractéristique par caractéristique** (soit colonne par colonne). Si vous n'y avez pas encore pensé, refaites un petit tour sur le cours d'agrégation pour faire ça de manière super efficace ! ;) 
#
# Menez la même étude que précédement (i.e. à partir de la section `Analyse en composantes principales`) jusqu'à tracer l'évolution des $\alpha_i$.

# %%
# votre code ici

# %%
# On d'abord construit une liste vide qui va contenir des 9 colonnes standardisées;
dataframe = []

# On standardise des colonnes en soustrayant des moyens et divisant par des écarts;
for cara in df.columns:
    values = np.array(((df.loc[:, cara] - df.describe().loc['mean', cara])/df.describe().loc['std', cara]))
    dataframe.append(pd.DataFrame(data  = {'id': df.index, cara : values}))

# %%
# On ajoute des éléments dans la liste 'dataframe' dans le df standardisé;
df_std = dataframe[0]
for i in range(len(df.columns)):
    df_std = df_std.merge(dataframe[i])
# On règle 'id' comme l'index de dataframe standardisé;
df_std = df_std.set_index('id')

# %%
# On affiche des premières ligne de ce df standardisé;
df_std.head()

# %%
# # On crée une matrice de forme (4040, 9);
# x_std = np.ndarray([4040,9])

# # On remplie cette matrice en parcourant des données dans le dataframe;
# for i in range(4040):
#     for j in range(9):
#         x_std[i][j] = df_std.iloc[i, j]

# %%
x_std = df_std.to_numpy()

# %%
# On construit la matrice des niveaux de corrélation des caractéristiques, et on vérifie sa taille;
c_std = np.dot(x_std.transpose(), x_std)
np.shape(c_std)

# %%
# On construit les valeurs propres et les vecteurs propres de la matrice C, et on vérifie leurs tailles;
val_std, vec_std = np.linalg.eig(c_std)
vec_std = vec_std.transpose()
val_std

# %%
# On fait mise en ordre de la liste de valeurs propres, et on affiche des valeurs;
element = val_std[6]
val_std1 = np.delete(val_std, 6)
val_std1 = np.append(val_std1,element)
val_std1

# %%
# On vérifie que toutes les valeurs propres sont positives;
val_std >= 0

# %% [markdown]
# ### Tracé des valeurs propres

# %%
# votre code ici
plt.plot(np.log(val_std1))

# %% [markdown]
# ### Analyse de l'importance relative des composantes principales

# %% [markdown]
# Vous devriez constater que les valeurs propres décroissent vite. Cette décroissance traduit l'importance relative des composantes principales.
#
# Dans le cas où les valeurs propres sont ordonnées de la plus grande à la plus petite ($\forall (i,j) \in \{1, \ldots, N\}^2, i>j
#  \Rightarrow \lambda_i \leq \lambda_j$), tracez l'évolution de la quantité suivante:
# \begin{equation*}
#  \alpha_i = \frac{\sum\limits_{j=1}^{j=i}\lambda_j}{\sum\limits_{j=1}^{j=N}\lambda_j}
# \end{equation*}
#
# $\alpha_i$ peut être interprété comme *la part d'information du jeu de données initial contenu dans les $i$ premières composantes principales*.

# %%
# votre code ici
s = 0
alpha = []
somme = sum(val_std1)

for i in range(9):
    s += val_std1[i]
    alpha.append(s/somme)

plt.plot(alpha)

# %% [markdown]
# ### Importance des composantes principales
#
# Quelle part d'information est contenue dans les 3 premières composantes principales ? 
#

# %%
# votre code ici
# On affiche les 3 premières composantes principales dont valeurs propres sont les 3 premières grandes;
# Donc ce sont juste les 3 premiers vecteurs dans la liste;
vec_std[0:3]

# %% [markdown]
# Cette part d'information est satisfaisante car nous avons tout de même réduit notre dimension de 9 à 3.

# %%
# On s'intéresse à combien de pourcent de l'information caractéristique est compris dans les trois premières composantes principales;
alpha[2]

# %% [markdown]
# ### Projection dans la base des composantes principales de nos 4040 défauts
#
# On va convertir les données initiales dans la base des (vecteurs propres des) composantes principales, et les projeter sur l'espace engendré par les premiers vecteurs propres.
#
# Faites attention, vous calculez désormais dans les données standardisées.
#
# Créez une nouvelle dataframe dont les colonnes correspondent aux projections sur le sous-espace des 3 vecteurs propres prépondérants; on appellera ses colonnes P1, P2 et P3

# %%
# votre code
base_composantes = vec_std[0:3].transpose()
base_composantes

# %%
# On construit le nouveau df en multipliant la base des composantes principales;
df_new = pd.DataFrame(np.dot(df_std.to_numpy(), base_composantes), columns = ['P1', 'P2', 'P3'], index = df.index)
df_new

# %% [markdown]
# Tracez les nuages de points correspondants dans les plans (P1, P2) et (P1, P3). 

# %%
# On continue à utiliser des fonctions définies avant;
correlation_plot(df_new['P1'], df_new['P2'])

# %%
correlation_plot(df_new['P1'], df_new['P3'])

# %% [markdown]
# ## La conclusion
#
# Reprenez maintenant le défaut `4022` et cherchez son plus proche voisin en utilisant la distance euclidienne dans l'espace des composantes principales. Que constatez-vous ? 

# %%
# %matplotlib notebook
from importlib import reload
reload(plt)
from utilities import plot_defect
from importlib import reload

# %%
# Votre code ici
# On refait le même processus avant pour trouver le défaut avec la distance euclidienne la plus proche dans cet espace;
id_proche = 1
dis_proche = 0
for col in np.array(df_new.columns):
    dis_proche += (df_new.loc[4022][col] - df_new.loc[id_proche][col])**2
    
for i in np.array(df_new.index):
    dis = 0
    for col in np.array(df_new.columns):
        dis += (df_new.loc[4022][col] - df_new.loc[i][col])**2
    if (dis < dis_proche) and (i != 4022):
        id_proche = i
        dis_proche = dis


# %%
id_proche, dis_proche

# %%
print(df_new.loc[id_proche], '\n')
print(df_new.loc[4022])

# %%
plot_defect(id_proche)

# %%
# %matplotlib inline
reload(plt)

# %% [markdown]
# **Une note de fin :** L'objectif de la démarche n'est pas seulement de trouver le plus proche voisin, mais l'ensemble des voisins (recherchez `triangulation de Delaunay` ou `tesselation de Voronoï` si vous être curieux). Or bien que les algorithmes de construction des triangulations/tessellations sont écrits en dimension quelconque, ils sont beaucoup plus efficaces en dimension faible. Dans le cas actuel, une triangulation en dimension 3 est instantanée (avec `scipy.spatial.Delaunay`) alors qu'elle met au moins une heure en dimension 9 (peut-être beaucoup plus, j'ai dû couper car mon train arrivait à destination...)

# %% [markdown]
# ***

# énoncé

Je pars donc d'un réseau donné (Bourges) i.e. des positions d'éoliennes, de villages. Je connais la consommation des villages et la production des éoliennes.
Si je veux trouver une solution optimale, il faut que je puisse considérer tout les états du système, mais les éoliennes produisent une quantité variable d'énergie, alors comment faire ?

## 1ere approche : 
considérer que les pertes moyennes sont les pertes dans le réseau ou on a la moyenne de chaque production
On montre en évaluant dans un système dont on connait les pertes en moyenne sur chaque scénario est bien plus grande donc mauvaise approche

## 2eme approche :
on discrétise la fonction de génération, on obtiens, pour chaque éolienne, entre 3 et 4 valeurs de génération possibles auxquelles on associe une probabilité.
En évaluant sur le meme système que précédemment, les pertes sont semblables

Dans notre approche, la solution optimale est donc celle qui présente le moins de perte sur l'ensemble des scénarios.
Cela implique qu'on doit évaluer chaque solution sur chaque scénario, par ex, si on a 50 éoliennes toutes avec 4 valeurs de gen possibles, on a 50⁴ scénarios a confronter à chaque candidats, c'est beaucoup trop.

On va donc faire l'hypothèse qu'un groupe d'éoliennes toutes géographiquement proches génèrent une quantité proche d'énergie, donc pour chaque 'cluster' d'éoliennes, on associe une unique valeur de production.
Procéder ainsi nous permet d'augmenter considérablement la taille de notre simulation sans ajouter un nombre exponentiel de scénarios.

Ensuite, considérant le développement d'infrastructures comme les smart-grids, est-ce qu'on pourrait, au lieu de trouver une solution bonne partout, on pourrait pas prendre un set de solutions chacune répondant à un ensemble de scénarios ?
Les pertes  moyennes sur l'ensemble des scénarios est-elle plus basse que précédemment ?

# Trouver la solution
Maintenant qu'on sait comment on définit notre meilleure solution, on va la trouver comment ?
De manière générale, lorsqu'il s'agit d'un problème de reconfiguration de réseau (NRP/DRP), la littérature montre que les algos génétique sont prévalent.
En effet, ce type de problème est NP-DIFFICILE (Preuve dans https://www.sciencedirect.com/topics/computer-science/reconfiguration-problem), c'est donc la piste qu'on va préferer.
Dans le papier ici : https://www.sciencedirect.com/science/article/pii/S0305054817302526?fr=RR-2&ref=pdf_download&rr=a38e35b8ca37ebb0, on va utiliser le BRKGA qui, 
au dela de présenter de meilleurs résultats que d'autres algos génétiques, il nous permet d'obtenir à chaque génération des solutions valides : des arbres, 
ce qui évite de devoir les corriger, et donc de gagner du temps de calcul.

# DREAMCRAFT - un world model visuel de Minecraft

> **Projet de recherche en cours (work in progress).**
> Le Monde 1 (prédire) et le choix hors ligne du Monde 2 sont faits et mesurés. Un agent joue maintenant au vrai jeu : dans une arène où tous les joueurs reçoivent la même salle, **il casse près de deux fois plus de blocs qu'un script écrit à la main** (section [En jeu](#en-jeu--un-agent-qui-apprend-de-ses-propres-erreurs)). En forêt, la même méthode coupe près de trois fois plus de bois que le script. Ce gain vient d'un détecteur appris, pas encore du world model. Résultats intermédiaires, pistes négatives comprises. Dernière mise à jour : 11 novembre 2026.

Un modèle qui **rêve la suite d'une partie** : à partir des dernières images et d'une action (« j'avance », « je tape »), il prédit les images suivantes. L'objectif final est qu'il se serve de ces rêves pour **décider seul** de ses actions.

<p align="center">
  <img src="docs/figures/reve_v4_avancer_taper.gif" width="620" alt="Réel à gauche, rêve du modèle à droite, sur 64 pas">
  <br><em>À gauche l'image réelle de départ, à droite le rêve du modèle sous l'action « avancer + taper ». Plus flou que le réel dès les premiers pas, il se dégrade avec l'horizon : c'est tout l'enjeu de la portée.</em>
</p>

## La question de recherche

Un world model qui ne fait que rêver ne sert à rien : il doit rêver **pour choisir**. Le projet est donc construit en deux étages :

1. **Monde 1, prédire** : `(images passées, action) → image suivante`, puis en boucle sur plusieurs pas.
2. **Monde 2, choisir** : rêver plusieurs plans candidats, noter le résultat de chacun, jouer le meilleur, recommencer.

Le levier central est la **portée `H`** : combien de pas le modèle peut rêver avant que le rêve ne devienne inutilisable pour décider.

Contraintes assumées : un seul GPU grand public (RTX 4070, 8 Go), un petit modèle (3,65 M de paramètres pour le modèle de référence) et une **évaluation qui refuse de se laisser tromper**.

## Résultats à ce stade

### 1. La « timidité » du rêve semble venir des données, pas du modèle

Le rêveur refusait souvent de casser le bloc visé : la scène restait figée. Pendant deux semaines, on a exploré les leviers de calcul (capacité, résolution 96 px, loss dédiée) sans rien gagner (plancher ~67 %). La variable qui a fait bouger le résultat, c'est **la fréquence de l'événement dans les données d'entraînement**. C'est une hypothèse forte, pas une loi : elle repose sur deux niveaux de densité, et la densité est confondue avec le domaine.

<p align="center"><img src="docs/figures/figure_timidite_donnees.png" width="620" alt="Timidité selon le domaine et la taille du modèle"></p>

Avec les mêmes 3,65 M de paramètres, nourrir le modèle de vidéos de minage (domaine dense) fait passer la timidité de 66 % à **47 %**. Un modèle 7× plus gros n'apporte rien de plus (49 %). Là où le minage reste rare dans les données (grottes), tous les modèles restent timides (~80 %).

<p align="center"><img src="docs/figures/figure_engagement.png" width="760" alt="Courbe d'engagement réel / timide / courageux"></p>

Sur des épisodes exclus de l'entraînement (mais découpés dans les mêmes vidéos, voir Méthode), le modèle nourri en domaine dense rattrape **44 %** de l'écart avec la réalité, contre **15 %** hors domaine. Tous les rêves restent en dessous de la réalité : les bonnes données divisent l'écart par deux environ, elles ne le suppriment pas.

### 2. L'effet de l'action s'efface avec l'horizon

Pour chaque départ, on fait rêver le modèle deux fois depuis la même scène : une fois en tapant, une fois avec le même plan sans taper. On mesure alors quelle part de l'effet réel le rêve attribue au bouton.

<p align="center"><img src="docs/figures/figure_portee_action.png" width="700" alt="Part de l'effet attribuée au bouton selon la portée"></p>

En domaine connu, cette part passe de 62 % à 4 pas, à 22 % à 16 pas, puis **change de signe à 24 pas** : la dérive du rêve efface l'effet de l'action. La fenêtre exploitable pour décider est donc **H ≤ 12**.

### 3. Le choix hors ligne : un signal, pas encore une preuve

Le chooseur rêve plusieurs plans, note chaque rêve avec un détecteur appris de « bloc cassé » et joue le meilleur. Une loss qui pénalise les rêves figés fait passer le bon choix en grotte claire de 63 % à **77 %** à H=8 (hasard : 33 %).

Ce chiffre ne compte que les scènes où taper est utile : un agent qui taperait toujours ferait 100 %. La mesure à retenir est l'AUC d'affordance, sa capacité à séparer les scènes utiles des autres : 0,70 à 0,77 selon les épisodes, et elle baisse avec l'horizon. Elle n'a pas encore été comparée à une baseline sans modèle.

### 4. Une bonne partie du gain sur la copie, un flou l'obtient aussi

Le rêve bat la copie nette de la dernière image. Mais en MSE, flouter coûte moins cher que se tromper de position : une copie floutée, sans aucun modèle, est donc une référence à battre.

| 4 épisodes, modèle v4, erreur Δ-region | 8 à 24 pas | 48 à 80 pas |
|---|---|---|
| part du gain sur la copie nette déjà captée par le meilleur flou | 46 à 85 % | 64 à 70 % (3 épisodes) |
| marge du rêve sur le meilleur flou | **+7 à +24 %** | **-5 à +21 %** |

Le meilleur flou est choisi après coup parmi quatre, ce qui avantage la référence. Le rêve apporte donc quelque chose, mais moins que ne le suggère la comparaison à la copie nette. Son énergie haute fréquence ne vaut que 0,3 à 0,7 de celle du réel, et ce dès le premier pas.

La portée de 69 pas donnée dans le journal est à lire avec prudence. Elle est mesurée contre la copie nette, et elle est instable : pour le même modèle et le même épisode, elle vaut 69 pas à H=128 et plus de 96 pas à H=96, car les départs tirés changent. Cet épisode sert en plus à sélectionner les checkpoints.

## Où ça bloque aujourd'hui

La question qui compte pour un agent est : *« si je tape ici, est-ce que ça servira à quelque chose ? »*. Sur des épisodes exclus de l'entraînement, on la pose au rêve et on la compare à des baselines **sans aucun modèle** :

| à apparence égale du bloc visé | AUC (50 % = hasard) |
|---|---|
| mesure simple sur les images déjà vues, sans modèle | 63,3 % |
| rêve du modèle | 64,6 % |
| effet causal du bouton dans le rêve | **46,1 %** |

*(Épisodes de grottes, hors domaine. En domaine strip, ces trois méthodes tombent à 45-53 %.)*

Le rêve ne fait donc qu'un point de mieux qu'une mesure triviale, et son estimation causale ne montre **aucun signal** : sans stratification elle vaut 47,6 %, avec un intervalle de confiance à 95 % de [41,7 ; 53,5] qui contient le hasard. Autrement dit : **en moyenne, le modèle attribue un effet au bouton à court terme (section 2), mais il ne sait pas dire où ce bouton servira.** Un premier chiffre brut à 84 % s'est révélé être un artefact de texture une fois le confondeur neutralisé, d'où la stratification systématique.

Deux limites de ce test. L'intervalle est un bootstrap sur les coups, donc optimiste, car il ignore la dépendance à l'intérieur d'un épisode. Et le label « le bloc visé a bougé » est en partie pollué par la caméra : son amplitude seule atteint 72 % d'AUC stratifiée sur ces épisodes. Sur le quart des coups où la caméra bouge le moins, l'estimation causale reste compatible avec le hasard (53,2 %, [40,8 ; 66,1]).

D'où l'étape suivante : jouer au vrai jeu, où l'on connaît exactement les touches envoyées et où le jeu lui-même dit ce qui a été cassé.

## En jeu : un agent qui apprend de ses propres erreurs

L'agent joue à Minecraft 1.19.4 comme un humain : capture d'écran, touches et souris injectées, 20 décisions par seconde. Deux outils rendent la mesure honnête :

- **La vérité du jeu.** Un datapack écrit dans le journal du jeu, à chaque tick, le bloc visé, sa distance et chaque bloc réellement cassé. On ne devine plus un « bloc cassé » à partir des pixels : on le lit. Vérifié sur 5 208 coups, les temps de cassage mesurés correspondent aux règles du jeu.
- **Une arène à salles identiques.** À chaque manche, une salle est creusée sous terre puis copiée, et remise à l'identique avant chaque joueur. Tous partent de la même image (écart mesuré : 0 sur 255) et visent le même bloc. Score : blocs réellement cassés en 30 s. L'ordre de passage tourne. Le verdict est un test de signe sur les manches, fixé avant de jouer (p < 0,05).

### Ce que le world model n'a pas réussi

L'agent a d'abord filmé des expériences contrôlées : la même scène, regardée 1,5 s sans rien faire, puis 1,5 s en tapant. Réentraîné sur ces paires, le rêve prédit si un coup va casser un bloc avec une AUC de 55,5 % (42,6 % avant). Une mesure sans modèle, le contraste du bloc visé, fait 75,9 %. Plus de données n'aide pas : avec 168 sessions collectées en une nuit, l'AUC reste entre 40 et 49 %. Dans l'arène, l'agent qui rêve fait à peine mieux que taper sans bouger (4,5 et 3,4 blocs contre 2,9 et 3,0) : il tape presque tout le temps.

### Ce qui marche : un détecteur appris, puis réentraîné sur ses erreurs

Un petit réseau convolutif regarde **une seule image** et répond à la question : « si je tape maintenant, un bloc cassera-t-il dans les 12 ticks ? ». Les étiquettes viennent de la vérité du jeu.

| AUC (50 % = hasard) | détecteur | contraste, sans modèle |
|---|---|---|
| validation, 1 023 coups | **92,9 %** | 71,4 % |
| test, monde jamais vu, 155 coups | **91,1 %** | 70,2 % |
| test, terrain naturel seulement, 93 coups | 85,7 % | 74,5 % |

L'agent `verifie` tape quand le détecteur dit oui ; sinon il tourne la tête et avance. Si 12 ticks de frappe ne changent rien au centre de l'image, il s'approche, puis va chercher ailleurs. Sur 20 manches, il bat le hasard (14 manches sur 17, p = 0,006), mais pas le script `mine_et_balaye` (« tape devant toi, tourne de temps en temps »).

La vérité du jeu montre pourquoi : **68 % de ses frappes visent un bloc hors de portée**, presque toujours entre 4,5 et 8 blocs, juste trop loin. Le détecteur n'avait jamais appris cette limite : ses données venaient d'une sonde qui tape sur place, souvent face à un mur proche.

Le correctif ne touche pas aux règles de l'agent. On réentraîne **le même réseau**, avec le même programme, en ajoutant 2 194 coups joués dans les arènes par ses versions précédentes, étiquetés par le jeu. Sur des coups hors de portée mis de côté, l'ancien détecteur disait « ça casse » 100 % du temps, le nouveau 1 %.

<p align="center"><img src="docs/figures/figure_arene.png" width="820" alt="Arène finale : blocs cassés par joueur et part des frappes dans le vide"></p>

Arène finale, 16 manches sur un terrain neuf, jouées une seule fois après le réentraînement :

| joueur | comment il joue | blocs cassés en 30 s (moyenne) |
|---|---|---|
| `appris` | comme `verifie`, avec le détecteur réentraîné | **30,6** |
| `mine_et_balaye` | script : tape devant lui, tourne de temps en temps | 16,2 |
| `verifie` | tape quand le détecteur d'origine dit oui | 14,1 |
| `hasard` | touches au hasard | 7,8 |

Manche par manche, `appris` casse plus de blocs que `verifie` 15 fois sur 15 (la 16e est une égalité à 0), que le script 15 fois sur 16 et que le hasard 15 fois sur 16. Ses deux seules défaites viennent de la même salle ratée (manche 12).

Les frappes dans le vide passent de 65 % à 28 %, et l'agent casse **près de deux fois plus de blocs que le script** (30,6 contre 16,2).

**Contre-épreuve, dans des salles fermées.** Une seconde arène bouche les grottes autour de chaque salle. `verifie` n'y frappe presque plus trop loin (12 % des frappes) : ses frappes perdues visaient surtout les parois des grottes voisines. L'avance d'`appris` se réduit (33,7 blocs contre 27,5) mais tient : 6 manches sur 6 (p = 0,016). Contre le script, l'écart reste du simple au double (33,7 contre 16,3). Cette arène s'est arrêtée après 6 manches, sur un bug corrigé depuis : à minuit, le jeu change de journal et la vérité s'était figée.

**Limites, à lire avec le résultat :**
- `verifie` et `appris` ont été conçus après avoir vu les arènes précédentes. L'arène finale et la contre-épreuve sont les seuls tests propres.
- Deux salles sur 16 étaient mal construites, les mêmes pour tous : départ coincé dans un bloc (manche 12), salle ouverte sur une grotte inondée (manche 13). Sans elles, `appris` bat `verifie` 14 fois sur 14.
- Le réentraînement coûte un peu sur l'ancienne tâche : 91,0 % contre 92,9 % en validation (écart [-3,4 ; -0,6]), 87,8 % contre 91,1 % en test (écart non significatif).
- Sous terre, une seule tâche (casser des blocs) et un seul outil (pioche en diamant). La forêt, ci-dessous, est une deuxième tâche.
- Le world model ne sert pas à cette décision. Le gain vient de la boucle « prédire, agir, vérifier, réapprendre », appliquée à un détecteur simple.

### En surface : couper du bois

Même méthode, autre tâche : un monde de forêt, une hache en diamant, et la vérité du jeu étendue aux troncs. À chaque manche, l'agent est posé sur la terre ferme, avec au moins un tronc en vue à moins de 8 blocs. Un cube de 25 blocs autour de lui, arbres compris, est copié sous la carte, puis remis à l'identique avant chaque joueur.

Le détecteur est le même réseau, entraîné de zéro par le même programme sur 82 sessions de sonde en forêt, avec une autre question : « si je tape maintenant, un tronc cassera-t-il dans les 12 ticks ? ». Les feuilles et l'herbe, qui cassent aussi à la hache, ne comptent pas.

| AUC (50 % = hasard) | détecteur bois | contraste, sans modèle |
|---|---|---|
| validation, 416 coups | **92,5 %** | 55,4 % |
| test, 260 coups enregistrés après l'entraînement | **94,5 %** | 51,5 % |

L'agent `bucheron` reprend les règles de `verifie` sans les modifier. Seuls changent le détecteur et la recherche du regard : de l'horizon vers le haut, au lieu de l'horizon vers le sol. Arène de 16 manches, règle fixée avant de jouer :

| joueur | troncs cassés en 60 s, total des 16 manches | moyenne par manche |
|---|---|---|
| `bucheron` | **55** | 3,4 |
| `mine_et_balaye` | 20 | 1,25 |
| `hasard` | 12 | 0,75 |
| `taper` | 4 | 0,25 |

`bucheron` bat le script 12 manches sur 14 décisives (p = 0,006), le hasard 13 sur 14 (p = 0,001) et `taper` 15 sur 15. Il coupe **près de trois fois plus de bois que le script**.

**Limites de la forêt :**
- Une seule arène, jouée une fois. Le test du détecteur et l'arène ont lieu dans le même monde que l'apprentissage, à d'autres endroits.
- Les départs sont moins identiques que sous terre : jusqu'à 10,6 sur 255 d'écart entre les images de départ. Sans doute le ciel, les feuilles ou les animaux, non vérifié.
- 33 % des frappes de `bucheron` tombent dans le vide (rien de visé, ou un bloc hors de portée). Sous terre, réentraîner le détecteur sur les coups de l'agent avait corrigé ce défaut. Ce n'est pas encore fait en forêt.

## Méthode

**Données**
- 234 épisodes [VPT](https://github.com/openai/Video-Pre-Training) (OpenAI, Minecraft 1.16.5) : vidéo 20 Hz et touches du joueur humain.
- 87 épisodes YouTube de joueurs en *full brightness* : grottes (1.18) et strip-mining (1.19). Leurs touches sont inférées par un **modèle de dynamique inverse** entraîné sur VPT (F1 = 0,92 sur « taper »).
- Images en 64×64 (et 96×96 pour l'étude de résolution). Environ 30 Go au total, non versionnés.

**Modèle** : encodeur-décodeur convolutif à connexions de saut, contexte de 4 images, action injectée par FiLM au goulot. Entraînement en rollout avec un curriculum K = 1 → 16 pas, baisse du taux d'apprentissage à chaque palier, EMA des poids et gradient checkpointing (VRAM constante quel que soit K).

**Évaluation** (protocole gelé avant tout entraînement, voir [docs/CADRE_DREAMCRAFT.md](docs/CADRE_DREAMCRAFT.md))
- Des **baselines d'abord** : copie de la dernière image, flux optique.
- Une erreur calculée **uniquement sur les pixels qui changent** (Δ-region). La MSE globale récompense un rêve figé, puisque la plupart des pixels ne bougent pas.
- Sélection des checkpoints sur la métrique honnête, jamais sur la loss.
- Épisodes de validation exclus de tout entraînement. **Limite connue** : les épisodes strip viennent tous d'une seule vidéo, et les épisodes de grottes d'une autre. Validation et entraînement partagent donc le même joueur et le même monde, sans indépendance réelle.
- Tests **contrefactuels** (même scène, action différente) et baselines **sans modèle** avec stratification des confondeurs, dont une copie floutée.
- Dix règles du jeu transformées en sondes automatiques (temps de cassage, dégâts de chute, faim…), à partir de 10 ans de pratique de Minecraft.

**En jeu** (Minecraft 1.19.4, Windows) : un monde pour apprendre, un autre pour tester. La vérité vient d'un datapack qui écrit dans le journal du jeu ; le chat est masqué pour ne pas salir les images. Les arènes avancent de 60 blocs à chaque manche : chaque manche est un terrain neuf.

## Journal de recherche

Tout le cheminement est consigné dans [docs/JOURNAL.md](docs/JOURNAL.md) : 26 leçons, **pistes négatives comprises** (gamma simulé, montée en capacité, résolution 96 px, première tentative de données confondue). C'est le document à lire pour comprendre *pourquoi* chaque choix a été fait.

## Organisation du repo

```
src/dreamcraft/
  data/      chargement VPT, encodage des actions, fenêtres de contexte
  models/    prédicteur conv + FiLM, modèle de dynamique inverse
  eval/      métrique Δ-region, baselines (copie, flux optique)
scripts/     entraînement, évaluation, sondes, agent (index : scripts/README.md)
docs/        cadre, journal de recherche, figures
data/        données brutes et converties       (non versionné)
outputs/     checkpoints, logs, rollouts         (non versionné)
```

## Lancer le projet

Testé sous Windows 11, Python 3.11, CUDA 12.6.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe scripts\smoke_test_data.py
```

```bash
.venv\Scripts\python.exe scripts\build_dataset.py 20
```

```bash
.venv\Scripts\python.exe scripts\train_v7.py
```

L'entraîneur principal se configure par variables d'environnement `DC_*` (résolution, capacité, loss, sources de données) ; elles sont décrites dans l'en-tête de [scripts/train_v7.py](scripts/train_v7.py). L'agent se teste sans le jeu :

```bash
.venv\Scripts\python.exe scripts\agent_loop.py --env replay --policy chooser --steps 100
```

En jeu : Minecraft 1.19.4 ouvert sur un monde, fenêtre au premier plan. La vérité s'installe une fois, jeu fermé, avec `scripts\installe_verite.py`.

```bash
.venv\Scripts\python.exe scripts\detecteur.py
```

```bash
.venv\Scripts\python.exe scripts\arene.py --essai
```

```bash
.venv\Scripts\python.exe scripts\arene.py --manches 16 --joueurs appris,verifie,mine_et_balaye,hasard --agent appris
```

En forêt, sur un monde « Biome unique : Forêt » où la vérité est installée (`installe_verite.py dc_foret`) :

```bash
.venv\Scripts\python.exe scripts\foret.py --collecte 80
```

Puis le détecteur bois (`detecteur.py` avec `DC_LABEL=tronc_12`, commande complète dans son en-tête). Les sessions de test et l'arène s'enchaînent dans un seul processus, pour garder la fenêtre du jeu au premier plan :

```bash
.venv\Scripts\python.exe scripts\foret.py --collecte 10 --tag vt_foret --manches 16
```

Basculer sur une autre fenêtre arrête l'arène proprement. Les checkpoints ne sont pas versionnés (trop lourds) : les scripts attendent les modèles de référence dans `outputs/`.

## Statut

| étape | état |
|---|---|
| Pipeline de données (VPT + YouTube + IDM) | fait |
| Monde 1 : prédiction multi-pas, bat la copie et le flou à court terme | fait |
| Timidité expliquée par la couverture de données | fait |
| Monde 2 : choix hors ligne entre plans rêvés | fait |
| Diagnostic du plafond (attribution de l'action) | fait |
| Agent sur le vrai jeu, vérité lue dans le jeu | fait |
| Paires contrefactuelles en jeu, réentraînement du world model | fait : gain réel, mais sous une mesure sans modèle |
| Détecteur appris « ce coup va-t-il casser un bloc ? » | fait : 91,1 % d'AUC sur un monde jamais vu |
| Agent qui réapprend sur ses propres erreurs, arène à salles identiques | fait : bat le script 15 manches sur 16 |
| Deuxième tâche en surface : couper du bois | fait : près de trois fois plus de troncs que le script |
| Remettre le world model dans la décision, buts à plusieurs étapes | prochaine étape |

## Crédits

- Les données d'entraînement viennent du projet [VPT](https://github.com/openai/Video-Pre-Training) d'OpenAI. Elles ne sont pas redistribuées ici : les scripts les téléchargent depuis la source.
- Le GIF et les figures montrent des images d'épisodes VPT (le GIF : `treechop-984393664dfd-20210924-174326`, un épisode jamais vu à l'entraînement).
- Minecraft est la propriété de Microsoft, ce projet n'est affilié ni à Microsoft ni à OpenAI.

## Licence

Le code est sous licence [MIT](LICENSE).

---
**Florent Baumet Dudbout**, élève-ingénieur à Centrale Méditerranée (vision, ML, PyTorch) · [LinkedIn](https://www.linkedin.com/in/florent-baumet-dudbout)

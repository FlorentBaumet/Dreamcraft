# Session en jeu : préparer et lancer

Pendant une collecte, l'agent prend le clavier et la souris. Passer sur une autre fenêtre (Alt+Tab) arrête tout et relâche les touches.

## 1. Le jeu
- Minecraft Java **1.19.4**, sans mods. Le launcher l'installe : Installations, Nouvelle installation, version `release 1.19.4`.
- Options : champ de vision 70, sensibilité par défaut (affichée 100 %), mode **fenêtré** (F11 pour sortir du plein écran). Ne pas redimensionner la fenêtre : sa taille d'ouverture fait 854×480, en 16:9. Aucun réglage n'est à chercher, celui de la résolution ne concerne que le plein écran.

## 2. Deux mondes
Créer deux mondes en survie, triche activée : `dc_train` et `dc_test`, avec des graines différentes. Le monde de test ne sert jamais à l'entraînement : c'est ce qui donne enfin une validation indépendante.

Dans chaque monde, entrer ces commandes une par une :

```text
/difficulty peaceful
/gamerule doDaylightCycle false
/time set noon
/weather clear 999999
/gamerule doImmediateRespawn true
/gamerule keepInventory true
/effect give @s minecraft:night_vision 999999 0 true
/effect give @s minecraft:resistance 999999 255 true
/effect give @s minecraft:fire_resistance 999999 0 true
/effect give @s minecraft:water_breathing 999999 0 true
/gamerule sendCommandFeedback false
/give @s diamond_pickaxe{Unbreakable:1b}
```

Prendre la pioche en main. La vision nocturne reproduit le rendu très lumineux des vidéos d'entraînement.

**Collecter sous terre, dans la pierre.** Une pioche ne va pas plus vite sur la terre : un bloc y casse en 0,75 s ou plus, trop tard pour la mesure à 12 images (0,6 s). En diamant, la pierre casse en 0,3 s. C'est aussi le décor des vidéos de minage de l'entraînement.

Avant chaque session, creuser une salle à un nouvel endroit (une commande par ligne). Le `~60` déplace de 60 blocs à chaque fois, il n'y a donc rien à changer :

```text
/tp @s ~60 30 ~
/fill ~-9 ~ ~-9 ~9 ~3 ~9 air
```

Si de l'eau ou de la lave apparaît dans la salle, recommencer plus loin. Au centre, les murs sont hors de portée (4,5 blocs) : les coups qui les atteignent et ceux qui tombent trop court visent la même texture, ce qui donne des cas négatifs difficiles. Regarder droit devant avant de lancer.

## 3. Vérifier la plomberie

```bash
.venv\Scripts\python.exe scripts\live_smoke.py --clic
```

À vérifier :
- `outputs/live/smoke_vue.png` montre le jeu sans bordure ni barre de titre.
- La ligne `[B]` indique une réaction de la souris.
- `outputs/live/smoke_clic.png` montre le bloc cassé.

## 4. Collecter

**En automatique :** créer le monde (survie, triche activée), le charger, inventaire et menus fermés, puis lancer la commande et cliquer dans le jeu. L'agent entre lui-même les commandes de la section 2, creuse une salle avant chaque session et enchaîne. Le PC reste occupé pendant ce temps, et le presse-papier est écrasé.

```bash
.venv\Scripts\python.exe scripts\agent_loop.py --env live --policy probe --steps 2400 --sessions 7 --tag dc_train
```

```bash
.venv\Scripts\python.exe scripts\agent_loop.py --env live --policy probe --steps 2400 --sessions 5 --tag dc_test
```

**Session par session, à la main :**
Un cycle dure 3,5 s, plus 2 s de marche tous les 6 cycles. 2400 pas durent 2 minutes et donnent environ 31 paires. Après avoir lancé la commande, cliquer dans le jeu dans les 10 s (inventaire et menus fermés), puis ne plus toucher au clavier ni à la souris. Une collecte s'arrête à tout moment avec Alt+Tab : ce qui précède est sauvé.

Objectif de départ : 8 sessions de 2 minutes dans `dc_train` (environ 250 paires) et 5 dans `dc_test` (environ 150 paires), chacune dans une nouvelle salle. Avec 5 sessions de test ou plus, les intervalles de confiance se calculent par session, pas par coup.

```bash
.venv\Scripts\python.exe scripts\agent_loop.py --env live --policy probe --steps 2400 --tag dc_train
```

Relancer pour une nouvelle session, à un autre endroit du monde. Faire de même dans `dc_test` avec `--tag dc_test`.

## 5. Mesurer
La mesure est fixée avant la collecte, dans `scripts/eval_pairs.py`. La référence est le modèle actuel :

```bash
.venv\Scripts\python.exe scripts\eval_pairs.py --ckpt avant=outputs/v13b-indomain/model_best.pt
```

## 6. La vérité du jeu

Le modèle ne voit que l'écran. Pour le noter, un datapack calcule à chaque tick le bloc dans le viseur, sa distance, le bloc cassé et le butin. Il envoie une ligne dans le chat, le jeu la recopie dans `logs/latest.log`, et Python lit ce journal en direct. Chaque image est ainsi annotée.

À lancer une fois, jeu fermé. La taille du chat passe à 0 pour que ces lignes restent hors des images, et l'ancien réglage est gardé dans `options.avant_dreamcraft.txt`.

```bash
.venv\Scripts\python.exe scripts\installe_verite.py
```

Temps de cassage à la pioche en diamant, selon la règle du jeu : `progrès par tick = vitesse de l'outil / dureté / 30`. La table est dans `src/dreamcraft/verite.py`, et `analyse_verite.py` la compare à la mesure.

| bloc | ticks | | bloc | ticks |
|---|---|---|---|---|
| netherrack | 2 | | terre | 15 |
| pierre | 6 | | citrouille | 30 |
| cobblestone | 8 | | planches de chêne | 60 |
| minerais | 12 | | obsidienne | 188 |

## 7. La boucle autonome

Monde `dc_train` ouvert, inventaire et menus fermés. Essai de quelques minutes, qui vérifie la vérité, les salles et l'agent :

```bash
.venv\Scripts\python.exe scripts\autonome.py --essai
```

Puis la boucle. Chaque cycle dure environ 1 h 15 : 24 sessions de collecte, moitié terrain naturel, moitié salles d'un seul bloc, deux entraînements, la mesure sur `dc_test`, puis l'agent contre trois références. Résultats dans `outputs/autonome/journal.jsonl`. Passer sur une autre fenêtre arrête tout, et ce qui est déjà sauvé reste.

```bash
.venv\Scripts\python.exe scripts\autonome.py --heures 8
```

Le programme ne choisit aucun modèle d'après `dc_test`, donc ce test reste valable après plusieurs cycles. Le seuil du label et le sens des références restent appris sur les 8 premières sessions, pour que la mesure ne bouge pas quand les données augmentent.

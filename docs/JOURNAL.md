# Journal de bord - DREAMCRAFT

> Cahier de laboratoire, tenu au fil de l'eau (y compris les pistes négatives). Chaque entrée : ce qu'on a fait, les chiffres, la leçon.

## 2026-06-26 - Cadrage + Phase 0 montée
- Cadre gelé ([CADRE_DREAMCRAFT.md](CADRE_DREAMCRAFT.md)) : prédire (Monde 1) → choisir (Monde 2), levier central = horizon `H`.
- MineRL abandonné (pas installable ici) → **dataset VPT** en HTTP direct. Piège d'URL : les relpaths de l'index contiennent déjà `.mp4`.
- Baselines gelées avant tout entraînement : B0 copie, B1 flux optique, métrique **Δ-region** (pixels qui changent).

## 2026-07-03 matin - Phase 0 : GO
- ConvPredictor 1,64 M (ctx 1, base 32), 53 s d'entraînement, val = épisode jamais vu (même joueur).
- **Δ-region : modèle 0.00397 < B1 0.00664 < B0 0.00931 → GO.** En MSE globale, la copie « gagne » (taxe de flou) - le protocole honnête était la bonne idée.
- Trouvaille données : casser une torche assombrit toute la scène = dynamique de *jeu* qu'aucune baseline ne capture.

## 2026-07-03 après-midi (longue session) - v1/v1.5/v2 : les pièges
| run | recette | corpus | résultat clé |
|---|---|---|---|
| v1 | 1-pas ctx1 | 4 eps | portée honnête 4-5 pas ; gen 0.0151 [ko] |
| v1.5 | 1-pas ctx1 | 25 eps | portée 4-6 pas ; gen 0.0136 [ko] (1 ep de test !) |
| v2 | multi-pas K1→8 | 25 eps | K>1 oscille ; best = epoch K=1 ; portée 4 |
- **Leçon 1** : la MSE globale en rollout récompense l'inertie → métrique de dérive amendée en **Δ-region par pas** (protocole §5).
- **Leçon 2** : early-stopper sur la MSE globale sélectionne l'inertie → sélection sur Δ-rollout@8.
- **Leçon 3** : ~68 % des blobs VPT sont morts ; scan HEAD → carte des vivants (`live_relpaths.json`).

## 2026-07-03 soir (session 4h) - v3 : le bond
- **Recette v3** : contexte **4 frames** (perception du mouvement), base 48 (3,65 M), curriculum K=1→8 avec **LR×0,6 à chaque palier**, sélection honnête Δ-rollout@8. 35 min sur 33 eps.
- Chaque palier K a amélioré la métrique (K=1 : 0.00650 → K=4 : 0.00503 → K=8 : **0.00486**) - le LR-decay a réparé l'instabilité de v2.

| métrique | v1.5 | **v3** |
|---|---|---|
| portée honnête | 4 pas (0,2 s) | **36 pas (1,8 s)** - mesurée à H=64, `outputs/v3-rollout-h64/` |
| gen Δ, treechop-984 | 0.0136 [ko] | **0.0123 ≈ B1 (0.0122)** |
| gen Δ, Player871 | 0.0108 [ok] | **0.0094 [ok] (−31 % vs B1)** |
| gen Δ, Player309 | 0.0055 [ok] | 0.0062 [ok] (−29 % vs B1) |
| contrefactuel (ep jamais vu) | - | caméra 0.035-0.037, taper 0.016, avancer 0.013 (ok) |

- **Leçon 4** : un seul épisode de généralisation = conclusion trompeuse (l'après-midi disait « ne généralise pas » sur 1 ep ; sur 3 : v1.5 gagnait déjà 2/3). Toujours ≥3 sessions.
- **Démos-rêves** (`outputs/demos/v3/`) : marche vers l'horizon et bûcheron sous action constante - cohérents ~8-16 pas, composition ciel/sol tenue jusqu'à h=48.
- Corpus porté à **110 épisodes** (176k frames) ; carte de 236 vivants. v3b (même recette, corpus complet) lancé en fin de session.

## 2026-07-03 fin de soirée - v3b : le corpus complet confirme
- Même recette que v3, corpus **105 eps / 176k frames** (48 min d'entraînement).
- **Portée honnête : 56 pas (2,8 s)** (H=64) · **généralisation : 3/3 victoires nettes** (treechop-984 : 0.0113 < B1 0.0122 ; Player871 : 0.0081, −41 % ; Player309 : 0.0060, −32 %). Sur Player871, bat même B0/B1 en MSE globale.
- ** v3b = modèle de référence** (`outputs/v3b-fullcorpus/`). L'arc du jour : portée 0,2 s → 1,8 s → **2,8 s**.
- Rêves 64 pas (`outputs/demos/v3b/`) : scène cohérente ≥16 pas, composition tenue à 48.

## 2026-07-05 - MONDE 2, marche 1 : les rêves savent choisir (offline)
- **Protocole** (`scripts/monde2_choice.py`) : à un instant réel « actif », rêver H pas sous 4 plans (vrai / caméra inversée / taper basculé / immobile), scorer chaque rêve vs la réalité → le modèle « choisit » le plan au meilleur rêve. Chance = 25 %. 192 fenêtres × 5 horizons × 2 épisodes.
- **Résultat : avec le scoreur honnête (Δ-region), v3b retrouve le vrai plan jusqu'à 77 %** (in-domain, H=4) et reste à 63-71 % sur l'épisode jamais vu. **Avec le scoreur MSE globale : ≈ chance in-domain (20-27 %).**
- **Leçon 5 (majeure)** : le piège de l'inertie frappe aussi la *décision* - la métrique honnête ne sert pas qu'à évaluer, elle **choisit 3× mieux**. Toute la méthodo anti-triche se transfère au Monde 2.
- **Le levier H, version décision** : in-domain, le choix est meilleur à H=4-8 puis se dégrade (77 % → 50 % à H=32) → **spécification du futur chooseur : rêves courts + ré-ancrage fréquent** (receding horizon), exactement mon intuition de départ.
- Breakdown : fenêtres « taper » discriminées à 68-79 % (H court) - le modèle sait ce que taper *fait* ; caméra OK à H court, s'effondre à H=32 hors-domaine (27 %).
- Sorties : `outputs/monde2/` (courbe accuracy_vs_H, visuels choix, results.json).

## 2026-07-05 soir - Axe A : v4 (K=16 + EMA) & leçon d'ops
- **Incident GPU** : le fine-tune v4 et mon `train_vlm.py` (autre projet) se partageaient les 8 Go → VRAM saturée → **les deux runs rampaient sans erreur** (GPU 100 %, zéro epoch - signature du débordement WDDM silencieux). Résolu : v4 arrêté, GPU rendu au VLM, **file d'attente auto** (monitor sur le process) → v4 relancé à la libération. Leçon d'ops : toujours `nvidia-smi --query-compute-apps` avant un job GPU.
- **v4 relancé sur le corpus complet** (230 eps d'entraînement, ~2× v3b) : cumule donc recette (K=16, EMA, sélection @16) ET données - attribution à démêler plus tard si besoin (ablation), priorité aux perfs.

## 2026-07-06 - v4 : nouveau modèle de référence
Fine-tune de v3b (58 min) : curriculum poussé à **K=16**, **EMA** 0.999, sélection @16, corpus complet (438k frames). La métrique de sélection ne montrait que +2,9 % - la batterie révèle le vrai gain :
| métrique | v3b | **v4** |
|---|---|---|
| portée honnête (H=128) | 56 pas (2,8 s) | **69 pas (3,45 s)** |
| portée MSE *globale* | 0 pas (tous modèles) | **34 pas (1,7 s)** - première historique |
| gen Δ 3 sessions | 0.0113 / 0.0081 / 0.0060 | **0.0111 / 0.0080 / 0.0060** (3/3 ≤) |
| choix in-domain H=16 / H=32 | 57 % / 50 % | **65 % / 60 %** (+8-10 pts) |
| choix OOD | 63-71 % | stable (58-68 %) |
- **Leçon 6 : l'Axe A se convertit en Axe B** - apprendre à rêver 16 pas (K=16) fait décider mieux à H=16/32 (+8-10 pts). La jonction rêveur→chooseur est mécanique, pas espérée.
- Leçon 6bis : la métrique de sélection (@16) sous-estimait le gain réel → la batterie complète reste le seul juge.
- Curiosité : in-domain, le scoreur global choisit *sous* la chance (19-22 %) - l'inertie est un anti-signal actif, pas juste du bruit.
- Incident ops résolu en cours de route (GPU partagé avec mon train_vlm.py → file d'attente auto). Démos : `outputs/demos/v4/`.

## 2026-07-06 - MONDE 2, marche 2 : le chooseur but-visible (v1)
- **Protocole** (`scripts/monde2_chooser.py`) : but = « fais changer durablement le centre de ta vue, pas la périphérie » (= casser le bloc visé). Menu de 6 plans constants, rêvés H∈{8,16,24} par v4, score du but calculé SUR les rêves, choix = argmax. Positifs = l'humain tapait ET le centre a réellement changé ; négatifs = pas de taper. Les baselines copie/flux sont structurellement hors-jeu (leur « rêve » ignore l'action → même score pour tous les plans) : cette tâche EXIGE un world model.
- **[ok] En terrain ouvert (Player871, session jamais vue) : le chooseur fonctionne - 77-79 % de bons choix à H=16-24 (chance 33 %)**, et le choix S'AMÉLIORE avec H (52→79 % : un rêve plus long laisse le bloc disparaître → signal plus net). Premières décisions orientées-but validées, out-of-domain.
- **[ko] En grotte (Player129) : choix ≈ chance - mais le coupable est identifié : la fonction de but, pas le rêveur.** Collé à un mur sombre, tous les plans changent « le centre » pareil (scores dans le bruit : 0.015-0.031) ; l'AUC d'affordance reste pourtant à 0.63-0.76 → le signal est dans les rêves, notre détecteur naïf ne l'extrait pas. Pistes : normalisation par le contraste de la scène, ratio centre/total (l'offset par moment ne change pas l'argmax → il faut une normalisation, pas une soustraction), patch croix plus petit, ou détecteur appris.
- **Leçon 7 : les horizons optimaux différent par tâche** - discriminer un plan : H court (4) ; détecter l'accomplissement d'un but : H moyen (16-24). Le futur agent devra mixer les deux.
- AUC affordance 0.70-0.77 partout : le rêve « sait » s'il y a quelque chose à casser devant. Sorties : `outputs/monde2_chooser/`.

### Marche 2.5 - duel de fonctions de but (mêmes rêves, 4 lectures)
| scoreur | forêt OOD (H=16) | grotte (H=8-16) |
|---|---|---|
| diff (v1, 24×24) | 77 % | 28-36 % (≈ chance) |
| ratio (24×24, invariant contraste) | 71 % | 39-42 % |
| diff_croix (12×12) | 88 % | 35-38 % |
| **ratio_croix (12×12 + invariant)** | **90 %** | **42-45 %** |
- ** `ratio_croix` adopté** : viser juste le bloc (12×12) + normaliser par le contraste = +13 pts en forêt, grotte enfin au-dessus de la chance. Confirmations : (a) le patch étroit compte (le bloc visé, pas le quart d'écran), (b) l'invariance au contraste répare partiellement les scènes sombres.
- La grotte reste dure (45 % max) : casser de la pierre sombre dans le noir à 64 px est à la limite du signal pixel. Prochaine vraie marche : **détecteur de but appris** (mini-classifieur « bloc cassé devant » sur frames réelles) ou résolution 96-128 px.
- Sweet spot du chooseur confirmé : **H=16** (90 % forêt ; la croix à H=24 retombe - flou de rêve × petit patch = bruit).

## 2026-07-06 - Front grotte : la piste « full brightness » (mon idée)
- Idée : utiliser des vidéos de joueurs **déjà en full brightness** (grottes claires → signal pixel lisible pour le rêveur ET le but).
- **Test raccourci (gamma post-hoc au scoring) : échec** - choix inchangé, AUC dégradée (0.70→0.61). Éclaircir après coup n'ajoute pas d'info : elle n'est pas dans les rêves sombres. → Le traitement doit être À LA SOURCE : l'idée data de Florent est la bonne direction.
- **Audits du corpus (234 eps)** : 98 eps >20 % de frames sombres ; audit « souterrain clair » → 49 candidats, mais la **vérification visuelle les invalide** (nez-contre-mur en surface, intérieurs de sable, canopée - pas de mine claire). **VPT = luminosité standardisée, pas de full brightness disponible.**
- **Conséquence : la seule source native de grottes claires = nos propres recordings** (l'option « mes enregistrements » de la spec initiale) - outil de capture à construire (écran 20 Hz + log clavier/souris → npz). Complément en cours : v5 « gamma-robuste » (jitter à l'entraînement + entrées éclaircies à l'inférence) = version simulée, verdict imminent.
- Leçon 8 : **vérifier visuellement les sorties d'heuristiques** avant d'y croire (49 « candidats » → 0 réels).

## 2026-07-07 (nuit) - Mon idée « full brightness », validée de bout en bout
La chaîne construite et exécutée dans la nuit :
1. **Recherche YouTube** (yt-dlp ytsearch) → longplay « Caving with Night vision » 2h49, vérifié visuellement (grottes lisibles, luma 0.30 vs 0.05-0.12 en VPT sombre ; bémol : minimap moddée).
2. **IDM entraîné en 3 min** sur nos 437k fenêtres VPT étiquetées : F1 taper **0.92**, caméra ±4.6 px, détection menus 99.8 % (boutons rares faibles - assumé). → pseudo-étiquetage de vidéos sans actions.
3. **Ingestion** : 2h49 → 32 épisodes npz format VPT (2 jetés « menus »), `data/processed_youtube/`.
4. **Test chooseur v4 sur grottes claires** : AUC affordance **record 0.90** (vs 0.70 sombre) et +6-13 pts de choix à H=8 → la luminosité porte le signal (hypothèse confirmée)… mais le choix décroît avec H (rêveur hors-domaine : joueur, palette, minimap inconnus).
5. **v6-lite = v4 fine-tuné 3 epochs sur VPT+30 nvcave** (tests exclus) : **grottes claires 59-63 % (H=8), 51-53 % (H=16), 45-47 % (H=24)** vs v4 51-58/38-47/30-44 → **+8 pts moyens, +13-15 aux horizons profonds.** La familiarité du rêveur avec le domaine se convertit directement en choix.
- Reste sous la forêt (90 %) : le confound géométrique des grottes (mur qui approche = gros changement au centre, taper ou pas) demeure → prochain front = détecteur de but appris.
- **Leçon 9 : le gamma post-hoc échoue, la donnée native réussit** - on ne fabrique pas de l'information, on la collecte. (v5 gamma-simulé reporté, faible priorité désormais.)
- Ops : crash machine (OOM, pic 2× de build_corpus → **pré-allocation partout maintenant**) ; puis crawl VRAM WDDM sur les stages K=16 de v6 (2 stalls, batch 64→48 insuffisant) → v6 arrêté en « lite » (3 epochs) qui suffit au verdict. **Mystère ouvert : pourquoi ce run déborde quand v4 (même forme) passait** - à investiguer de jour (fragmentation ? budget WDDM sous pression RAM 7.6 Go ?).

## 2026-07-07 - Batterie « règles du jeu » v1 (mes 10 règles)
Pipeline « une phrase de joueur → une loi → un test automatique », calibré sur la réalité avant chaque verdict.
| règle | verdict |
|---|---|
| #1 temps de cassage | [ok] respect **96 %** quand le rêve casse (durée 15-17 pas vs 9 réels) - MAIS **81 % des blocs ne cassent jamais** |
| #10 manger→faim | [ok] respect **93-97 %** (contrefactuel apparié : rêves jumeaux manger/pas-manger, la fonte du HUD s'annule dans la paire) |
| #5 chute→dégâts | [partiel] non concluant en tant que règle (réalité 24 % de flash - eau ; n=29) mais **les rêves n'atterrissent que dans 7-10 % des cas** |
| #2 seau |  337 hits minés (~50 % précision visuelle) - sonde à détecteur resserré à écrire |
| #6 mobs / #9 bateau |  4913 / 458 candidats à trier |
| #4 lit | [ko] 1 instance dans 292 épisodes - intestable |
| #3 bulles |  96-128px requis | #7/#8 touche E |  extension espace d'action (E est dans le jsonl VPT, jamais mappé) |
- **Leçon 10 - LA TIMIDITÉ DU RÊVEUR, confirmée par 3 sondes indépendantes** : le modèle connaît les lois d'état (manger→faim (ok), cassé→reste cassé (ok), timing du cassage (ok)) mais **refuse de s'engager sur les transitions d'événement** : 81 % des blocs jamais cassés, 90 % des chutes sans atterrissage. Signature du biais MSE (« dans le doute, rien n'advient »). **→ v7 désigné : entraînement pondéré-événements** (les moments d'engagement, désormais minables automatiquement, sur-pondérés dans la loss).
- Leçon 10bis (méthodo) : détecteurs à calibrer sur la réalité AVANT verdict (2 bugs attrapés ainsi : ancrage HUD du détecteur de chute, fonte contaminant la sonde faim → contrefactuel apparié).

## 2026-07-08 - Session longue : v7 (négatif honnête) → v7b (commit-loss)
- **v7 (sur-échantillonnage d'événements ×50 % du batch) : ÉCHEC PROPRE.** Timidité inchangée sur épisodes clairs vierges (83 % vs 82 % de « jamais cassé ») ; seul le timing s'affûte quand il ose (14 pas vs 17,5, réel : 9). **Leçon 11 : le sur-échantillonnage change le menu, pas le calcul de risque** - le MSE récompense toujours « rien n'advient » à chaque exemple, quelle qu'en soit la fréquence. Le remède doit changer l'objectif.
- ** Leçon 11bis - 3ᵉ récidive du même piège de sélection** : v2 (sélection sur MSE globale → inertie), v5 (sélection en mode normal → poids gamma jetés), v7 (sélection engagement sur val SOMBRE → détecteur aveugle, plancher à 5 %). **Règle d'or désormais : valider la métrique de sélection sur données réelles AVANT d'entraîner** (elle doit voir le phénomène qu'elle est censée récompenser).
- **v7b lancé : la commit-loss** - hinge différentiable : là où la réalité s'est engagée (patch croix parti à >0.10 de la référence), le rêve resté collé (<0.08) paie λ=2. Val engagement déplacé sur épisode clair (nvcave_012, proxy assumé ; tests nvcave_008/026 vierges).
- Détecteurs seau v2→v3 : 3 confounds successifs (glints → torches/planches → épées diamant) → **règles #2/#9 garées : plafond des heuristiques pixel à 64px atteint**, il faudra un détecteur appris. Pool mobs nettoyé des clics d'inventaire (4913 → 1970).

### v7b (commit-loss) : le gain était ailleurs
- Entraînement : engagement val 9→20 % monotone, fidélité clouée sous la garde. Sur épisodes VIERGES : timidité inchangée (83 %) - le gain du val reflétait en partie la mémorisation (nvcave_012 ∈ train, proxy déclaré→confirmé) ; le cassage rêvé est plus RAPIDE (14 vs 17,5 pas), pas plus fréquent.
- **MAIS le chooseur bondit : grottes claires H=8 : 63→77 % (scoreur large : 85 %) ; H=16 : 51→70 %.** Forêt tenue (87-88 %), portée 64/64 pas sans croisement, lois R2-R4 100 %.
- **Leçon 12 - courage ≠ contraste : la commit-loss n'a pas créé l'engagement, elle a affûté le CONTREFACTUEL** (rêve-taper ≠ rêve-pas-taper bien plus nettement) - et c'est ça que le choix consomme. La timidité brute reste un problème ouvert (candidats : λ_commit ↑, plus d'epochs - ça montait encore -, loss adversariale légère).
- ** v7b = nouveau modèle de référence** (`outputs/v7b-commitloss/`) : meilleur sur l'axe décision, neutre partout ailleurs. Le scoreur `ratio` large (24×24) redevient compétitif avec v7b (85 %) - les rêves engagés rendent le grand patch informatif ; à réévaluer comme fonction de but par défaut.

## 2026-07-10 - Reprise « recherche pure » : le front grotte sombre TOMBE
Cap choisi : recherche pure (consolidation et rédaction repoussées).
- **Détecteur de but APPRIS** (`scripts/train_goal_detector.py`, GoalNet 1.75M) : entraîné sur les vrais moments de destruction (2 412 positifs localisés, négatifs durs = minage-sans-cassage, split par épisode, test = 6 eps vierges).
- v1 (AUC 0.79) et v2 (+seuil sombre adaptatif, +gamma jitter : AUC 0.79) : **négatifs au chooseur** (21-57 % partout) - entraîné sur du réel, déployé sur des rêves flous = hors-domaine.
- **v3 : + augmentation FLOU (domaine rêve) + agrégation top-3 → déblocage** : **grotte sombre 45 → 71/77/84 %** (H=8/16/24), grottes claires profondes 48 → 77-89 %. Une ligne d'augmentation a fait ce que l'architecture ne faisait pas.
- **Leçon 13 : l'instrument s'entraîne dans son domaine d'OPÉRATION** - le scoreur note des rêves flous, il doit apprendre sur du flou. (4ᵉ occurrence de la famille « instrument hors-domaine », la première retournée en victoire le jour même.)
- **Ensemble (rangs appris+ratio)** : records absolus en grottes claires (**88-91 %**), dilue là où un parent domine seul. **Portefeuille final : sombre→appris (84 %), clair→ensemble (88-91 %), forêt→heuristiques (88-91 %)** - le chooseur décide à 77-91 % dans TOUS les environnements testés (plancher du matin : 45 %). AUC de l'ensemble non-interprétable (rangs par-moment) - déclaré.
- Suites classées : meta-sélecteur de scoreur par scène (gating trivial luma/ouverture) ; détecteur unifié entraîné sur rêves étiquetés ; même recette pour dégeler #2/#6/#9.
- **v8 (commit-loss λ=4, +5 epochs K=12) : négatif propre** - chooseur identique à v7b (±2 pts), timidité inchangée (84 %), lois intactes. **Le levier commit-loss sature à λ=2.** La timidité demande un changement STRUCTUREL : candidats classés = adversarial léger (un discriminateur « ce rêve est-il figé ? »), résolution 96-128px (le signal d'engagement est minuscule à 64px), architecture. v7b reste la référence rêveur.

## 2026-07-11 - Chantier 96px : hypothèse partiellement confirmée, verrou = la CAPACITÉ
- Ré-ingestion 265/266 eps @96 en réutilisant les actions 64 (résolution-indépendantes) ; refactor résolution-agnostique du pipeline (2 pièges de coordonnées attrapés : HUD 52-64→78-96, buffers de dream - dont un patch silencieusement raté → **règle : assert sur tout .replace()**).
- v9 = v7b warm-starté @96 (transplant brut : engagement s'effondre 20→5 % - l'adaptation de résolution n'est pas gratuite), +10 epochs (OOM dur à batch 32 → **frontière practicable 8 Go : batch 24 à K=8/96px**) → engagement 14 % (plateau).
- **Verdict épisodes vierges : timidité 83→79 % (−4 pts, SEUL levier à l'avoir jamais bougée) MAIS portée 69→32 pas (−54 %).** À capacité constante, la résolution échange l'engagement contre la portée.
- **Leçon 14 : le verrou de la timidité n'est ni la loss (v7/v8) ni la résolution seule (v9) - c'est la CAPACITÉ.** 96px exige base 64+ → gradient checkpointing ou GPU plus gros. 96px reclassé « attend l'échelle » ; **v7b reste la référence**. Bonus dormant : le corpus 96 complet existe (data/processed_96) pour le jour venu.

## 2026-07-12 - L'échelle de capacité (idée : « on peut scaler ») : la courbe est tracée
Gradient checkpointing implémenté (VRAM plate quelle que soit K - validé à batch 48@96px, le double de l'ancien plafond) → l'échelle devient possible sur la 4070.
| modèle | params | timidité (vierges) | portée Δ H=64 | chooseur sombre (appris) | chooseur clair |
|---|---|---|---|---|---|
| v7b | 3,65 M | 83 % | **69 pas** | 71/77/84 | **75-91 %** |
| v10a | 6,4 M | 71 % | 56 pas | - | - |
| v11 | 25,7 M | **69 %** | 36 pas | 70/69/**91** (record H=24) | non, 48-66 % (chute) |
- **Leçon 15 (la plus subtile du projet) : courage, portée et choix ne se maximisent pas ensemble à données constantes.** (a) Le courage est une propriété d'échelle qui SATURE (~70 % à 438k frames - ×4 de capacité n'achète que 2 pts au 2ᵉ barreau) ; (b) la portée DÉCROÎT avec l'échelle (69→56→36) : un rêveur courageux qui casse son bloc un pas « trop tôt » diverge du futur enregistré pour toujours - or le futur enregistré n'est qu'UN échantillon des futurs possibles → **la métrique de portée porte un biais d'inertie structurel**, quantifié de bout en bout (v1 inerte : portée superbe ; v11 courageux : portée courte). Métrique future : portée événementielle / distributionnelle.
- **Le facteur limitant a encore bougé : loss → résolution → capacité → DONNÉES** (la saturation à ~70 % pointe vers les 2,3k événements d'engagement du corpus ; prochain levier courage = données riches en événements : plus de vidéos type nvcave, mes propres enregistrements).
- **Références : v7b garde la couronne SYSTÈME** (meilleur portefeuille chooseur + portée) ; v11 = champion du sombre-profond (91 %) + point d'échelle ; v10a = meilleur compromis courage/portée. Figure : `outputs/courbe_capacite_courage.png`.

## 2026-07-15 - v12 = v11@96px : la grille capacité×résolution est COMPLÈTE (checkpointing → tenu en local)
Le gradient checkpointing (ajouté après l'échec 96px de v9) fait tenir **base128@96px sur la 4070** (6,7/8,2 Go, batch 16) → v12 entraîné en local, **Kaggle évité** (chaîne montée, gardée pour le vrai 16 Go).
| | 64 px | 96 px |
|---|---|---|
| 3,65 M | v7b : **83 %** / portée 69 | v9 : 79 % / 32 |
| 6,4 M | v10a : 71 % / 56 | - |
| 25,7 M | v11 : 69 % / 36 | **v12 : 67 % / 24** |
- **Leçon 16 - la frontière matérielle du rêveur est cartographiée de bout en bout, et elle est plate.** Timidité : plancher **~67-69 %** atteint par TOUTE recette (v12 ne gagne que 2 pts sur v11 malgré ×7 params + ×2,25 pixels). → le courage est bien une limite de **DONNÉES d'événements** (2,3k dans le corpus), pas de compute (confirme leçon 15, définitivement). Portée : capacité et résolution la rognent de façon **cumulative** (v12 = les deux pénalités → portée 24, la pire) - mécanique du biais d'inertie de notre métrique (leçon 1/15).
- **Conclusion opérationnelle : sur cette machine, à ce corpus, la recherche sur le rêveur est épuisée.** Le seul levier restant est plus de données-événements (chaîne YouTube, non-compute). Figure : `outputs/grille_capacite_resolution.png`. v7b reste la référence système.

## 2026-07-15 (suite) - Pari données v13a : négatif CONFONDU
+55 épisodes strip-mining (KXNdUK2qnoI, 10h, IDM-étiquetés, taper médian 81 %) → événements 2304→**7512 (×3,3)**. v13a = recette v7b exacte sur corpus élargi.
- **Résultat : engagement 9→5 % (baisse monotone), fidélité dégradée aussi → model_best = warm-start v6-lite inchangé (hash identique). Data-scaling naïf = pire, pas mieux.**
- **MAIS confondu, à ne pas sur-interpréter** : (a) 69 % des événements viennent maintenant de l'IDM (bruit d'étiquetage ~8 %+, concentré sur le signal qu'on injecte en masse) ; (b) domaine strip (deepslate, torches, overlay texte) ≠ domaine d'éval nvcave → la fidélité qui se dégrade AUSSI signe un décalage de distribution, pas juste un problème de courage. Test propre = seeder les événements depuis les VRAIS labels VPT + données plus proches du domaine nvcave.
- **Leçon 17 : le data-scaling par pseudo-labels IDM + domaine décalé ne transfère pas.** Avec la grille (loss (non), résolution (non), capacité→sature 67 %, data-naïf (non)), le plancher ~67-69 % résiste à TOUS les leviers bon marché testés. La recherche locale sur le rêveur a un dernier essai propre possible (data même-domaine + vrais labels), sinon elle est close.

## 2026-07-15 (suite) - v13b : LE RÉSULTAT - la timidité est un défaut de COUVERTURE DE DONNÉES
Protocole propre (corrige v13a) : entraîner sur strip-mining (dense en événements), tenir 5 eps strip à l'écart, val engagement IN-DOMAIN (strip_050), tester le courage sur strip vierge ET nvcave.
| domaine (densité événements) | v6-lite baseline | **v13b** |
|---|---|---|
| strip vierge (DENSE ~80 % taper) | 66 % timide | **47 % timide** (record projet, base 3,65M) |
| nvcave (RARE ~35 %) | 82 % | 79 % |
- **Contraste −19 pts in-domain, +réaliste** (v13b casse en 9 pas ≈ réel 6,5 ; v6-lite = flashs instantanés 2,5). Robuste au bruit détecteur (même détecteur/événements → comparaison relative valide malgré strip « instantané » 18 %).
- **Leçon 18 (POSITIVE, la thèse du projet) : la timidité n'est pas une fatalité d'architecture - c'est la densité de l'événement dans les données. Même rêveur : COURAGEUX là où l'événement est fréquent (47 %), TIMIDE là où il est rare (79 %).** Réconcilie tout : v7/v8/v9/v11/v12 échouaient car ils gardaient la MÊME couverture ; le levier était les données-événements EN-DOMAINE (pas naïves comme v13a, qui mélangeait domaines).
- GIF timide-vs-courageux (`outputs/gif_courage/`) : contraste visuel faible sur strip sombre 64px (les 2 floutent) - le résultat est CHIFFRÉ, à rendre visuel via (a) v13b scale base128 (moins de flou + timidité encore plus basse ?), (b) domaine de minage plus clair. **v13b = modèle de référence « courage ».**

## 2026-07-15 (fin) - v14 : la thèse cristallise (figure_timidite_donnees.png)
v14 = base128 (25,7M) × strip in-domain (warm v11). Timidité strip **49 %** - quasi = v13b 3,65M (47 %). nvcave 81 % (comme tous).
- **Leçon 19 (LE résultat du projet) : la timidité d'un world model est fixée par la DENSITÉ D'ÉVÉNEMENTS DANS LES DONNÉES, point. Ni la loss, ni la résolution, ni la capacité (3,65M ≈ 25,7M une fois in-domain). Les 4 leviers "compute" contre lesquels on a buté (plancher 67 % sur nvcave) combattaient la mauvaise variable.** Preuve : 2×3 (nvcave/strip × baseline/petit/gros) - l'in-domain fait −19 pts, la capacité fait ±2 pts.
- Histoire complète et contre-intuitive : "on a cru pendant 2 semaines à une limite d'architecture/compute (grille capacité×résolution, leçons 14-16), c'était une limite de couverture de données (leçons 18-19). Un petit modèle bien nourri bat un gros modèle mal nourri."
- Modèle courage de référence : **v13b** (petit, timidité 47 %, durée cassage 9 pas ≈ réel 6,5 ; v14 casse trop vite, 5 pas). Figures : `figure_timidite_donnees.png` + `grille_capacite_resolution.png`. GIF (`gif_courage/`) reste flou (strip sombre 64px) → à polir sur domaine minage CLAIR pour la démo visuelle.

## 2026-07-15 (fin+) - Figure visuelle : la COURBE D'ENGAGEMENT (`figure_engagement.png`)
Le GIF pixel restait flou (mollesse intrinsèque 64px+petit modèle, pas seulement l'obscurité - vetting de 3 vidéos de minage "claires" non concluant). **Bon artefact = la courbe d'engagement** : changement moyen du bloc visé vs pas de rêve, sur épisodes jamais vus, 3 courbes (réel / timide / courageux), 2 panneaux (dense/rare).
- **Domaine DENSE : le courageux SUIT la courbe réelle, le timide traîne** → récupère **44 % de l'écart** d'engagement vers la réalité.
- **Domaine RARE : les 2 modèles superposés**, loin du réel → **15 %** seulement.
- Caveat honnête affiché : tous les rêves sous-estiment la réalité partout ; l'in-domain divise l'écart par ~2, il ne le résout pas.
- **Trio de figures du projet** : `figure_engagement.png` (la thèse, visuelle) + `figure_timidite_donnees.png` (la thèse, chiffrée, capacité-indépendante) + `grille_capacite_resolution.png` (les fausses pistes). Scripts : `figure_engagement.py`, `gif_courage.py`.

## Provenance des données - VERSIONS Minecraft (vérifiée 2026-07-15)
| source | version | rôle |
|---|---|---|
| VPT (234 eps) | **1.16.5** (démos OpenAI) | corpus général |
| nvcave (32 eps) | **1.18.1 Java** (titre vidéo) | grottes claires |
| strip-mining (55 eps) | **1.19** (description vidéo, 2023-05) | **source du courage v13b/v14** |
→ **Pour la marche 3 (jeu live) : lancer 1.19.x** (1.18-1.20 équivalent) = version du domaine qui a produit le modèle courageux. Écarts 1.16→1.19 : génération de grottes, deepslate (1.17), éclairage ; textures identiques depuis 1.14, HUD identique. **Autres réglages critiques pour le match de domaine : FOV 70 (défaut), GUI scale défaut, fenêtre 16:9, zone éclairée aux torches, vanilla strict (0 shader/resource pack/minimap).**

## 2026-09-02 - L'AGENT (« on a atteint la limite du modèle »)

Boucle `scripts/agent_loop.py` : politiques interchangeables × backends (`replay` à vide / `live`).
Validée à vide sur épisode jamais vu : **20,0 Hz tenus, latence de décision 28 ms** (budget 200 ms
à `--replan 4`). La plomberie du live est prête ; reste à lancer le jeu.

Mais la boucle a d'abord servi de **banc de mesure**, et elle a chiffré la limite que je sentais. Trois diagnostics, tous sur épisodes JAMAIS VUS.

**(a) `diag_separation.py --sweep` - la loi de la PORTÉE (mon levier de départ).**
`SEP` = part de l'effet réel que le rêve attribue au **bouton** (rêve en tapant − rêve du même
plan sans taper, rapporté à réel − rêve sans taper).

| modèle / domaine | H=2 | H=4 | H=8 | H=12 | H=16 | H=20 | H=24 |
|---|---|---|---|---|---|---|---|
| v6 timide / strip | 73 % | 16 % | 17 % | 11 % | 6 % | 2 % | −2 % |
| v13b courageux / strip | 133 % | 62 % | **56 %** | 49 % | 22 % | 3 % | **−19 %** |
| v13b courageux / nvcave | 24 % | 37 % | 39 % | 35 % | 41 % | 37 % | 37 % |

→ **La signature de l'action s'efface avec la portée** : fenêtre exploitable H ≤ 12, inversion de
signe à H=24 (la dérive mange le signal). v13b bat v6 d'un facteur ~3 à chaque portée : la donnée
en-domaine améliore aussi la **sensibilité à l'action**, pas seulement l'engagement (leçons 18-19).

**(b) `diag_chooser.py` - le chooseur est au niveau du bruit.** Écart minage/contrôle 5-9 pts pour
toutes les portées et toutes les formes de score. Cause identifiée : `ratio_croix` et
`contrastif-vs-rien` récompensent tous deux le mouvement de caméra, **provoqué par le plan lui-même**.
→ D'où le **score jumeau** (même plan, boutons d'interaction coupés) : le seul contrefactuel qui
annule l'égo-mouvement. Implémenté dans `ChooserPolicy(mode="jumeau")`.

**(c) `diag_utilite.py` - LE verdict, et il est négatif.** Seule décision qui compte pour l'agent :
*« taper ICI, ça sert à quelque chose ? »*. Label sans annotation humaine : la réalité tranche
(le bloc visé bouge dans les 12 frames = coup productif). AUC = 50 % → le modèle ne sait pas.

| | AUC brute | **à texture comparable** |
|---|---|---|
| strip - ratio SANS MODÈLE (baseline) | 68,1 % | 52,6 % |
| strip - rêve v13b : ratio | 63,1 % | 51,6 % |
| strip - rêve v13b : **jumeau** | 45,0 % | **45,1 %** |
| nvcave - ratio SANS MODÈLE | 67,0 % | 63,3 % |
| nvcave - rêve v13b : ratio | 69,6 % | **64,6 %** |
| nvcave - rêve v13b : jumeau | 47,6 % | 46,1 % |

Le `std` du patch central (3 lignes, aucun modèle) faisait 84 % en brut → **artefact de texture**
(un patch texturé change forcément plus de pixels). Stratifié, tout s'effondre.

- **Leçon 20 (le plafond, chiffré) : pour la décision que l'agent doit prendre, le rêve n'apporte
  que ~+1,3 pt d'AUC sur une baseline sans modèle, et son score CAUSAL est SOUS 50 % (anti-corrélé).
  Le modèle a appris à *dessiner* du Minecraft, pas à *attribuer le changement à l'action*.**
  C'est le vrai mur, et il est différent de la timidité : la timidité disait « le rêve n'ose pas
  bouger », la leçon 20 dit « quand il bouge, ce n'est pas à cause du bouton ».
- **Discipline confirmée : toujours une baseline SANS MODÈLE + stratification du confondeur.**
  Sans ça on publiait 84 % d'AUC pour du bruit de texture.

**Ce que l'agent doit collecter (conséquence directe).** Pas « jouer », mais faire des **expériences
contrôlées**. `ProbePolicy` : cycle A = taper 30 frames / B = témoin 30 frames *sans bouger* /
C = pivot caméra vers un nouveau bloc. A et B partagent le point de vue ; la SEULE variable est le
bouton. C'est le signal que ni VPT ni YouTube ne donnent proprement (là-bas l'action est devinée par
l'IDM et confondue avec le mouvement - cf. l'échec confondu de v13a). Données sauvées avec
**labels d'actions EXACTS** (on sait quelles touches on a envoyées) et **domaine exact** (mon monde, 1.19, mon rendu) → `data/processed_agent/{tag}_{policy}.npz`, champ `phase` = A/B/C.

**Métrique cible de la boucle : l'AUC stratifiée de `diag_utilite.py` et le SEP de
`diag_separation.py`.** Les deux sont à ~50 % / ~22 %. Les données de paires doivent les faire monter.
Politiques : `probe` (paires, cible SEP) · `scripted` (données riches, 3/4 minage) · `chooser`
(démo Monde 2, deux étages : taper si l'effet du bouton dépasse ε, sinon se déplacer) · `random`.

### Prochaine session - pistes classées
1. ~~Résultat v3b~~ → fait : nouveau modèle de référence.
2. Scénario S2 quantifié (casser un bloc : le bloc disparaît-il dans le rêve ?) - exploite l'expertise (I4).
3. Rollouts plus longs (H=64-128) : où v3 casse-t-il vraiment ?
4. LPIPS en métrique complémentaire (perceptuelle, moins « moyennante »).
5. Monde 2 (premier pas) : choisir entre 2 actions imaginées vers un but visible, en offline d'abord.

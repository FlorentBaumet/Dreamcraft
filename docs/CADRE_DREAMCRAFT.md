# CADRE DREAMCRAFT - décisions gelées (kickoff)

> **Statut** : gelé le **2026-06-26**, *avant tout entraînement*.
> Ce doc fixe les **décisions de kickoff** issues de la session de cadrage : ce qu'on cherche, comment on l'évalue, et ce qu'on s'interdit.

---

## 1. La vision en une phrase
Un **world model visuel de Minecraft** qui apprend d'abord à **prédire** les conséquences des actions, puis à **décider tout seul** (atteindre un but visible) - le tout **évalué honnêtement**, autour d'un levier central : **l'horizon de rêve `H`** (combien d'actions le modèle imagine avant de se ré-ancrer sur le réel).

---

## 2. Décisions gelées

| Décision | Choix | Note |
|---|---|---|
| **Route** | A : prédicteur de frames **supervisé** | pas de RL d'entraînement pour le cœur (I1, I5) |
| **Âme du projet** | **décision autonome**, construite en 2 temps (prédire → choisir) | voir §3 |
| **Levier central** | **horizon `H`** = nb d'actions rêvées avant ré-ancrage sur le réel | fusionne décision + dérive + efficacité |
| **Angle (I3)** | **éval honnête = socle** + **efficacité = axe 2**, *fusionnés dans `H`* | « la décision se dégrade comment quand `H` grandit ? » |
| **Données (offline, Monde 1)** | **VPT** (OpenAI) - contractor *« early game »* (index 6xx), `mp4` + `jsonl` | MineRL 0.4 ne s'installe pas ici (pas de MSVC, Python 3.11) → VPT téléchargé en HTTP direct, riche en bûcheronnage. *Décidé par le smoke test du 2026-06-26.* |
| **Archi Phase 0** | **une** version canonique = conv encoder-decoder action-conditionné **@ 64×64** | le **sweep d'archis** (Transformer, diffusion légère…) = Phase 2 / axe efficacité |
| **Stretch hors-scope** | DreamerV3 (RL complet) | [ko] trop lourd (I1, I5) |

---

## 3. Les deux mondes : prédire → choisir

- **Monde 1 - PRÉDIRE** : l'action est *donnée*, le modèle prédit la frame suivante. **La fondation.** C'est ici qu'on mesure le levier `H` (jusqu'où la prédiction tient avant de dériver).
- **Monde 2 - CHOISIR** : seul un **but** est donné (« ramasse du bois »). Le modèle imagine plusieurs actions, compare, et **joue la meilleure tout seul** = l'autonomie.

> **Règle d'or** : on ne passe au Monde 2 que quand le Monde 1 tient. *On ne choisit bien que si on prédit bien.*
> C'est l'idée « augmenter la portée au fur et à mesure » : prédire 1 pas → étendre `H` → brancher le choix par-dessus.

---

## 4. Garde-fou honnêteté : prédictible vs inconnaissable

Un prédicteur de frames ne peut pas deviner ce que la frame ne contient pas (état caché). On **score au pixel uniquement le déterminé** ; l'inconnaissable sert de **sonde d'hallucination**, jamais à décider.

| Phénomène | Connaissable depuis frame + action ? | Rôle dans l'éval |
|---|---|---|
| Caméra (tourner / avancer) | oui (géométrie) | scoré (Δ-region) |
| Casser un bloc visé (fissure → disparition → drop) | oui (le visible immédiat) | scoré |
| Gravité (sable/gravier) · eau/lave · jour/nuit | oui | scoré |
| Permanence d'objet (tourner puis revenir) | oui (mémoire) | scoré - **excellente sonde** |
| Contenu derrière un bloc / **diamant** | non (état caché) | **pas scoré au pixel → sonde d'hallucination** |
| Terrain au-delà de l'horizon (« dream walk ») | non (procédural/seed) | jugé *cohérence + dérive*, pas pixel-match |

---

## 5. PROTOCOLE D'ÉVAL - GELÉ (avant d'entraîner)

**Baselines obligatoires (à battre) :**
- **B0 - copie-dernière-frame** : `pred = frame_t`. La reine de la triche.
- **B1 - flux optique** (Farnebäck) : warpe `frame_t` selon le flux estimé.

**Métriques :**
- **Pixel global** (MSE / PSNR / SSIM / LPIPS) - reportées, **mais assumées trichables**.
- **Erreur Δ-region** (sur les pixels qui *changent* vraiment) - la métrique honnête. *Skill score* = erreur Δ-region vs B0.
- **Test contrefactuel d'action-conditioning** - même frame, actions différentes → la prédiction change-t-elle *correctement* ?
- **Dérive vs horizon `H`** - la courbe centrale du projet.
  *Amendement 2026-07-03 (leçon de l'après-midi v1/v1.5/v2)* : la dérive se mesure **aussi en Δ-region à chaque pas** (erreur là où la réalité bouge à ce pas). La MSE globale en rollout **récompense le rêve inerte** (un brouillard figé « gagne ») - même piège que la copie à 1 pas. La **portée honnête** = dernier pas où le rêve bat la frame gelée en Δ-region. Corollaire : **la sélection de checkpoint doit se faire sur cette métrique**, pas sur la MSE globale de rollout (on a accidentellement sélectionné de l'inertie en v2).

**Splits :** out-of-sample **par épisode** (pas de fuite temporelle).

**Scénarios d'expert (I4), Phase 0 :**
- **S1** - rotation caméra vs avancer (le test d'action-conditioning de base).
- **S2** - casser un bloc (fissure → disparition → drop).
- **+ 1 sonde d'état caché** - casser vers de l'inconnu, ou regarder ailleurs puis revenir.

---

## 6. Critère Go/No-Go (Phase 0)
Le mini conv enc-dec **bat-il B0** sur la métrique **Δ-region**, **out-of-sample**, sur **S1 + S2** ?
→ **Oui** : on continue (étendre `H`, puis Monde 2). **Non** : on revoit l'archi/les données avant d'aller plus loin.

---

## 7. Roadmap (mise à jour)

| Phase | Contenu | Monde / `H` |
|---|---|---|
| **0 - MVP** [ok] **GO (2026-07-03)** | prédiction 1 pas + éval gelée + Go/No-Go → modèle 1.64M params bat B0 et B1 en Δ-region (0.00397 vs 0.00931/0.00664) sur épisode jamais vu ; détail dans `outputs/phase0/GO_NOGO.md` | Monde 1, `H=1` |
| **1 - Le world model** [ok] **v3 (2026-07-03 soir)** | rollout multi-pas + GIFs + étude de la portée `H` → **portée honnête : 4 pas (v1.5) → 32 pas / ≥1,6 s (v3 : contexte 4 frames + curriculum K + LR-decay + sélection honnête)** ; généralisation 1-pas : 3/3 sessions jamais vues battues/égalées vs flux optique ; démos-rêves dans `outputs/demos/v3/` | Monde 1, `H` jusqu'à 32+ |
| **2 - Décision autonome** [partiel] marche 1 faite (2026-07-05) | boucle où **le modèle choisit** pour ramasser du bois (env *live*), sur le meilleur `H` + **sweep d'archis** (efficacité). **Marche 1 (offline) [ok] : les rêves discriminent les plans - 77 % de bons choix (chance 25 %) avec le scoreur Δ-region ; ≈ chance avec la MSE globale → le protocole honnête s'applique aussi au CHOIX. Optimum décisionnel : H=4-8 → chooseur = rêves courts + ré-ancrage fréquent.** **Marche 2 (offline) [ok]/[partiel] : chooseur but-visible (« casse devant toi ») : 77-79 % de bons choix en terrain ouvert jamais vu (chance 33 %) ; en grotte, la fonction de but naïve sature (le rêveur, lui, garde le signal : AUC 0.7+) → prochain chantier = le détecteur de but, pas le modèle.** | Monde 2 |
| **3 - Packaging** | repo + README + **GIFs de démo** + write-up | - |

> La Phase 2 (env *live* + choix autonome) est le vrai **cran de scope** du projet. On n'y va qu'avec un Monde 1 solide. On s'attend à ce que **grand `H` casse** (erreur qui s'accumule) - **c'est un résultat, pas un échec**.

---

## 8. Encore ouvert (à trancher en temps voulu)
- [x] ~~Smoke test data~~ → **fait (2026-06-26)** : MineRL écarté, **données VPT OK** (640×360 @ 20 fps, `(frame, action)` alignés, bûcheronnage présent). Voir `scripts/smoke_test_data.py`.
- [ ] **Monde 2 - environnement *live*** : MineRL ne s'installant pas ici, comment exécuter les actions pour de vrai en Phase 2 ? (MineRL 1.0 depuis GitHub / autre machine / vrai jeu + capture). *À trancher avant la Phase 2.*
- [ ] Fonction de but du Monde 2 (détecteur visuel « bois cassé » maison, vu qu'on n'a plus la récompense de l'env).
- [ ] Détails du sweep d'archis (petit Transformer, diffusion légère).
- [ ] Résolution (64 px d'abord ; tester 96/128 px pour le détail fin ?) / longueur de contexte (nb de frames passées en entrée).

---

## 9. Invariants (rappel - inchangés)
| # | Invariant |
|---|---|
| **I1** | **Route tractable d'abord** : un prédicteur de frames supervisé. Le RL lourd (DreamerV3 sur MineRL) est un stretch, pas le cœur. |
| **I2** | **Visuel** : on doit pouvoir *voir* le modèle prédire et « rêver » (frames, GIFs). |
| **I3** | **Évaluation honnête** comme colonne vertébrale : le modèle capte-t-il la dynamique, ou triche-t-il ? |
| **I4** | **Expertise Minecraft** (10 ans de jeu) : scénarios de test pertinents, règles du jeu transformées en sondes automatiques. |
| **I5** | **Low-compute** : petit modèle, données existantes, une seule RTX 4070 (8 Go). |

# Scripts

Chaque script est autonome et se lance depuis la racine du repo :

```bash
.venv\Scripts\python.exe scripts\<script>.py
```

Les scripts sont gardés à plat, dans l'ordre où le projet les a produits. Ce tableau les regroupe par rôle. Le raisonnement derrière chacun est dans le [journal](../docs/JOURNAL.md).

## Données

| script | rôle |
|---|---|
| `smoke_test_data.py` | vérifie le pipeline VPT de bout en bout sur 3 épisodes |
| `scan_live_episodes.py` | repère les épisodes VPT encore téléchargeables (~32 % des blobs) |
| `build_dataset.py` | télécharge VPT (mp4 + actions jsonl), convertit en `npz` 64×64 à 20 Hz |
| `ingest_youtube.py` | vidéo YouTube → épisodes, actions pseudo-étiquetées par l'IDM |
| `reingest_96.py` | ré-ingestion du corpus en 96×96 |
| `audit_brightness.py`, `audit_bright_caves.py` | audits de luminosité du corpus (chasse aux grottes éclairées) |

## Monde 1 : prédire

| script | rôle |
|---|---|
| `eval_baselines.py` | baselines gelées avant tout entraînement : copie (B0), flux optique (B1) |
| `train_phase0.py` | Go/No-Go : premier prédicteur 1-pas |
| `train_phase1.py`, `rollout_phase1.py` | entraînement multi-pas et mesure de la portée honnête |
| `train_v3.py` | contexte 4 frames, curriculum K=1→8, sélection sur Δ-rollout |
| `train_v4_finetune.py` | fine-tune K=16 + EMA |
| `train_v5_gamma.py` | gamma simulé pour les grottes (piste négative, leçon 9) |
| `train_v7.py` | **entraîneur principal**, configurable par variables `DC_*` (commit-loss, résolution, capacité, données de l'agent) |
| `train_idm.py` | modèle de dynamique inverse : devine les touches depuis la vidéo |
| `eval_1step.py`, `compare_curves.py` | évaluation 1-pas et comparaison de courbes de rollout |
| `dream_demo.py` | GIFs « réel vs rêve » |

## Règles du jeu : sondes automatiques

| script | rôle |
|---|---|
| `regles_du_jeu.py` | sondes R1-R4 : règles Minecraft transformées en tests sur les rêves |
| `regles_du_jeu_2.py`, `probe_chute.py`, `probe_faim.py` | règles #5 (dégâts de chute) et #10 (manger → faim) |
| `mine_events.py`, `mine_falls_meals.py`, `mine_seau_v2.py` | extraction automatique d'événements dans les vidéos |

## Timidité et couverture de données

| script | rôle |
|---|---|
| `figure_engagement.py` | le rêve s'engage-t-il sur l'événement ? (réel / timide / courageux) |
| `gif_courage.py` | GIF timide vs courageux |

## Monde 2 : choisir

| script | rôle |
|---|---|
| `train_goal_detector.py` | GoalNet : détecteur appris de « bloc cassé » |
| `monde2_choice.py`, `monde2_chooser.py` | choix hors ligne entre plans rêvés |

## Agent et diagnostic du plafond (en cours)

| script | rôle |
|---|---|
| `live_smoke.py` | test de plomberie sur le vrai jeu : capture d'écran + rotation caméra |
| `agent_loop.py` | **la boucle agent** : politiques `probe` / `scripted` / `chooser` / `random` × backends `replay` / `live` |
| `diag_separation.py` | part de l'effet réel attribuée au bouton ; `--sweep` trace la loi de la portée |
| `diag_chooser.py` | le chooseur choisit-il de taper quand le joueur réel tape ? |
| `diag_utilite.py` | « taper ici sert-il ? » : AUC vs baselines sans modèle, stratifiée par texture |

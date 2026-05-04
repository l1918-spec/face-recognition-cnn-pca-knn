# Presentation Guide — Exposé 2: Reconnaissance Faciale

Step-by-step playbook for a 10–15 minute Exposé. Combines the Jupyter notebook (`notebook.ipynb`) for theory + numbers and the live webcam demo for impact.

---

## Before you start (10 min before)

```powershell
cd C:\Users\debai\Desktop\bio
.venv\Scripts\activate
python main.py                        # warm Olivetti cache, ensure models exist
jupyter notebook notebook.ipynb       # open notebook in browser tab
```

Have these tabs ready:
1. Notebook (rendered, all cells expanded)
2. PowerShell window in `C:\Users\debai\Desktop\bio` ready to run `webcam_demo`
3. File explorer on `results/` to show plots full-screen if needed

Test the webcam **once** before you stand up. Lighting must be on.

---

## Plan (12 minutes)

| Time | Section | What you do |
|------|---------|-------------|
| 0:00 – 0:30 | **Intro** | Title slide / notebook §1 |
| 0:30 – 2:00 | **Dataset & pipeline** | Notebook §2–3 |
| 2:00 – 4:00 | **KNN** | Notebook §4 |
| 4:00 – 6:30 | **PCA + KNN (Eigenfaces)** | Notebook §5 |
| 6:30 – 9:00 | **CNN** | Notebook §6 |
| 9:00 – 10:30 | **Comparison** | Notebook §7–8 |
| 10:30 – 12:00 | **Live demo** | Webcam |
| 12:00 – 13:00 | **Conclusion + Q&A** | Notebook §9 |

---

## Section-by-section script

### 1. Intro (30 s)

> "Reconnaissance faciale = identifier une personne à partir de son visage. Trois approches comparées : K-Nearest Neighbors sur pixels bruts, PCA + KNN (Eigenfaces, 1991), et CNN moderne. Même dataset, mêmes métriques."

Show notebook §1 title.

### 2. Dataset & pipeline (90 s)

Show §2 — the 4 × 10 sample grid.

> "Olivetti : 40 sujets, 10 photos chacun, 64 × 64 niveaux de gris. 400 images au total. On split 80/20 stratifié = 320 train / 80 test."

Mention pipeline:

> "Pour chaque image : détection Haar → crop → resize 64 × 64 → normalisation [0,1]. Identique pour les trois algorithmes. Les différences viennent de ce qu'on fait *après*."

### 3. KNN (2 min)

Show §4 — confusion matrix.

> "Premier algo, simple : on aplatit l'image en un vecteur de 4096 valeurs et on cherche les 3 voisins les plus proches en distance euclidienne. **Aucun entraînement** — KNN stocke juste les vecteurs."
>
> "Résultat : **88,7 %** d'accuracy. Pas mal pour zéro entraînement, mais l'inférence prend **20 ms par image** parce qu'il faut comparer à toute la base. Et c'est très sensible à l'éclairage."

### 4. PCA + KNN (Eigenfaces) (2.5 min) ⭐

Show §5 — the **Eigenfaces plot** is the visual highlight of the Exposé.

> "PCA = Analyse en Composantes Principales. On projette les 4096 dimensions dans 50 dimensions qui capturent la variance dominante. Ces 50 composantes, on peut les *visualiser* — ce sont des visages 'fantômes' qu'on appelle **Eigenfaces** (Turk & Pentland, 1991)."

Point to the eigenfaces. Comment on what each captures (lighting, glasses, gender contour).

> "La courbe de variance cumulée montre que 50 composantes capturent ~90 % de la variance — on perd 10 % de signal pour gagner un facteur 80 sur la dimension."
>
> "Résultat : **90 %** accuracy, mieux que KNN brut, et inférence en **0,1 ms** — 200 fois plus rapide. Idéal pour des contraintes de ressources."

### 5. CNN (2.5 min)

Show §6 — architecture, then the loss/accuracy curves.

> "CNN = réseau convolutionnel. Trois blocs Conv + MaxPool, puis Flatten + Dense. Augmentation : flip horizontal et translation aléatoire de 5 %. Optimiseur Adam, early stopping sur val_accuracy."

Show the loss/accuracy curves:

> "On voit la convergence : la loss descend, l'accuracy de validation monte vers ~99 %. Early stopping arrête l'entraînement quand le modèle ne progresse plus."
>
> "Résultat : **98,8 % à 100 %** selon le seed. La matrice de confusion montre une diagonale presque parfaite — souvent 1 seule erreur sur 80 images de test."

### 6. Comparison (90 s)

Show §7 — table + bar chart.

> "Bilan quantitatif :"
>
> | Modèle | Accuracy | Train | Inference |
> |--------|----------|-------|-----------|
> | KNN | 88,7 % | < 1 ms | 20 ms |
> | PCA+KNN | 90 % | 50 ms | 0,1 ms |
> | CNN | 98,8 % | 15 s | 2 ms |
>
> "CNN gagne en précision. PCA+KNN gagne en vitesse d'inférence. KNN brut est dominé partout."

Show §8 — error grid:

> "Les erreurs résiduelles concernent les lunettes, les angles latéraux, et les expressions inhabituelles. Limites communes aux trois algorithmes."

### 7. Live demo (90 s) ⭐

Switch to PowerShell:

```powershell
python -m src.webcam_demo --model cnn
```

> "Démo en temps réel. Le modèle est entraîné sur Olivetti, donc il *ne me reconnaît pas* — il va dire 'le sujet d'Olivetti dont je ressemble le plus'. Mais on voit le pipeline complet : détection Haar, crop, prédiction, affichage de la confiance."

Point to the colored box:

> "La couleur de la boîte = niveau de confiance : **vert > 80 %**, **orange 50–80 %**, **rouge < 50 %**. La barre au-dessus de la boîte montre la probabilité."

If you have time and a custom model:

> "Avec un modèle entraîné sur mon visage (`python main.py --data data/raw`), il afficherait mon nom directement avec une confiance > 95 %."

Press `q`, return to notebook.

### 8. Conclusion (60 s)

Show §9.

> "Résumé : trois approches, du plus simple au plus complexe. KNN brut = baseline. PCA+KNN = compromis vitesse/précision. CNN = précision maximale au prix du temps d'entraînement."
>
> "Limites : Haar échoue sur les profils, dataset Olivetti est petit (10 images / personne), pas d'anti-spoofing. Pour la production, on irait vers MTCNN + transfer learning sur ResNet ou MobileNet."
>
> "Questions ?"

---

## Anticipated Q&A

| Question | Réponse |
|----------|---------|
| *"Pourquoi pas plus de composantes en PCA ?"* | 50 capture déjà 90 % de variance. Au-delà, gain marginal et risque de réintroduire du bruit. |
| *"Le CNN va-t-il marcher sur des nouvelles personnes ?"* | Non — il faut le réentraîner. C'est de la *closed-set recognition*. Pour open-set, il faudrait des embeddings (FaceNet, ArcFace). |
| *"Combien d'images pour entraîner sur quelqu'un ?"* | 30+ minimum, 100+ idéal. Varier angles, expressions, éclairage. |
| *"Pourquoi KNN est lent et pas le CNN ?"* | KNN compare à *toutes* les images d'entraînement à chaque inférence. Le CNN a tout encodé dans ses poids. |
| *"Si je porte un masque ?"* | Tous les modèles échouent — entraînés sur visages complets. Solution : modèle dédié (RetinaFace, etc.). |
| *"Anti-spoofing ?"* | Pas implémenté. Une photo imprimée passerait. Solutions : détection de clignement, profondeur (caméra IR). |
| *"Sur GPU ça change quoi ?"* | Le CNN passerait de 15 s à ~2 s d'entraînement. L'inférence resterait ~2 ms (déjà rapide). |

---

## What to put on slides (if separate from notebook)

1. **Slide 1**: Title + Modalité + Algorithmes étudiés (= image of brief)
2. **Slide 2**: Dataset Olivetti + sample grid (export from §2)
3. **Slide 3**: Pipeline diagram
4. **Slide 4**: KNN — formula + result + confusion matrix
5. **Slide 5**: Eigenfaces — the 12 components plot ⭐
6. **Slide 6**: Variance curve + PCA result
7. **Slide 7**: CNN architecture diagram
8. **Slide 8**: CNN loss/accuracy curves
9. **Slide 9**: Comparison table + bar chart ⭐
10. **Slide 10**: Live demo (full screen webcam) ⭐
11. **Slide 11**: Errors grid (failure cases)
12. **Slide 12**: Conclusion + limitations + references

Plots are already in `results/`:
- `results/eigenfaces.png`
- `results/comparison.png`
- `results/cm_knn.png`, `results/cm_pca_knn.png`, `results/cm_cnn.png`
- `results/test_errors.png` (from `show_test.py --errors`)

---

## Backup plans

| If… | Do this |
|-----|---------|
| Webcam fails | `python demo_offline.py` — works without camera |
| Internet down (first Olivetti download) | Run `python main.py` at home before, dataset cached |
| Notebook kernel dies | Pre-rendered HTML: `jupyter nbconvert --to html notebook.ipynb` |
| Projector resolution awkward | Plots in `results/` are PNG, open full-screen in image viewer |
| Tight on time | Skip §4 (KNN section) — go straight from intro to PCA |

---

## Confidence-builders

- The notebook is **already executed**. Even if all the live code crashes, the displayed outputs stay correct.
- `results/metrics.json` has the raw numbers. If asked "exact accuracy?", open it.
- The live demo recognizes Olivetti subjects, **not** you. Make this clear up-front so people don't expect their name on screen.

---

## After the Exposé

If audience interested:
- Hand them `notebook.ipynb` + `results/` folder
- Or zip the whole project (see "Email package" in `HOW_TO_RUN.md`)

Done. Bonne présentation.

# Email template

Two versions below — pick one. Replace `[NAME]` and `[YOUR NAME]`.

---

## Version FR (formal)

**Sujet :** Exposé 2 — Reconnaissance faciale (CNN, PCA, KNN)

Bonjour [NAME],

Vous trouverez en pièce jointe le projet complet de l'Exposé 2 sur la reconnaissance faciale, avec une comparaison de trois algorithmes (KNN, PCA + KNN, CNN).

**Pour lire le rapport (sans rien installer) :**
1. Décompressez `bio_expose.zip`
2. Ouvrez `README.md` à la racine — c'est le point d'entrée et la table des matières
3. Ouvrez `notebook.ipynb` dans VS Code ou un navigateur — toutes les sorties et les graphiques sont déjà intégrés, pas besoin de réexécuter

**Pour exécuter le code :**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # entraîne et évalue les 3 modèles
python -m src.webcam_demo --model cnn   # démo webcam en temps réel
```

Détails complets dans `HOW_TO_RUN.md`.

**Documents inclus :**
- `README.md` — point d'entrée
- `notebook.ipynb` — rapport Jupyter complet (déjà exécuté)
- `HOW_TO_RUN.md` — toutes les commandes
- `DEMO_README.md` — guide de la démo webcam
- `NOTEBOOK_README.md` — guide du notebook
- `PRESENTATION_GUIDE.md` — script de présentation (12 min)
- `results/` — graphiques (Eigenfaces, matrices de confusion, comparaison)
- `models/` — modèles entraînés prêts à l'emploi

**Résultats principaux** (dataset Olivetti, 80 images de test) :

| Modèle | Accuracy | F1 | Inférence |
|--------|----------|-----|-----------|
| KNN | 88,8 % | 0,888 | 20 ms/img |
| PCA + KNN | 90,0 % | 0,900 | 0,1 ms/img |
| **CNN** | **98,8 – 100 %** | **0,99** | 2 ms/img |

N'hésitez pas à me contacter si vous avez des questions ou si vous souhaitez voir la démo en direct.

Cordialement,
[YOUR NAME]

---

## Version EN (concise)

**Subject:** Exposé 2 — Face Recognition (CNN, PCA, KNN)

Hi [NAME],

Attached is the complete project for Exposé 2 — face recognition comparing three algorithms (KNN, PCA + KNN, CNN).

**To read the report (zero install):**
1. Unzip `bio_expose.zip`
2. Open `README.md` at the root — it's the entry point and index
3. Open `notebook.ipynb` in VS Code or a browser — all outputs and plots are already embedded

**To run the code:**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py                    # train + evaluate all 3 models
python -m src.webcam_demo --model cnn   # live webcam demo
```

Full instructions in `HOW_TO_RUN.md`.

**Files included:**
- `README.md` — entry point
- `notebook.ipynb` — full Jupyter report (executed)
- `HOW_TO_RUN.md` — every command
- `DEMO_README.md` — webcam demo guide
- `NOTEBOOK_README.md` — notebook guide
- `PRESENTATION_GUIDE.md` — 12-min presentation script
- `results/` — plots (Eigenfaces, confusion matrices, comparison)
- `models/` — trained models ready to use

**Headline results** (Olivetti, 80 test images):

| Model | Accuracy | F1 | Inference |
|-------|----------|-----|-----------|
| KNN | 88.8 % | 0.888 | 20 ms/img |
| PCA + KNN | 90.0 % | 0.900 | 0.1 ms/img |
| **CNN** | **98.8 – 100 %** | **0.99** | 2 ms/img |

Happy to walk through it or run the live demo if helpful.

Best,
[YOUR NAME]

---

## Short version (3 sentences, if recipient is busy)

Hi [NAME],

Attached: full Exposé 2 project on face recognition (KNN, PCA, CNN). Start with `README.md`, then `notebook.ipynb` for the runnable report — all plots already embedded, no install needed to read it. Setup + commands in `HOW_TO_RUN.md`.

Best,
[YOUR NAME]

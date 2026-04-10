# Détection du Cancer Pulmonaire — IA Multimodale
**M2 ESIC — Intelligence Artificielle | Sanae Najimi |**

---

## Structure du projet

```
cancer-detection-tp/
├── app.py                              # Interface Streamlit
├── EDA.py                              # Partie 0 — Analyse exploratoire
├── Classification_tab.py               # Partie 1 — ML tabulaire
├── CNN.py                              # Partie 2 — Deep Learning
├── Dockerfile                          # Conteneurisation
├── requirements.txt                    # Dépendances Python
├── .github/workflows/deploy.yml        # Pipeline CI/CD GitHub Actions
└── models/
    ├── modele1_meilleur.pkl            # Random Forest
    ├── modele2a_cnn_image.keras        # CNN image seul
    └── modele2b_cnn_multimodal.keras   # CNN multimodal
```

---

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

## Lancement via Docker

```bash
docker build -t cancer-detection-app .
docker run -p 8501:8501 cancer-detection-app
# → http://localhost:8501
```

---

## Application déployée

**URL publique :** https://cancer-detection-tp.onrender.com/

Déploiement automatique via **GitHub Actions CI/CD** à chaque push sur `main` :
- Job 1 — Tests Python + vérification des modèles
- Job 2 — Build Docker
- Job 3 — Déploiement Render

---

## Pipeline IA

```
Données cliniques (14 features)
        ↓
[Modèle 1 — Random Forest]  →  Risque : Faible / Intermédiaire / Élevé
        ↓
[Modèle 2B — CNN Multimodal]  →  Cancer : Probable / Non probable
(image thoracique + probabilités Modèle 1)
```

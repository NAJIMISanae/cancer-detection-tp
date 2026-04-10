import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, roc_auc_score)
from xgboost import XGBClassifier

# CONFIG 
DATA_PATH  = r"C:\Users\BARRAQ\Desktop\ML ESIC\patients_cancer_poumon 1.csv"
OUTPUT_DIR = r"C:\Users\BARRAQ\Desktop\ML ESIC\Partie1_output"
MODEL_DIR  = r"C:\Users\BARRAQ\Desktop\ML ESIC\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLORS  = ["#2ecc71", "#f39c12", "#e74c3c"]
SEED    = 42

FEATURES = [
    "age", "sexe_masculin", "presence_nodule", "subtilite_nodule",
    "taille_nodule_px", "x_nodule_norm", "y_nodule_norm",
    "tabagisme_paquets_annee", "toux_chronique", "dyspnee",
    "douleur_thoracique", "perte_poids", "spo2", "antecedent_familial"
]
TARGET = "risque_malignite"


# 1. CHARGEMENT & SPLIT

df = pd.read_csv(DATA_PATH)
X  = df[FEATURES]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print("SPLIT TRAIN / TEST")
print(f"  Train : {X_train.shape[0]} patients")
print(f"  Test  : {X_test.shape[0]} patients")
print(f"  Distribution train : {dict(y_train.value_counts().sort_index())}")
print(f"  Distribution test  : {dict(y_test.value_counts().sort_index())}")

# 2. DÉFINITION DES 3 MODÈLES (pipelines)

models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=SEED
        ))
    ]),
    "XGBoost": Pipeline([
        ("clf", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=SEED
        ))
    ]),
    "Régression Logistique": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
            random_state=SEED
        ))
    ]),
}

# 3. ENTRAÎNEMENT & ÉVALUATION

print("ÉVALUATION DES MODÈLES")

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, pipeline in models.items():
    print(f"\n── {name} ──")

    # Cross-validation F1 macro
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring="f1_macro")
    print(f"  CV F1-macro (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement sur train complet
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")

    # AUC multiclasse OVR
    y_bin   = label_binarize(y_test, classes=[0, 1, 2])
    auc     = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 macro  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred,
          target_names=["Faible (0)", "Intermédiaire (1)", "Élevé (2)"]))

    results[name] = {
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": acc,
        "f1_macro": f1,
        "auc":      auc,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }
# 4. COMPARAISON & SÉLECTION DU MEILLEUR MODÈLE

print("COMPARAISON DES MODÈLES")
comparison = pd.DataFrame({
    name: {
        "CV F1-macro": f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        "Test Accuracy": f"{r['accuracy']:.4f}",
        "Test F1-macro": f"{r['f1_macro']:.4f}",
        "AUC-ROC":       f"{r['auc']:.4f}",
    }
    for name, r in results.items()
}).T
print(comparison.to_string())

best_name = max(results, key=lambda k: results[k]["f1_macro"])
best      = results[best_name]
print(f"\n  Meilleur modèle : {best_name}  (F1-macro test = {best['f1_macro']:.4f})")


# 5. FIGURE 1 — COMPARAISON DES MODÈLES (barplot)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 5 — Comparaison des performances des modèles",
             fontsize=14, fontweight="bold")

metrics   = ["accuracy", "f1_macro", "auc"]
titles    = ["Accuracy", "F1-score Macro", "AUC-ROC"]
bar_color = ["#3498db", "#e67e22", "#9b59b6"]

for ax, metric, title, color in zip(axes, metrics, titles, bar_color):
    vals  = [results[n][metric] for n in results]
    names = ["Random\nForest", "XGBoost", "Régr.\nLogistique"]
    bars  = ax.bar(names, vals, color=color, alpha=0.8, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Seuil 0.90")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig5_comparaison_modeles.png", dpi=150, bbox_inches="tight")
plt.close()


# 6. FIGURE 2 — MATRICES DE CONFUSION (3 modèles)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 6 — Matrices de confusion (jeu de test)",
             fontsize=14, fontweight="bold")

labels = ["Faible", "Interm.", "Élevé"]
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig6_matrices_confusion.png", dpi=150, bbox_inches="tight")
plt.close()


# 7. FIGURE 3 — FEATURE IMPORTANCE 

if best_name in ["Random Forest", "XGBoost"]:
    clf  = best["pipeline"].named_steps["clf"]
    imps = clf.feature_importances_
    idx  = np.argsort(imps)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [FEATURES[i] for i in idx],
        imps[idx],
        color=["#e74c3c" if imps[i] > 0.1 else "#3498db" for i in idx],
        edgecolor="white", linewidth=0.8
    )
    ax.set_xlabel("Importance relative")
    ax.set_title(f"Figure 7 — Importance des features ({best_name})",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(ax.patches, imps[idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig7_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

# 8. PROBABILITÉS DU MEILLEUR MODÈLE (aperçu)

print(f"PROBABILITÉS DE PRÉDICTION — {best_name} (10 premiers patients test)")
proba_df = pd.DataFrame(
    best["y_proba"],
    columns=["P(Faible)", "P(Intermédiaire)", "P(Élevé)"]
).round(3)
proba_df["Prédit"]  = best["y_pred"]
proba_df["Réel"]    = y_test.values
proba_df["Correct"] = proba_df["Prédit"] == proba_df["Réel"]
print(proba_df.head(10).to_string(index=False))


# 9. SAUVEGARDE DU MEILLEUR MODÈLE

model_path = os.path.join(MODEL_DIR, "modele1_meilleur.pkl")
joblib.dump(best["pipeline"], model_path)

# Sauvegarde aussi les 3 pour référence
for name, r in results.items():
    safe_name = name.lower().replace(" ", "_").replace("é", "e").replace("è", "e")
    joblib.dump(r["pipeline"], os.path.join(MODEL_DIR, f"modele1_{safe_name}.pkl"))
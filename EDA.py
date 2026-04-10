import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins et paramètres
DATA_PATH  = r"C:\Users\BARRAQ\Desktop\ML ESIC\patients_cancer_poumon 1.csv"
IMG_DIR    = r"C:\Users\BARRAQ\Desktop\ML ESIC\jsrt_subset"
OUTPUT_DIR = r"C:\Users\BARRAQ\Desktop\ML ESIC\EDA_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"0 – Faible": "#2ecc71", "1 – Intermédiaire": "#f39c12", "2 – Élevé": "#e74c3c"}
COLORS  = ["#2ecc71", "#f39c12", "#e74c3c"]

#Chargement du dataset
df = pd.read_csv(DATA_PATH)

#Définition des features et du target
FEATURES = [
    "age", "sexe_masculin", "presence_nodule", "subtilite_nodule",
    "taille_nodule_px", "x_nodule_norm", "y_nodule_norm",
    "tabagisme_paquets_annee", "toux_chronique", "dyspnee",
    "douleur_thoracique", "perte_poids", "spo2", "antecedent_familial"
]
TARGET_TAB = "risque_malignite"
TARGET_IMG = "cancer_image"

label_map = {0: "0 – Faible", 1: "1 – Intermédiaire", 2: "2 – Élevé"}
df["risque_label"] = df[TARGET_TAB].map(label_map)

print("Aperçu du dataset:")
print()
print(f"  Lignes : {df.shape[0]}  |  Colonnes : {df.shape[1]}")
print(f"  Valeurs manquantes : {df.isnull().sum().sum()}")
print()

print("Distribution de la cible (risque_malignite) :")
print(df[TARGET_TAB].value_counts().sort_index()
      .rename({0:"Faible (0)", 1:"Intermédiaire (1)", 2:"Élevé (2)"}).to_string())
print()
print("Distribution cible image (cancer_image) :")
print(df[TARGET_IMG].value_counts().rename({0:"Non cancer", 1:"Cancer"}).to_string())
print()
print("Statistiques descriptives :")
print(df[FEATURES].describe().round(2).to_string())


# FIGURE 1: DISTRIBUTION DES CIBLES &  ÂGE PAR RISQUE
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 1 — Distribution des cibles et profil démographique",
             fontsize=14, fontweight="bold", y=1.02)
# 1a. Distribution du risque de malignité
counts = df[TARGET_TAB].value_counts().sort_index()
bars = axes[0].bar(
    ["Faible (0)", "Intermédiaire (1)", "Élevé (2)"],
    counts.values, color=COLORS, edgecolor="white", linewidth=1.5
)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val}\n({val/len(df)*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_title("Distribution du risque de malignité", fontsize=12)
axes[0].set_ylabel("Nombre de patients")
axes[0].set_ylim(0, max(counts.values) * 1.2)
axes[0].spines[["top","right"]].set_visible(False)
# 1b. Répartition binaire cancer_image
img_counts = df[TARGET_IMG].value_counts()
wedges, texts, autotexts = axes[1].pie(
    img_counts.values,
    labels=["Non cancer", "Cancer"],
    colors=["#3498db", "#e74c3c"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontweight("bold")
axes[1].set_title("Répartition cible image (cancer_image)", fontsize=12)
# 1c. Distribution de l'âge par niveau de risque
for i, (code, label) in enumerate(label_map.items()):
    subset = df[df[TARGET_TAB] == code]["age"]
    axes[2].hist(subset, bins=12, alpha=0.65, label=label, color=COLORS[i], edgecolor="white")
axes[2].set_title("Distribution de l'âge par risque", fontsize=12)
axes[2].set_xlabel("Âge (années)")
axes[2].set_ylabel("Fréquence")
axes[2].legend()
axes[2].spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig1_distribution_cibles.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 2 : VARIABLES CLINIQUES PAR RISQUE

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Figure 2 — Variables cliniques selon le niveau de risque",
             fontsize=14, fontweight="bold")
# Définition des variables à visualiser
plot_configs = [
    ("tabagisme_paquets_annee", "Tabagisme (paquets/an)", "boxplot"),
    ("spo2",                    "SpO₂ (%)",               "boxplot"),
    ("taille_nodule_px",        "Taille nodule (px)",     "boxplot"),
    ("toux_chronique",          "Toux chronique",         "barplot"),
    ("dyspnee",                 "Dyspnée",                "barplot"),
    ("antecedent_familial",     "Antécédent familial",    "barplot"),
]
for ax, (col, title, kind) in zip(axes.flat, plot_configs):
    if kind == "boxplot":
        data_grouped = [df[df[TARGET_TAB] == k][col].values for k in [0,1,2]]
        bp = ax.boxplot(data_grouped, patch_artist=True,
                        medianprops={"color":"black","linewidth":2})
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(["Faible", "Interm.", "Élevé"])
    else:  # barplot de proportions
        props = df.groupby(TARGET_TAB)[col].mean() * 100
        bars = ax.bar(["Faible", "Interm.", "Élevé"], props.values,
                      color=COLORS, edgecolor="white", linewidth=1.5, alpha=0.85)
        for bar, val in zip(bars, props.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("% de patients")
        ax.set_ylim(0, 115)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig2_variables_cliniques.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 3 : MATRICE DE CORRÉLATION

fig, ax = plt.subplots(figsize=(13, 10))

corr_cols = FEATURES + [TARGET_TAB, TARGET_IMG]
corr = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))  # masque triangle sup
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Figure 3 — Matrice de corrélation (features + cibles)",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig3_correlation.png", dpi=150, bbox_inches="tight")
plt.close()


# FIGURE 4 : RADIOS THORACIQUES PAR CLASSE

classes = {"sain": "Sain", "benin": "Bénin", "malin": "Malin"}
n_per_class = 3
fig = plt.figure(figsize=(14, 6))
fig.suptitle("Figure 4 — Radios thoraciques représentatives par classe",
             fontsize=14, fontweight="bold")
col_idx = 0
for classe, label in classes.items():
    class_dir = os.path.join(IMG_DIR, classe)
    images = sorted(os.listdir(class_dir))[:n_per_class]
    for row_idx, img_name in enumerate(images):
        ax = fig.add_subplot(n_per_class, 3, row_idx * 3 + col_idx + 1)
        img = Image.open(os.path.join(class_dir, img_name)).convert("L")
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if row_idx == 0:
            color = {"sain":"#2ecc71","benin":"#f39c12","malin":"#e74c3c"}[classe]
            ax.set_title(f" {label}" if classe=="sain" else
                         f" {label}" if classe=="benin" else f" {label}",
                         fontsize=12, fontweight="bold", color=color)
    col_idx += 1
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig4_radios_par_classe.png", dpi=150, bbox_inches="tight")
plt.close()
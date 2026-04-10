import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# CONFIG 
DATA_PATH   = r"C:\Users\BARRAQ\Desktop\ML ESIC\patients_cancer_poumon 1.csv"
IMG_DIR     = r"C:\Users\BARRAQ\Desktop\ML ESIC\jsrt_subset"
OUTPUT_DIR  = r"C:\Users\BARRAQ\Desktop\ML ESIC\Partie2_output"
MODEL_DIR   = r"C:\Users\BARRAQ\Desktop\ML ESIC\models"
MODEL1_PATH = r"C:\Users\BARRAQ\Desktop\ML ESIC\models\modele1_meilleur.pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

IMG_SIZE = (64, 64)   # résolution d'entrée du CNN
SEED     = 42
EPOCHS   = 30

# 1. CHARGEMENT DES DONNÉES TABULAIRES & MODÈLE 1
print("CHARGEMENT DES DONNÉES")

df       = pd.read_csv(DATA_PATH)
model1   = joblib.load(MODEL1_PATH)

FEATURES_TAB = [
    "age", "sexe_masculin", "presence_nodule", "subtilite_nodule",
    "taille_nodule_px", "x_nodule_norm", "y_nodule_norm",
    "tabagisme_paquets_annee", "toux_chronique", "dyspnee",
    "douleur_thoracique", "perte_poids", "spo2", "antecedent_familial"
]

# Probabilités du Modèle 1 pour chaque patient
proba_m1 = model1.predict_proba(df[FEATURES_TAB])   # shape (184, 3)
print(f"Probabilités Modèle 1 générées : {proba_m1.shape}")

# 2. CHARGEMENT DES IMAGES
print("\nChargement des images...")

# Mapping classe JSRT → label binaire cancer_image
cls_map = {"sain": 0, "benin": 0, "malin": 1}

images, labels, patient_ids = [], [], []

for idx, row in df.iterrows():
    img_path_rel = row["image_path"]                        # ex: jsrt_subset/malin/JPCLN001.jpg
    # Reconstruit le chemin absolu
    img_path = os.path.join(
        os.path.dirname(IMG_DIR),
        img_path_rel.replace("/", os.sep)
    )
    if not os.path.exists(img_path):
        # fallback : chercher dans IMG_DIR directement
        parts    = img_path_rel.split("/")
        img_path = os.path.join(IMG_DIR, parts[-2], parts[-1])

    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    images.append(np.array(img, dtype=np.float32) / 255.0)
    labels.append(row["cancer_image"])
    patient_ids.append(idx)

images    = np.array(images)    # (184, 64, 64, 3)
labels    = np.array(labels)    # (184,)
proba_m1  = proba_m1[patient_ids]

print(f"Images chargées : {images.shape}")
print(f"Labels : {dict(zip(*np.unique(labels, return_counts=True)))}")

# 3. SPLIT (stratifié, même seed que Partie 1)
idx_all = np.arange(len(images))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.2,
                                   random_state=SEED, stratify=labels)

X_img_tr, X_img_te   = images[idx_tr],   images[idx_te]
y_tr,      y_te       = labels[idx_tr],   labels[idx_te]
p_tr,      p_te       = proba_m1[idx_tr], proba_m1[idx_te]

print(f"\nSplit — Train: {len(idx_tr)} | Test: {len(idx_te)}")
print(f"Distribution train : {dict(zip(*np.unique(y_tr, return_counts=True)))}")
print(f"Distribution test  : {dict(zip(*np.unique(y_te, return_counts=True)))}")

# 4. DATA AUGMENTATION
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

# 5A. MODÈLE IMAGE SEUL (CNN simple)

print("MODÈLE 2A — CNN IMAGE ")

def build_cnn_image_only():
    inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x   = data_aug(inp)
    x   = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name="CNN_image_only")

model_img = build_cnn_image_only()
model_img.summary()

model_img.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss")
]

hist_img = model_img.fit(
    X_img_tr, y_tr,
    validation_data=(X_img_te, y_te),
    epochs=EPOCHS, batch_size=16,
    callbacks=callbacks, verbose=1
)

loss_img, acc_img, auc_img = model_img.evaluate(X_img_te, y_te, verbose=0)
y_pred_img = (model_img.predict(X_img_te, verbose=0) > 0.5).astype(int).flatten()

print(f"\n Résultats CNN image seul ")
print(f"  Accuracy : {acc_img:.4f}")
print(f"  AUC-ROC  : {auc_img:.4f}")

# 5B. MODÈLE MULTIMODAL (image + probas Modèle 1)

print("MODÈLE 2B — CNN MULTIMODAL (image + probas Modèle 1)")

def build_cnn_multimodal():
    # Branche image
    img_inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x       = data_aug(img_inp)
    x       = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.MaxPooling2D(2)(x)
    x       = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.MaxPooling2D(2)(x)
    x       = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)

    # Branche tabulaire (3 probas du Modèle 1)
    tab_inp = keras.Input(shape=(3,), name="tabular_input")
    t       = layers.Dense(16, activation="relu")(tab_inp)
    t       = layers.Dense(16, activation="relu")(t)

    # Fusion
    merged  = layers.Concatenate()([x, t])
    merged  = layers.Dense(32, activation="relu")(merged)
    merged  = layers.Dropout(0.3)(merged)
    out     = layers.Dense(1, activation="sigmoid", name="output")(merged)

    return Model([img_inp, tab_inp], out, name="CNN_multimodal")

model_mm = build_cnn_multimodal()
model_mm.summary()

model_mm.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

hist_mm = model_mm.fit(
    [X_img_tr, p_tr], y_tr,
    validation_data=([X_img_te, p_te], y_te),
    epochs=EPOCHS, batch_size=16,
    callbacks=callbacks, verbose=1
)

loss_mm, acc_mm, auc_mm = model_mm.evaluate([X_img_te, p_te], y_te, verbose=0)
y_pred_mm = (model_mm.predict([X_img_te, p_te], verbose=0) > 0.5).astype(int).flatten()

print(f"\n── Résultats CNN multimodal ──")
print(f"  Accuracy : {acc_mm:.4f}")
print(f"  AUC-ROC  : {auc_mm:.4f}")

# 6. COMPARAISON FINALE

print("COMPARAISON MODÈLE IMAGE SEUL vs MULTIMODAL")
print(f"  {'Modèle':<30} {'Accuracy':>10} {'AUC-ROC':>10}")
print(f"  {'-'*50}")
print(f"  {'CNN image seul':<30} {acc_img:>10.4f} {auc_img:>10.4f}")
print(f"  {'CNN multimodal':<30} {acc_mm:>10.4f} {auc_mm:>10.4f}")
delta_acc = acc_mm - acc_img
delta_auc = auc_mm - auc_img
print(f"  {'Gain multimodal (Δ)':<30} {delta_acc:>+10.4f} {delta_auc:>+10.4f}")

# 7. FIGURE 8 — COURBES D'APPRENTISSAGE
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Figure 8 — Courbes d'apprentissage des deux CNN",
             fontsize=14, fontweight="bold")

# Image seul
axes[0,0].plot(hist_img.history["loss"],     label="Train loss",  color="#3498db")
axes[0,0].plot(hist_img.history["val_loss"], label="Val loss",    color="#e74c3c", linestyle="--")
axes[0,0].set_title("CNN Image seul — Loss", fontweight="bold")
axes[0,0].set_xlabel("Époque"); axes[0,0].set_ylabel("Loss")
axes[0,0].legend(); axes[0,0].spines[["top","right"]].set_visible(False)

axes[0,1].plot(hist_img.history["accuracy"],     label="Train acc",  color="#3498db")
axes[0,1].plot(hist_img.history["val_accuracy"], label="Val acc",    color="#e74c3c", linestyle="--")
axes[0,1].set_title("CNN Image seul — Accuracy", fontweight="bold")
axes[0,1].set_xlabel("Époque"); axes[0,1].set_ylabel("Accuracy")
axes[0,1].set_ylim(0, 1.05); axes[0,1].legend()
axes[0,1].spines[["top","right"]].set_visible(False)

# Multimodal
axes[1,0].plot(hist_mm.history["loss"],     label="Train loss",  color="#9b59b6")
axes[1,0].plot(hist_mm.history["val_loss"], label="Val loss",    color="#e67e22", linestyle="--")
axes[1,0].set_title("CNN Multimodal — Loss", fontweight="bold")
axes[1,0].set_xlabel("Époque"); axes[1,0].set_ylabel("Loss")
axes[1,0].legend(); axes[1,0].spines[["top","right"]].set_visible(False)

axes[1,1].plot(hist_mm.history["accuracy"],     label="Train acc",  color="#9b59b6")
axes[1,1].plot(hist_mm.history["val_accuracy"], label="Val acc",    color="#e67e22", linestyle="--")
axes[1,1].set_title("CNN Multimodal — Accuracy", fontweight="bold")
axes[1,1].set_xlabel("Époque"); axes[1,1].set_ylabel("Accuracy")
axes[1,1].set_ylim(0, 1.05); axes[1,1].legend()
axes[1,1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig8_courbes_apprentissage.png", dpi=150, bbox_inches="tight")
plt.close()

# 8. FIGURE 9 — COMPARAISON BARPLOT IMAGE vs MULTIMODAL
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 9 — Comparaison CNN image seul vs multimodal",
             fontsize=13, fontweight="bold")

for ax, metric, vals, title in zip(
    axes,
    ["Accuracy", "AUC-ROC"],
    [[acc_img, acc_mm], [auc_img, auc_mm]],
    ["Accuracy", "AUC-ROC"]
):
    bars = ax.bar(["Image seul", "Multimodal"], vals,
                  color=["#3498db", "#9b59b6"], alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig9_comparaison_image_vs_multimodal.png",
            dpi=150, bbox_inches="tight")
plt.close()


# 9. SAUVEGARDE DES MODÈLES
model_img.save(os.path.join(MODEL_DIR, "modele2a_cnn_image.keras"))
model_mm.save(os.path.join(MODEL_DIR,  "modele2b_cnn_multimodal.keras"))
print(f"\n modele2a_cnn_image.keras     sauvegardé")
print(f" modele2b_cnn_multimodal.keras sauvegardé")

"""
app.py — Interface Streamlit Premium Light
Détection du Cancer Pulmonaire — IA Multimodale
Design : Clean white / Medical precision
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image

st.set_page_config(
    page_title="PulmoAI — Détection Cancer Pulmonaire",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #f8f8fc;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] { display: none; }

/* ── NAV ── */
.pulmo-nav {
    position: sticky; top: 0; z-index: 100;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid #e8e8f4;
    padding: 0 56px;
    display: flex; align-items: center; justify-content: space-between;
    height: 64px;
    box-shadow: 0 1px 20px rgba(109,40,217,0.06);
}
.pulmo-nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: #1a1a2e; letter-spacing: -0.5px;
}
.pulmo-nav-logo span { color: #7c3aed; }
.pulmo-nav-right { display: flex; align-items: center; gap: 12px; }
.pulmo-nav-badge {
    background: #f3f0ff;
    border: 1px solid #ddd6fe;
    color: #7c3aed;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500;
}
.pulmo-nav-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px rgba(16,185,129,0.5);
    display: inline-block; margin-right: 4px;
}

/* ── HERO ── */
.pulmo-hero {
    padding: 64px 56px 48px;
    text-align: center;
    background: linear-gradient(180deg, #ffffff 0%, #f8f8fc 100%);
    border-bottom: 1px solid #e8e8f4;
}
.pulmo-hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    background: #f3f0ff; border: 1px solid #ddd6fe;
    color: #7c3aed; padding: 6px 18px; border-radius: 30px;
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    margin-bottom: 24px;
}
.pulmo-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800; line-height: 1.15;
    color: #1a1a2e; margin-bottom: 16px;
    letter-spacing: -1.5px;
}
.pulmo-hero h1 .accent { color: #7c3aed; }
.pulmo-hero p {
    font-size: 1rem; color: #6b6b85;
    max-width: 560px; margin: 0 auto 20px;
    line-height: 1.7; font-weight: 300;
}
.pulmo-disclaimer {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fffbeb; border: 1px solid #fde68a;
    color: #92400e; padding: 7px 16px; border-radius: 8px;
    font-size: 0.78rem; font-weight: 500;
}

/* ── STATS ── */
.pulmo-stats {
    display: flex; justify-content: center;
    margin: 0 56px;
    background: #ffffff;
    border: 1px solid #e8e8f4;
    border-radius: 20px; overflow: hidden;
    box-shadow: 0 4px 24px rgba(109,40,217,0.06);
    transform: translateY(-20px);
}
.pulmo-stat {
    flex: 1; padding: 24px 28px; text-align: center;
    border-right: 1px solid #f0f0f8;
    transition: background 0.2s;
}
.pulmo-stat:last-child { border-right: none; }
.pulmo-stat:hover { background: #f8f6ff; }
.pulmo-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: #1a1a2e; display: block; margin-bottom: 4px;
}
.pulmo-stat-num .unit { color: #7c3aed; }
.pulmo-stat-label {
    font-size: 0.75rem; color: #9898b0;
    text-transform: uppercase; letter-spacing: 0.8px;
}

/* ── MAIN GRID ── */
.pulmo-wrap {
    padding: 8px 40px 56px;
    max-width: 1400px; margin: 0 auto;
}

/* ── CARDS ── */
.pulmo-card {
    background: #ffffff;
    border: 1px solid #e8e8f4;
    border-radius: 20px; padding: 28px 28px 28px 28px;
    box-shadow: 0 2px 16px rgba(109,40,217,0.05);
    margin-bottom: 20px;
    transition: box-shadow 0.3s, transform 0.2s;
}
.pulmo-card:hover {
    box-shadow: 0 8px 32px rgba(109,40,217,0.1);
    transform: translateY(-1px);
}
.pulmo-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #1a1a2e; margin-bottom: 24px;
    display: flex; align-items: center; gap: 10px;
}
.pulmo-card-icon {
    width: 36px; height: 36px;
    background: #f3f0ff; border: 1px solid #ddd6fe;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.pulmo-section-label {
    font-size: 0.68rem; font-weight: 600; color: #9898b0;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin-bottom: 12px; margin-top: 20px;
}
.pulmo-section-label:first-child { margin-top: 0; }
.pulmo-divider {
    border: none; border-top: 1px solid #f0f0f8; margin: 20px 0;
}

/* ── RESULT BOXES ── */
.result-cancer {
    background: #fff5f5; border: 1px solid #fecaca;
    border-left: 4px solid #dc2626;
    border-radius: 12px; padding: 18px 22px; margin: 14px 0;
}
.result-ok {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 12px; padding: 18px 22px; margin: 14px 0;
}
.result-warn {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #d97706;
    border-radius: 12px; padding: 18px 22px; margin: 14px 0;
}
.result-info {
    background: #f8f6ff; border: 1px solid #ddd6fe;
    border-left: 4px solid #7c3aed;
    border-radius: 12px; padding: 18px 22px; margin: 14px 0;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    margin-bottom: 5px; color: #1a1a2e;
}
.result-sub { font-size: 0.84rem; color: #6b6b85; line-height: 1.5; }

/* ── PROBA CARDS ── */
.proba-grid { display: flex; gap: 12px; margin: 16px 0; }
.proba-card {
    flex: 1; text-align: center;
    background: #f8f8fc; border: 2px solid #e8e8f4;
    border-radius: 14px; padding: 18px 8px;
    transition: all 0.25s;
}
.proba-card.active-ok   { border-color: #16a34a; background: #f0fdf4; }
.proba-card.active-warn { border-color: #d97706; background: #fffbeb; }
.proba-card.active-red  { border-color: #dc2626; background: #fff5f5; }
.proba-icon  { font-size: 1.1rem; margin-bottom: 6px; }
.proba-label { font-size: 0.68rem; color: #9898b0; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.proba-value { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #1a1a2e; }

/* ── PROGRESS BAR ── */
.prog-wrap { margin: 10px 0 6px; }
.prog-label { font-size: 0.75rem; color: #9898b0; margin-bottom: 6px; }
.prog-track { background: #f0f0f8; border-radius: 8px; height: 8px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 8px; transition: width 1s ease; }
.prog-val   { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: #1a1a2e; margin-top: 6px; }

/* ── SYNTH TABLE ── */
.synth-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.synth-table th {
    font-size: 0.68rem; color: #9898b0; text-transform: uppercase;
    letter-spacing: 0.8px; padding: 8px 12px; text-align: left;
    border-bottom: 2px solid #f0f0f8;
}
.synth-table td {
    padding: 11px 12px; font-size: 0.84rem; color: #3a3a5c;
    border-bottom: 1px solid #f8f8fc;
}
.synth-table tr:last-child td { border-bottom: none; }
.synth-table tr:hover td { background: #f8f6ff; }
.badge-green { background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.badge-red   { background: #fee2e2; color: #dc2626; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.badge-gold  { background: #fef3c7; color: #b45309; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.badge-purple{ background: #f3f0ff; color: #7c3aed; padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }

/* ── UPLOAD ── */
.upload-placeholder {
    border: 2px dashed #ddd6fe; border-radius: 14px;
    padding: 36px 24px; text-align: center;
    background: #faf8ff; transition: all 0.25s;
}
.upload-placeholder:hover { border-color: #7c3aed; background: #f5f0ff; }
.upload-icon { font-size: 2.4rem; margin-bottom: 10px; }
.upload-text { font-size: 0.9rem; color: #6b6b85; font-weight: 500; }
.upload-sub  { font-size: 0.75rem; color: #b0b0c8; margin-top: 4px; }

/* ── CTA BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; padding: 14px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(109,40,217,0.3) !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(109,40,217,0.45) !important;
}
.stButton > button:disabled {
    background: #e8e8f4 !important; color: #9898b0 !important;
    box-shadow: none !important; transform: none !important;
}

/* ── WIDGET OVERRIDES ── */
[data-testid="stSlider"] label,
.stCheckbox > label,
.stRadio > label,
.stRadio > div > div > label { color: #3a3a5c !important; font-size: 0.87rem !important; }

[data-testid="stFileUploader"] {
    background: #faf8ff !important;
    border: 2px dashed #ddd6fe !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label { color: #7c3aed !important; font-weight: 500 !important; }

.stExpander {
    background: #faf8ff !important;
    border: 1px solid #e8e8f4 !important;
    border-radius: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #1a1a2e !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stMetricLabel"] { color: #9898b0 !important; font-size: 0.78rem !important; }

[data-testid="stTabs"] [role="tab"] { color: #9898b0 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #7c3aed !important; border-bottom: 2px solid #7c3aed !important;
}

.stDataFrame { border-radius: 10px !important; border: 1px solid #e8e8f4 !important; }

/* ── FOOTER ── */
.pulmo-footer {
    text-align: center; padding: 36px 56px;
    border-top: 1px solid #e8e8f4;
    color: #b0b0c8; font-size: 0.78rem;
    background: #ffffff;
}
.pulmo-footer strong { color: #1a1a2e; }
.pulmo-footer span   { color: #7c3aed; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles IA...")
def load_models():
    model1 = model2a = model2b = None
    try:
        model1 = joblib.load("models/modele1_meilleur.pkl")
    except Exception as e:
        st.error(f"❌ Modèle 1 : {e}")
        return None, None, None
    try:
        import tensorflow as tf
        model2a = tf.keras.models.load_model("models/modele2a_cnn_image.keras")
        model2b = tf.keras.models.load_model("models/modele2b_cnn_multimodal.keras")
    except Exception as e:
        st.warning(f"⚠️ CNN : {e}")
    return model1, model2a, model2b

model1, model2a, model2b = load_models()

FEATURES = [
    "age", "sexe_masculin", "presence_nodule", "subtilite_nodule",
    "taille_nodule_px", "x_nodule_norm", "y_nodule_norm",
    "tabagisme_paquets_annee", "toux_chronique", "dyspnee",
    "douleur_thoracique", "perte_poids", "spo2", "antecedent_familial"
]
IMG_SIZE = (64, 64)

# ══════════════════════════════════════════════════════════════
# NAV
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="pulmo-nav">
    <div class="pulmo-nav-logo">Pulmo<span>AI</span></div>
    <div class="pulmo-nav-right">
        <span><span class="pulmo-nav-dot"></span>Système actif</span>
        <div class="pulmo-nav-badge">M2 ESIC — TP Noté 2025–2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="pulmo-hero">
    <div class="pulmo-hero-tag">🤖 Intelligence Artificielle Médicale</div>
    <h1>Detecting Tomorrow.<br><span class="accent">Diagnosing Today.</span></h1>
    <p>Pipeline IA multimodal combinant données cliniques et imagerie thoracique
    pour estimer le risque de cancer pulmonaire.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STATS BAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="pulmo-stats">
    <div class="pulmo-stat">
        <span class="pulmo-stat-num">184<span class="unit">+</span></span>
        <span class="pulmo-stat-label">Patients — Dataset JSRT</span>
    </div>
    <div class="pulmo-stat">
        <span class="pulmo-stat-num">3</span>
        <span class="pulmo-stat-label">Modèles IA interconnectés</span>
    </div>
    <div class="pulmo-stat">
        <span class="pulmo-stat-num">100<span class="unit">%</span></span>
        <span class="pulmo-stat-label">F1-Score Modèle 1</span>
    </div>
    <div class="pulmo-stat">
        <span class="pulmo-stat-num">14</span>
        <span class="pulmo-stat-label">Features cliniques</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MAIN — 2 colonnes
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="pulmo-wrap">', unsafe_allow_html=True)
col_left, col_right = st.columns(2, gap="large")

# ════════════════════════════════════
# GAUCHE — Saisie patient
# ════════════════════════════════════
with col_left:
    # Card données cliniques
    st.markdown("""
    <div class="pulmo-card">
        <div class="pulmo-card-title">
            <div class="pulmo-card-icon">👤</div>
            Données cliniques du patient
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pulmo-section-label">Profil général</div>', unsafe_allow_html=True)
    age = st.slider("Âge", 20, 85, 60, format="%d ans")
    sexe = st.radio("Sexe biologique", ["Féminin", "Masculin"], horizontal=True)
    sexe_masculin = 1 if sexe == "Masculin" else 0
    tabagisme = st.slider("Tabagisme cumulé", 0, 50, 25, format="%d paquets-années")
    antecedent = st.checkbox("Antécédent familial de cancer du poumon")

    st.markdown('<hr class="pulmo-divider">', unsafe_allow_html=True)
    st.markdown('<div class="pulmo-section-label">Nodule pulmonaire</div>', unsafe_allow_html=True)
    presence = st.checkbox("Nodule détecté à l'imagerie", value=True)
    if presence:
        subtilite = st.slider("Subtilité du nodule", 1, 5, 3,
                              help="1 = très subtil → 5 = évident")
        taille_px = 1
        cx, cy = st.columns(2)
        with cx: x_norm = st.slider("Position X (normalisée)", 0.0, 1.0, 0.5, 0.01)
        with cy: y_norm = st.slider("Position Y (normalisée)", 0.0, 1.0, 0.4, 0.01)
    else:
        subtilite, taille_px, x_norm, y_norm = 1, 0, 0.0, 0.0

    st.markdown('<hr class="pulmo-divider">', unsafe_allow_html=True)
    st.markdown('<div class="pulmo-section-label">Symptômes & Biologie</div>', unsafe_allow_html=True)
    spo2 = st.slider("SpO2 — Saturation en oxygène", 85, 100, 95,
                     format="%d%%", help="Normale ≥ 95%")
    c1, c2 = st.columns(2)
    with c1:
        toux        = st.checkbox("Toux chronique")
        dyspnee     = st.checkbox("Dyspnée")
    with c2:
        douleur     = st.checkbox("Douleur thoracique")
        perte_poids = st.checkbox("Perte de poids")

    st.markdown('</div>', unsafe_allow_html=True)

    # Card radio
    st.markdown("""
    <div class="pulmo-card">
        <div class="pulmo-card-title">
            <div class="pulmo-card-icon">📷</div>
            Radio thoracique
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Glisser-déposer ou sélectionner",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="✅ Radio chargée", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-placeholder">
            <div class="upload-icon">🫁</div>
            <div class="upload-text">Chargez une radio thoracique</div>
            <div class="upload-sub">JPG ou PNG · Active le Modèle 2</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# DROITE — Résultats
# ════════════════════════════════════
with col_right:
    st.markdown("""
    <div class="pulmo-card">
        <div class="pulmo-card-title">
            <div class="pulmo-card-icon">🧠</div>
            Analyse IA — Résultats
        </div>
    """, unsafe_allow_html=True)

    btn = st.button(
        "⚡  Lancer l'analyse complète",
        disabled=(model1 is None),
        use_container_width=True
    )

    if not btn:
        st.markdown("""
        <div class="result-info" style="margin-top:16px">
            <div class="result-title">En attente d'analyse</div>
            <div class="result-sub">Renseignez les données cliniques à gauche,
            chargez une radio thoracique, puis lancez l'analyse.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Features ──────────────────────────────────────────
        fd = {
            "age": age, "sexe_masculin": sexe_masculin,
            "presence_nodule": int(presence), "subtilite_nodule": subtilite,
            "taille_nodule_px": taille_px, "x_nodule_norm": x_norm,
            "y_nodule_norm": y_norm, "tabagisme_paquets_annee": tabagisme,
            "toux_chronique": int(toux), "dyspnee": int(dyspnee),
            "douleur_thoracique": int(douleur), "perte_poids": int(perte_poids),
            "spo2": spo2, "antecedent_familial": int(antecedent),
        }
        X = pd.DataFrame([fd])[FEATURES]
        proba_m1 = model1.predict_proba(X)[0]
        pred_m1  = int(model1.predict(X)[0])
        p0, p1, p2 = proba_m1

        # ── Modèle 1 ──────────────────────────────────────────
        st.markdown('<div class="pulmo-section-label">Modèle 1 — Risque de malignité</div>',
                    unsafe_allow_html=True)

        a0 = "active-ok"   if pred_m1 == 0 else ""
        a1 = "active-warn" if pred_m1 == 1 else ""
        a2 = "active-red"  if pred_m1 == 2 else ""

        st.markdown(f"""
        <div class="proba-grid">
            <div class="proba-card {a0}">
                <div class="proba-icon">🟢</div>
                <div class="proba-label">Faible</div>
                <div class="proba-value">{p0*100:.1f}%</div>
            </div>
            <div class="proba-card {a1}">
                <div class="proba-icon">🟡</div>
                <div class="proba-label">Intermédiaire</div>
                <div class="proba-value">{p1*100:.1f}%</div>
            </div>
            <div class="proba-card {a2}">
                <div class="proba-icon">🔴</div>
                <div class="proba-label">Élevé</div>
                <div class="proba-value">{p2*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        r_map = {
            0: ("result-ok",
                "🟢 Risque FAIBLE",
                "Aucun facteur de risque majeur identifié dans le profil clinique."),
            1: ("result-warn",
                "🟡 Risque INTERMÉDIAIRE",
                "Facteurs de risque présents. Surveillance radiologique recommandée."),
            2: ("result-cancer",
                "🔴 Risque ÉLEVÉ",
                "Profil fortement associé à une malignité. Consultation spécialisée urgente."),
        }
        cls, title, sub = r_map[pred_m1]
        st.markdown(f"""
        <div class="{cls}">
            <div class="result-title">{title}</div>
            <div class="result-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔎 Voir le détail des 14 features"):
            st.dataframe(
                pd.DataFrame(fd.items(), columns=["Variable", "Valeur"]),
                use_container_width=True, hide_index=True
            )

        st.markdown('<hr class="pulmo-divider">', unsafe_allow_html=True)

        # ── Modèle 2 ──────────────────────────────────────────
        st.markdown('<div class="pulmo-section-label">Modèle 2 — Diagnostic image CNN</div>',
                    unsafe_allow_html=True)

        if uploaded is None:
            st.markdown("""
            <div class="result-warn">
                <div class="result-title">⚠️ Radio manquante</div>
                <div class="result-sub">Chargez une radio thoracique pour activer le Modèle 2.</div>
            </div>
            """, unsafe_allow_html=True)

        elif model2a is None or model2b is None:
            st.markdown("""
            <div class="result-cancer">
                <div class="result-title">❌ Modèles CNN indisponibles</div>
                <div class="result-sub">Vérifiez que les fichiers .keras sont dans models/</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            img_arr   = np.array(img_pil.resize(IMG_SIZE), dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_arr, 0)
            p_tab     = proba_m1.reshape(1, -1)

            p2a    = float(model2a.predict(img_batch, verbose=0)[0][0])
            p2b    = float(model2b.predict([img_batch, p_tab], verbose=0)[0][0])
            pred2a = int(p2a > 0.5)
            pred2b = int(p2b > 0.5)

            tab_a, tab_b = st.tabs([
                "📷 Image seule (2A)",
                "⭐ Multimodal (2B) — Recommandé"
            ])

            def render_cnn(tab, proba, pred, note):
                with tab:
                    pct = proba * 100
                    color = "#dc2626" if pred == 1 else "#16a34a"
                    st.markdown(f"""
                    <div class="prog-wrap">
                        <div class="prog-label">P(Cancer)</div>
                        <div class="prog-track">
                            <div class="prog-fill"
                                 style="width:{int(pct)}%; background:{color};"></div>
                        </div>
                        <div class="prog-val">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if pred == 1:
                        st.markdown("""
                        <div class="result-cancer">
                            <div class="result-title">🔴 Cancer PROBABLE</div>
                            <div class="result-sub">Patterns malins détectés sur l'imagerie.</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-ok">
                            <div class="result-title">🟢 Cancer NON PROBABLE</div>
                            <div class="result-sub">Pas de pattern tumoral détecté.</div>
                        </div>""", unsafe_allow_html=True)
                    st.caption(note)

            render_cnn(tab_a, p2a, pred2a,
                       "ℹ️ Accuracy : 54.05% | AUC : 0.38 — Limité sans contexte clinique")
            render_cnn(tab_b, p2b, pred2b,
                       "✅ Accuracy : 100% | AUC : 1.00 — Fusion image + clinique")

            # ── Synthèse ──────────────────────────────────────
            st.markdown('<hr class="pulmo-divider">', unsafe_allow_html=True)
            st.markdown('<div class="pulmo-section-label">Synthèse globale</div>',
                        unsafe_allow_html=True)

            def badge(val, kind="cancer"):
                if kind == "risk":
                    m = {0:("badge-green","Faible"), 1:("badge-gold","Intermédiaire"), 2:("badge-red","Élevé")}
                    cls, lbl = m[val]
                else:
                    cls = "badge-red" if val == 1 else "badge-green"
                    lbl = "Cancer probable" if val == 1 else "Non cancer"
                return f'<span class="{cls}">{lbl}</span>'

            st.markdown(f"""
            <table class="synth-table">
                <thead>
                    <tr><th>Modèle</th><th>Résultat</th><th>Confiance</th><th>Fiabilité</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Modèle 1 — Random Forest</td>
                        <td>{badge(pred_m1, "risk")}</td>
                        <td>{max(proba_m1)*100:.1f}%</td>
                        <td><span class="badge-purple">F1 = 1.00</span></td>
                    </tr>
                    <tr>
                        <td>CNN Image seule (2A)</td>
                        <td>{badge(pred2a)}</td>
                        <td>{max(p2a,1-p2a)*100:.1f}%</td>
                        <td><span class="badge-gold">AUC = 0.38</span></td>
                    </tr>
                    <tr>
                        <td>CNN Multimodal (2B) ⭐</td>
                        <td>{badge(pred2b)}</td>
                        <td>{max(p2b,1-p2b)*100:.1f}%</td>
                        <td><span class="badge-purple">AUC = 1.00</span></td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pulmo-footer">
    <strong>PulmoAI</strong> — Pipeline IA Multimodal &nbsp;·&nbsp;
    M2 ESIC Intelligence Artificielle 2025–2026 &nbsp;·&nbsp;
    <span>Sanae Najimi</span> &nbsp;·&nbsp;
    Déployé sur <span>Render</span> via GitHub Actions CI/CD
</div>
""", unsafe_allow_html=True)
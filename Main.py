import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_nav_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURES = {
    "Pregnancies":              {"label": "Pregnancies",               "min": 0,   "max": 20,   "step": 1,    "default": 3,    "unit": "count",    "fmt": "%.0f"},
    "Glucose":                  {"label": "Glucose",                   "min": 50,  "max": 250,  "step": 1,    "default": 110,  "unit": "mg/dL",    "fmt": "%.0f"},
    "BloodPressure":            {"label": "Blood Pressure (Diastolic)","min": 30,  "max": 130,  "step": 1,    "default": 72,   "unit": "mm Hg",    "fmt": "%.0f"},
    "SkinThickness":            {"label": "Skin Thickness",            "min": 0,   "max": 100,  "step": 1,    "default": 20,   "unit": "mm",       "fmt": "%.0f"},
    "Insulin":                  {"label": "Insulin (2-Hour Serum)",    "min": 0,   "max": 900,  "step": 1,    "default": 80,   "unit": "μU/mL",    "fmt": "%.0f"},
    "BMI":                      {"label": "BMI",                       "min": 10.0,"max": 70.0, "step": 0.1,  "default": 28.0, "unit": "kg/m²",    "fmt": "%.1f"},
    "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree Func.",   "min": 0.0, "max": 2.5,  "step": 0.01, "default": 0.45, "unit": "",         "fmt": "%.2f"},
    "Age":                      {"label": "Age",                       "min": 18,  "max": 100,  "step": 1,    "default": 35,   "unit": "years",    "fmt": "%.0f"},
}

# Healthy reference means (class 0 = no diabetes)
HEALTHY_MEAN = {
    "Pregnancies": 6.7, "Glucose": 101.3, "BloodPressure": 65.3,
    "SkinThickness": 19.8, "Insulin": 98.7, "BMI": 31.3,
    "DiabetesPedigreeFunction": 0.45, "Age": 49.2,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1a56db; margin-bottom:0}
    .sub-header  {font-size:1rem; color:#6b7280; margin-top:0; margin-bottom:1.5rem}
    .risk-high   {background:#fee2e2; border:2px solid #ef4444; border-radius:12px; padding:1.2rem; text-align:center}
    .risk-low    {background:#dcfce7; border:2px solid #22c55e; border-radius:12px; padding:1.2rem; text-align:center}
    .risk-label  {font-size:1.6rem; font-weight:800; margin:0}
    .risk-prob   {font-size:1.1rem; margin:0.3rem 0 0}
    .metric-box  {background:#f3f4f6; border-radius:8px; padding:0.8rem 1rem}
    .disclaimer  {font-size:0.75rem; color:#9ca3af; margin-top:1rem}
    div[data-testid="stSlider"] > label {font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🩺 Diabetes Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Gaussian Naïve Bayes · Pima Indians Diabetes Dataset · 8 Clinical Features</p>', unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ── Left: Inputs ──────────────────────────────────────────────────────────────
with left_col:
    st.subheader("Patient Parameters")
    user_vals = {}
    for feat, meta in FEATURES.items():
        val = st.slider(
            f"{meta['label']}  ({meta['unit']})" if meta["unit"] else meta["label"],
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=float(meta["default"]),
            step=float(meta["step"]),
            format=meta["fmt"],
            key=feat,
        )
        user_vals[feat] = val

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Prediction logic ──────────────────────────────────────────────────────────
input_arr = np.array([[user_vals[f] for f in model.feature_names_in_]])
proba = model.predict_proba(input_arr)[0]
pred_class = model.predict(input_arr)[0]
prob_diabetic = proba[1]
prob_healthy = proba[0]

# ── Right: Results ────────────────────────────────────────────────────────────
with right_col:

    # ── Risk card ──────────────────────────────────────────────────────────────
    st.subheader("Prediction Result")
    if pred_class == 1:
        st.markdown(f"""
        <div class="risk-high">
            <p class="risk-label" style="color:#dc2626">⚠️ High Diabetes Risk</p>
            <p class="risk-prob" style="color:#7f1d1d">Probability: <b>{prob_diabetic*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <p class="risk-label" style="color:#16a34a">✅ Low Diabetes Risk</p>
            <p class="risk-prob" style="color:#14532d">Probability of diabetes: <b>{prob_diabetic*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Probability gauge (matplotlib) ────────────────────────────────────────
    col_g, col_m = st.columns([1.3, 1])

    with col_g:
        fig, ax = plt.subplots(figsize=(4, 0.55))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        bar_bg   = ax.barh(0, 1, color="#e5e7eb", height=0.55, left=0)
        bar_fill = ax.barh(0, prob_diabetic, color="#ef4444" if prob_diabetic >= 0.5 else "#22c55e", height=0.55, left=0)
        ax.axvline(0.5, color="#6b7280", linewidth=1.5, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.text(prob_diabetic / 2, 0, f"{prob_diabetic*100:.0f}%",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(0, -0.42, "0%", ha="left",   va="top", fontsize=7, color="#9ca3af")
        ax.text(0.5,-0.42, "50%",ha="center",va="top", fontsize=7, color="#9ca3af")
        ax.text(1,  -0.42, "100%",ha="right",va="top", fontsize=7, color="#9ca3af")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_m:
        st.metric("P(Diabetic)",  f"{prob_diabetic*100:.1f}%")
        st.metric("P(Healthy)",   f"{prob_healthy*100:.1f}%")

    # ── Feature comparison chart ───────────────────────────────────────────────
    st.subheader("Your Values vs. Healthy Average")

    feat_labels   = [FEATURES[f]["label"] for f in model.feature_names_in_]
    user_v        = np.array([user_vals[f] for f in model.feature_names_in_])
    healthy_v     = np.array([HEALTHY_MEAN[f] for f in model.feature_names_in_])

    # Normalise relative to healthy mean for radar-like bar comparison
    ratio = np.where(healthy_v != 0, user_v / healthy_v, 1.0)

    fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("#f9fafb")

    x      = np.arange(len(feat_labels))
    width  = 0.35
    colors = ["#ef4444" if r > 1.15 else "#22c55e" if r < 0.85 else "#60a5fa" for r in ratio]

    ax2.bar(x - width/2, healthy_v, width, label="Healthy avg.", color="#94a3b8", alpha=0.7)
    bars = ax2.bar(x + width/2, user_v,    width, label="Your value",   color=colors, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(feat_labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Value", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)

    red_p   = mpatches.Patch(color="#ef4444", label=">15% above avg")
    green_p = mpatches.Patch(color="#22c55e", label=">15% below avg")
    blue_p  = mpatches.Patch(color="#60a5fa", label="Within range")
    gray_p  = mpatches.Patch(color="#94a3b8", label="Healthy avg.")
    ax2.legend(handles=[gray_p, red_p, green_p, blue_p],
               fontsize=7.5, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── Feature detail table ───────────────────────────────────────────────────
    with st.expander("📋 Full Feature Breakdown"):
        rows = []
        for f in model.feature_names_in_:
            uv = user_vals[f]
            hv = HEALTHY_MEAN[f]
            diff_pct = ((uv - hv) / hv * 100) if hv != 0 else 0
            flag = "🔴 High" if diff_pct > 15 else ("🟢 Low" if diff_pct < -15 else "🔵 Normal")
            rows.append({
                "Feature":        FEATURES[f]["label"],
                "Your Value":     f"{uv:.2f} {FEATURES[f]['unit']}".strip(),
                "Healthy Avg.":   f"{hv:.2f} {FEATURES[f]['unit']}".strip(),
                "Difference":     f"{diff_pct:+.1f}%",
                "Status":         flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model info ─────────────────────────────────────────────────────────────
    with st.expander("ℹ️ Model Information"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm",  "Gaussian NB")
        c2.metric("Features",   str(model.n_features_in_))
        c3.metric("Train size", f"{int(model.class_count_.sum())} samples")
        st.caption(f"Class prior — No diabetes: {model.class_prior_[0]*100:.1f}%  |  Diabetes: {model.class_prior_[1]*100:.1f}%")

    st.markdown('<p class="disclaimer">⚠️ This tool is for educational/demo purposes only and is not a medical device. Always consult a qualified healthcare professional for clinical decisions.</p>', unsafe_allow_html=True)
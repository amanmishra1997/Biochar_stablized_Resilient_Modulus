# ==========================================================
# PI-MS + PHYSICS + SHAP GUI
# FINAL PROFESSIONAL UI VERSION
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="MR Prediction System",
    layout="centered"
)

# ==========================================================
# GLOBAL CSS (INPUT + RESULTS FONT FIX)
# ==========================================================

st.markdown("""
<style>

/* Input Labels */

label {
font-size:20px !important;
font-weight:bold !important;
color:white !important;
}

/* Slider Numbers */

.stSlider div {
font-size:18px !important;
font-weight:bold !important;
color:black !important;
}

/* Section Titles */

h3 {
font-size:28px !important;
font-weight:bold !important;
color:black !important;
}

/* Button */

.stButton > button {
font-size:22px;
font-weight:bold;
height:55px;
}

/* General Text */

p {
font-size:20px !important;
font-weight:bold !important;
color:black !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# MATPLOTLIB FONT STYLE
# ==========================================================

plt.rcParams.update({
    'font.size': 13,
    'font.weight': 'bold'
})

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""

<div style="background:#003366;
padding:20px;
border-radius:12px;
text-align:center;
margin-bottom:25px;">

<div style="font-size:42px;
font-weight:bold;
color:white;">

🧠 PI-MS + Physics MR Prediction System

</div>

<div style="font-size:22px;
color:#d9e6f2;
margin-top:10px;">

📊 Hybrid Intelligent Soil Model  
📈 With Confidence Interval & SHAP Explainability

</div>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    phys_model = joblib.load("physics_model.pkl")

    model = tf.keras.models.load_model(
        "pi_ms_model.h5",
        compile=False
    )

    s1 = joblib.load("scaler_X1.pkl")
    s2 = joblib.load("scaler_X2.pkl")
    s3 = joblib.load("scaler_X3.pkl")

    physics_stats = joblib.load(
        "physics_stats.pkl"
    )

    data = pd.read_excel(
        "MR-140_physics_corrected.xlsx"
    )

    data.columns = [
        'BC','PI','gd','w',
        'curing','FT',
        'sc','sd','MR'
    ]

    return (
        phys_model,
        model,
        s1,
        s2,
        s3,
        physics_stats,
        data
    )

phys_model,model,s1,s2,s3,physics_stats,data = load_models()

# ==========================================================
# LOAD SHAP
# ==========================================================

@st.cache_resource
def load_shap():

    shap_model = tf.keras.models.load_model(
        "Final_PI_MS_Modelamanji.h5",
        compile=False
    )

    background = joblib.load(
        "Final_SHAP_Backgroundamanji.pkl"
    )

    return shap_model, background

shap_model, background = load_shap()

# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown("### 📥 Input Parameters")

BC = st.slider("Biochar Content (%)",0.0,10.0,2.0,0.1)
PI = st.slider("Plasticity Index",5.0,40.0,18.0,0.1)
gd = st.slider("Dry Density (g/cc)",1.5,2.2,1.78,0.01)
w  = st.slider("Moisture Content (%)",5.0,30.0,18.0,0.1)
curing = st.slider("Curing Days",1,28,1)
FT = st.slider("Freeze-Thaw Cycles",0,12,0)
sc = st.slider("Confining Stress σ₃ (kPa)",10.0,100.0,27.6,0.1)
sd = st.slider("Deviator Stress σd (kPa)",10.0,100.0,27.6,0.1)

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("Predict Resilient Modulus (MR)"):

    pa=100
    theta=sc*3

    PI_s=(PI-physics_stats["PI_mean"])/physics_stats["PI_std"]
    gd_s=(gd-physics_stats["gd_mean"])/physics_stats["gd_std"]
    w_s=(w-physics_stats["w_mean"])/physics_stats["w_std"]

    curing_term=(BC/100)*np.log(curing+1)
    FT_effect=-FT

    X_phys=np.array([[

        np.log(theta/pa),
        np.log(sd/pa),
        PI_s,
        gd_s,
        w_s,
        curing_term,
        FT_effect

    ]])

    Y_phys=phys_model.predict(X_phys)[0]

    MR_phys=np.exp(Y_phys)

    theta_val=sc*3

    X_base=np.array([[

        BC,PI,gd,w,
        curing,FT,
        sc,sd,
        theta_val

    ]],dtype=np.float32)

    X1=s1.transform(X_base)
    X2=s2.transform(X_base**2)

    stress_ratio=sd/(sc+1e-6)
    density_moisture=gd*w

    X3=s3.transform(
        np.array([[stress_ratio,density_moisture]])
    )

    residual=model.predict(
        [X1,X2,X3],
        verbose=0
    )[0][0]

    MR_pims=np.exp(Y_phys+residual)

# ======================================================
# CONFIDENCE INTERVAL
# ======================================================

    std_error = np.std(data["MR"])

    phys_lower = MR_phys - 1.96 * std_error
    phys_upper = MR_phys + 1.96 * std_error

    pims_lower = MR_pims - 1.96 * std_error
    pims_upper = MR_pims + 1.96 * std_error

# ======================================================
# BOLD RESULT DISPLAY
# ======================================================

    st.markdown("### 📊 Prediction Results")

    st.markdown(f"""

<div style="font-size:30px;
font-weight:bold;
color:black;">

🔵 Physics Model MR  
{MR_phys:.2f} MPa  
(95% CI: {phys_lower:.2f} – {phys_upper:.2f})

</div>

<br>

<div style="font-size:32px;
font-weight:bold;
color:black;">

🟢 PI-MS Model MR  
{MR_pims:.2f} MPa  
(95% CI: {pims_lower:.2f} – {pims_upper:.2f})

</div>

""", unsafe_allow_html=True)

# ======================================================
# GAUGE
# ======================================================

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=MR_pims,

        title={
            'text': "<b style='color:black'>PI-MS MR Prediction (MPa)</b>",
            'font': {'size': 26}
        },

        number={
            'font': {'size':60,'color':"black"},
            'suffix': " MPa"
        },

        gauge={

            'axis': {
                'range':[0,400],
                'tickfont': {'size':18,'color':'black'}
            },

            'bar': {'color':"#003366"},

            'steps':[

                {'range':[0,100],'color':"#f8c9c9"},
                {'range':[100,200],'color':"#fde49c"},
                {'range':[200,300],'color':"#b9e4c9"},
                {'range':[300,400],'color':"#63c76a"}

            ]

        }

    ))

    st.plotly_chart(fig,use_container_width=True)

# ======================================================
# SHAP
# ======================================================

    st.markdown("## 🧠 SHAP Model Explanation")

    explainer = shap.DeepExplainer(
        shap_model,
        background
    )

    shap_values = explainer.shap_values(
        [X1,X2,X3]
    )

    feature_symbols=[

        "BC (%)",
        "PI",
        "γd",
        "w (%)",
        "t (days)",
        "NFT",
        "σ₃",
        "σd"

    ]

    base_value=explainer.expected_value[0]

    values_fixed=np.array(
        shap_values[0][0]
    ).flatten()

    expl=shap.Explanation(

        values=values_fixed[:8],
        base_values=float(base_value),
        data=X_base[0][:8],
        feature_names=feature_symbols

    )

    fig1=plt.figure(figsize=(9,6))

    shap.plots.waterfall(
        expl,
        max_display=8,
        show=False
    )

    st.pyplot(fig1)
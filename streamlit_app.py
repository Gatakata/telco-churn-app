import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Oracle",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --surface2:  #1a1a24;
    --border:    #2a2a3a;
    --accent:    #00e5ff;
    --accent2:   #ff3d71;
    --accent3:   #a259ff;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --success:   #00e096;
    --warning:   #ffaa00;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0a0a15 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(162,89,255,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 60%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 600px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    margin-top: 1rem;
}

.result-churn {
    background: linear-gradient(135deg, rgba(255,61,113,0.12), rgba(255,61,113,0.05));
    border: 1px solid rgba(255,61,113,0.4);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: pulseRed 2s ease-in-out infinite;
}
.result-safe {
    background: linear-gradient(135deg, rgba(0,224,150,0.12), rgba(0,224,150,0.05));
    border: 1px solid rgba(0,224,150,0.4);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: pulseGreen 2s ease-in-out infinite;
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,61,113,0); }
    50%       { box-shadow: 0 0 24px 4px rgba(255,61,113,0.15); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,224,150,0); }
    50%       { box-shadow: 0 0 24px 4px rgba(0,224,150,0.15); }
}
.result-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
}
.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.25rem 0;
}
.churn-color   { color: var(--accent2); }
.safe-color    { color: var(--success); }

.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.6rem 0;
}
.prob-name {
    font-size: 0.75rem;
    color: var(--muted);
    width: 90px;
    text-align: right;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--surface2);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill-churn {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent2), #ff6b9d);
}
.prob-bar-fill-safe {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--success), #00ffbb);
}
.prob-pct {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    width: 48px;
    flex-shrink: 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.metric-tile {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-lbl {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
}

.risk-bar-container { margin: 1.5rem 0 0.5rem; }
.risk-bar-track {
    width: 100%;
    height: 12px;
    background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--accent2) 100%);
    border-radius: 100px;
    position: relative;
}
.risk-needle {
    position: absolute;
    top: -4px;
    width: 20px; height: 20px;
    background: white;
    border-radius: 50%;
    border: 2px solid var(--bg);
    transform: translateX(-50%);
    box-shadow: 0 0 8px rgba(255,255,255,0.4);
}
.risk-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: var(--muted);
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.sidebar-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.sidebar-section {
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 1.25rem 0 0.5rem;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

label, .stSelectbox label, .stSlider label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">📡 ML-Powered Intelligence</div>
  <div class="hero-title">Churn Oracle</div>
  <div class="hero-sub">
    Real-time telecom customer churn prediction powered by a Random Forest model
    trained on the IBM Telco dataset. Adjust customer attributes in the sidebar
    to get an instant risk assessment.
  </div>
  <div class="hero-badge">✦ Random Forest · SMOTE Balanced · 19 Features</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙ Customer Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    gender        = st.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.selectbox('Senior Citizen', ('No', 'Yes'))
    Partner       = st.selectbox('Partner', ('No', 'Yes'))
    Dependents    = st.selectbox('Dependents', ('No', 'Yes'))

    st.markdown('<div class="sidebar-section">Account</div>', unsafe_allow_html=True)
    tenure           = st.slider('Tenure (months)', 0, 72, 12)
    Contract         = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.selectbox('Paperless Billing', ('No', 'Yes'))
    PaymentMethod    = st.selectbox('Payment Method', (
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.slider('Monthly Charges ($)', 18.0, 120.0, 50.0)
    TotalCharges   = st.slider('Total Charges ($)', 0.0, 9000.0, 500.0)

    st.markdown('<div class="sidebar-section">Services</div>', unsafe_allow_html=True)
    PhoneService     = st.selectbox('Phone Service', ('No', 'Yes'))
    MultipleLines    = st.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    InternetService  = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity   = st.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    OnlineBackup     = st.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
    DeviceProtection = st.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    TechSupport      = st.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    StreamingTV      = st.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    StreamingMovies  = st.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))

# ── Feature engineering ───────────────────────────────────────────────────────
def std(val): return 'No' if 'service' in val.lower() else val

ML_std  = std(MultipleLines);   OS_std  = std(OnlineSecurity)
OB_std  = std(OnlineBackup);    DP_std  = std(DeviceProtection)
TS_std  = std(TechSupport);     STV_std = std(StreamingTV)
SM_std  = std(StreamingMovies)

AvgMonthlyCharge  = TotalCharges / (tenure + 1) if tenure > 0 else MonthlyCharges
TotalServices     = sum([PhoneService=='Yes', ML_std=='Yes', OS_std=='Yes',
                         OB_std=='Yes', DP_std=='Yes', TS_std=='Yes',
                         STV_std=='Yes', SM_std=='Yes'])
PremiumServices   = sum([OS_std=='Yes', OB_std=='Yes', DP_std=='Yes', TS_std=='Yes'])
StreamingServices = sum([STV_std=='Yes', SM_std=='Yes'])

def tenure_cat(t):
    if t <= 12: return 'New'
    if t <= 24: return 'Short-term'
    if t <= 48: return 'Medium-term'
    return 'Long-term'

def charge_cat(c):
    if c <= 40: return 'Low'
    if c <= 70: return 'Medium'
    return 'High'

input_df = pd.DataFrame([{
    'gender': gender, 'SeniorCitizen': SeniorCitizen,
    'Partner': Partner, 'Dependents': Dependents,
    'tenure': tenure, 'PhoneService': PhoneService,
    'MultipleLines': ML_std, 'InternetService': InternetService,
    'OnlineSecurity': OS_std, 'OnlineBackup': OB_std,
    'DeviceProtection': DP_std, 'TechSupport': TS_std,
    'StreamingTV': STV_std, 'StreamingMovies': SM_std,
    'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges,
    'AvgMonthlyCharge': AvgMonthlyCharge,
    'TotalServices': TotalServices, 'PremiumServices': PremiumServices,
    'StreamingServices': StreamingServices,
    'TenureCategory': tenure_cat(tenure),
    'MonthlyChargeCategory': charge_cat(MonthlyCharges),
}])

numerical_cols   = ['tenure','MonthlyCharges','TotalCharges',
                    'AvgMonthlyCharge','TotalServices','PremiumServices','StreamingServices']
categorical_cols = ['gender','SeniorCitizen','Partner','Dependents',
                    'PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies',
                    'Contract','PaperlessBilling','PaymentMethod',
                    'TenureCategory','MonthlyChargeCategory']

@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.joblib')

@st.cache_resource
def build_preprocessor():
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num_pipe, numerical_cols),
                              ('cat', cat_pipe, categorical_cols)])
    ref = pd.DataFrame([
        dict(gender='Male',   SeniorCitizen='No',  Partner='No',  Dependents='No',
             tenure=0,  PhoneService='No',  MultipleLines='No',
             InternetService='DSL',         OnlineSecurity='No',  OnlineBackup='No',
             DeviceProtection='No',         TechSupport='No',
             StreamingTV='No',              StreamingMovies='No',
             Contract='Month-to-month',     PaperlessBilling='No',
             PaymentMethod='Electronic check',
             MonthlyCharges=18.0, TotalCharges=0.0,
             AvgMonthlyCharge=18.0, TotalServices=0, PremiumServices=0, StreamingServices=0,
             TenureCategory='New',          MonthlyChargeCategory='Low'),
        dict(gender='Female', SeniorCitizen='Yes', Partner='Yes', Dependents='Yes',
             tenure=72, PhoneService='Yes', MultipleLines='Yes',
             InternetService='Fiber optic', OnlineSecurity='Yes', OnlineBackup='Yes',
             DeviceProtection='Yes',        TechSupport='Yes',
             StreamingTV='Yes',             StreamingMovies='Yes',
             Contract='Two year',           PaperlessBilling='Yes',
             PaymentMethod='Credit card (automatic)',
             MonthlyCharges=120.0, TotalCharges=9000.0,
             AvgMonthlyCharge=125.0, TotalServices=8, PremiumServices=4, StreamingServices=2,
             TenureCategory='Long-term',    MonthlyChargeCategory='High'),
        dict(gender='Male',   SeniorCitizen='No',  Partner='Yes', Dependents='No',
             tenure=24, PhoneService='Yes', MultipleLines='No',
             InternetService='No',          OnlineSecurity='No',  OnlineBackup='No',
             DeviceProtection='No',         TechSupport='No',
             StreamingTV='No',              StreamingMovies='No',
             Contract='One year',           PaperlessBilling='Yes',
             PaymentMethod='Mailed check',
             MonthlyCharges=50.0, TotalCharges=1200.0,
             AvgMonthlyCharge=48.0, TotalServices=2, PremiumServices=0, StreamingServices=0,
             TenureCategory='Short-term',   MonthlyChargeCategory='Medium'),
        dict(gender='Female', SeniorCitizen='No',  Partner='No',  Dependents='Yes',
             tenure=36, PhoneService='Yes', MultipleLines='Yes',
             InternetService='DSL',         OnlineSecurity='No',  OnlineBackup='Yes',
             DeviceProtection='No',         TechSupport='No',
             StreamingTV='Yes',             StreamingMovies='No',
             Contract='Month-to-month',     PaperlessBilling='No',
             PaymentMethod='Bank transfer (automatic)',
             MonthlyCharges=65.0, TotalCharges=2340.0,
             AvgMonthlyCharge=64.9, TotalServices=4, PremiumServices=1, StreamingServices=1,
             TenureCategory='Medium-term',  MonthlyChargeCategory='Medium'),
    ])
    pre.fit(ref[numerical_cols + categorical_cols])
    return pre

model        = load_model()
preprocessor = build_preprocessor()

# ── Predict & Render ──────────────────────────────────────────────────────────
try:
    processed  = preprocessor.transform(input_df[numerical_cols + categorical_cols])
    prediction = model.predict(processed)[0]
    proba      = model.predict_proba(processed)[0]
    churn_prob = proba[1]
    safe_prob  = proba[0]
    churn_w    = int(churn_prob * 100)
    safe_w     = int(safe_prob  * 100)
    needle_pos = churn_w

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
              <div class="result-icon">⚠️</div>
              <div class="result-label">Prediction</div>
              <div class="result-verdict churn-color">WILL CHURN</div>
              <div style="color:var(--muted);font-size:0.8rem;margin-top:0.5rem;">
                High risk — retention action recommended
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
              <div class="result-icon">✅</div>
              <div class="result-label">Prediction</div>
              <div class="result-verdict safe-color">WILL STAY</div>
              <div style="color:var(--muted);font-size:0.8rem;margin-top:0.5rem;">
                Low risk — relationship looks healthy
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Confidence breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-name">Churn</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill-churn" style="width:{churn_w}%"></div></div>
          <div class="prob-pct churn-color">{churn_w}%</div>
        </div>
        <div class="prob-row">
          <div class="prob-name">Retain</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill-safe" style="width:{safe_w}%"></div></div>
          <div class="prob-pct safe-color">{safe_w}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Risk meter</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-bar-container">
          <div class="risk-bar-track">
            <div class="risk-needle" style="left:{needle_pos}%"></div>
          </div>
          <div class="risk-labels">
            <span>Low Risk</span><span>Medium</span><span>High Risk</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-label">Customer snapshot</div>', unsafe_allow_html=True)
        tc = tenure_cat(tenure)
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-tile"><div class="metric-val">{tenure}</div><div class="metric-lbl">Months</div></div>
          <div class="metric-tile"><div class="metric-val">{TotalServices}</div><div class="metric-lbl">Services</div></div>
          <div class="metric-tile"><div class="metric-val">${MonthlyCharges:.0f}</div><div class="metric-lbl">Monthly</div></div>
          <div class="metric-tile"><div class="metric-val">{PremiumServices}</div><div class="metric-lbl">Premium</div></div>
          <div class="metric-tile"><div class="metric-val">${AvgMonthlyCharge:.0f}</div><div class="metric-lbl">Avg/Mo</div></div>
          <div class="metric-tile"><div class="metric-val">{tc[:3].upper()}</div><div class="metric-lbl">Segment</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">Risk factor signals</div>', unsafe_allow_html=True)

        factors = []
        if Contract == 'Month-to-month':
            factors.append(("🔴", "Month-to-month contract", "High churn risk"))
        elif Contract == 'One year':
            factors.append(("🟡", "One-year contract", "Moderate stability"))
        else:
            factors.append(("🟢", "Two-year contract", "Strong retention"))

        if InternetService == 'Fiber optic':
            factors.append(("🔴", "Fiber optic internet", "Higher churn tendency"))
        elif InternetService == 'No':
            factors.append(("🟢", "No internet service", "Lower churn risk"))

        if tenure <= 12:
            factors.append(("🔴", "New customer", "< 1 year tenure"))
        elif tenure >= 48:
            factors.append(("🟢", "Long-term customer", f"{tenure} months"))

        if PremiumServices >= 3:
            factors.append(("🟢", "Heavy premium user", f"{PremiumServices} premium services"))
        elif PremiumServices == 0:
            factors.append(("🔴", "No premium services", "Low product stickiness"))

        if MonthlyCharges > 80:
            factors.append(("🔴", "High monthly bill", f"${MonthlyCharges:.0f}/mo"))

        fhtml = ""
        for icon, title, desc in factors[:5]:
            fhtml += f"""
            <div style="display:flex;align-items:center;gap:0.75rem;
                        padding:0.6rem 0.75rem;margin:0.4rem 0;
                        background:var(--surface2);border-radius:8px;
                        border:1px solid var(--border);">
              <span style="font-size:1rem">{icon}</span>
              <div>
                <div style="font-size:0.8rem;color:var(--text);font-family:'Syne',sans-serif;font-weight:600">{title}</div>
                <div style="font-size:0.65rem;color:var(--muted)">{desc}</div>
              </div>
            </div>"""
        st.markdown(fhtml, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
                border-top:1px solid var(--border);color:var(--muted);
                font-size:0.7rem;letter-spacing:0.1em;">
      CHURN ORACLE · RANDOM FOREST · IBM TELCO DATASET · BUILT WITH STREAMLIT
    </div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Prediction error: {e}")
    st.code(str(e))

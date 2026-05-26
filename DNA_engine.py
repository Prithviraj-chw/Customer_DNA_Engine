import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer DNA Engine",
    page_icon="🧬",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 720px;
}

/* Title */
.dna-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.dna-subtitle {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.02em;
}

/* Input card */
.input-card {
    background: #13131a;
    border: 1px solid #1f1f2e;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* Result cards */
.result-lost {
    background: linear-gradient(135deg, #1a1020, #2d1b4e);
    border: 1px solid #7c3aed44;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}
.result-loyal {
    background: linear-gradient(135deg, #0d1f2d, #1a3a4a);
    border: 1px solid #0ea5e944;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}
.result-vip {
    background: linear-gradient(135deg, #0f1f18, #1a3d2b);
    border: 1px solid #34d39944;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}

.segment-badge {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.strategy-text {
    font-size: 1rem;
    color: #d1d5db;
    font-weight: 300;
    line-height: 1.6;
}

.rfm-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}

.rfm-chip {
    background: #1f1f2e;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    color: #9ca3af;
}

/* Streamlit number input overrides */
div[data-testid="stNumberInput"] label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

div[data-testid="stNumberInput"] input {
    background: #1a1a26 !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}

div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
}

.divider {
    border: none;
    border-top: 1px solid #1f1f2e;
    margin: 2rem 0;
}

.hint {
    color: #4b5563;
    font-size: 0.8rem;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Segment config ────────────────────────────────────────────────────────────
SEGMENTS = {
    0: {
        "name": "😶 Lost Customer",
        "card_class": "result-lost",
        "color": "#a78bfa",
        "strategy": "We miss you! Here's a 25% off coupon to come back.",
        "desc": "Bought a long time ago, rarely ordered, low spend. High churn risk."
    },
    1: {
        "name": "🤝 Loyal Regular",
        "card_class": "result-loyal",
        "color": "#60a5fa",
        "strategy": "Thank you for being loyal! You get early access to new arrivals.",
        "desc": "Fairly recent, orders regularly, decent spender. Your core customer base."
    },
    2: {
        "name": "🐋 VIP Wholesaler",
        "card_class": "result-vip",
        "color": "#34d399",
        "strategy": "VIP treatment — dedicated account manager + bulk discount.",
        "desc": "Buys very recently, extremely high frequency, massive spend. Likely a wholesaler."
    }
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="dna-title">Customer DNA Engine 🧬</div>', unsafe_allow_html=True)
st.markdown('<div class="dna-subtitle">Enter a customer\'s RFM profile to instantly reveal their segment and targeted promotion.</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure `kmeans_model.pkl` and `scaler.pkl` are in the same folder.\n\n`{load_error}`")
    st.stop()

# Input card
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        max_value=1000,
        value=45,
        help="Days since the customer's last purchase. Lower = more recent."
    )

with col2:
    frequency = st.number_input(
        "Frequency (orders)",
        min_value=1,
        max_value=1000,
        value=5,
        help="Total number of unique orders placed."
    )

with col3:
    monetary = st.number_input(
        "Monetary (£)",
        min_value=0.0,
        max_value=1000000.0,
        value=1200.0,
        step=100.0,
        help="Total £ spent across all orders."
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="hint">💡 Cluster 0 → high recency, low freq/spend &nbsp;|&nbsp; Cluster 1 → balanced &nbsp;|&nbsp; Cluster 2 → very low recency, high freq/spend</p>', unsafe_allow_html=True)

predict_btn = st.button("Analyse Customer →")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    input_arr = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_arr)
    cluster = int(model.predict(input_scaled)[0])

    seg = SEGMENTS[cluster]

    st.markdown(f"""
    <div class="{seg['card_class']}">
        <div class="segment-badge" style="color:{seg['color']}">{seg['name']}</div>
        <div class="strategy-text">{seg['desc']}</div>
        <hr style="border:none;border-top:1px solid #ffffff11;margin:1.2rem 0;">
        <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Recommended Promotion</div>
        <div class="strategy-text" style="color:#f3f4f6;font-weight:500;">🎯 {seg['strategy']}</div>
        <div class="rfm-row">
            <div class="rfm-chip">Recency: {recency}d</div>
            <div class="rfm-chip">Frequency: {frequency} orders</div>
            <div class="rfm-chip">Monetary: £{monetary:,.2f}</div>
            <div class="rfm-chip">Cluster: {cluster}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib, os

# Page Config — MUST be the first Streamlit command in the script,
# before any st.markdown / st.write / etc, or Streamlit will either
# error out or silently ignore layout/sidebar settings.
st.set_page_config(
    page_title="SympAI-Virtual Disease Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "sample_datasets")
DATASET = os.path.join(DATA_DIR, "dataset.csv")
DESC = os.path.join(DATA_DIR, "symptom_description.csv")
PREC = os.path.join(DATA_DIR, "symptom_precaution.csv")
SEV = os.path.join(DATA_DIR, "symptom_severity.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
VOCAB_PATH = os.path.join(DATA_DIR, "symptom_vocab.pkl")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")  # resolved relative to this file, not the CWD

# Custom CSS + Title
st.markdown(
    """
    <style>

    .centered-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #F3E8FF;
        font-family: 'Poppins', sans-serif;

        margin-top: 10px;
        margin-bottom: 30px;

        text-shadow:
            0 0 10px rgba(168,85,247,0.7),
            0 0 20px rgba(168,85,247,0.5);
    }

    /* Labels */
    label[data-baseweb="label"] {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: white !important;
    }

    /* Dropdown box */
    div[data-baseweb="select"] > div {
        font-size: 13px !important;
        color: white !important;

        background-color: #2B1E45 !important;
        border: 1px solid #A855F7 !important;
        border-radius: 10px !important;
    }

    /* Dropdown options */
    ul[role="listbox"] li {
        font-size: 13px !important;
        color: white !important;
        background-color: #1E1633 !important;
    }

    </style>

    <h1 class="centered-title">
        ⚕️ SympAI - Virtual Disease Predictor
    </h1>

    """,
    unsafe_allow_html=True
)

# Load external stylesheet (fails loudly and clearly instead of silently
# breaking the theme if the file is missing)
if os.path.exists(STYLE_PATH):
    with open(STYLE_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"style.css not found at {STYLE_PATH} — theming will look incomplete.")


# Load CSVs
@st.cache_data
def load_csvs():
    df = pd.read_csv(DATASET)
    desc = pd.read_csv(DESC)
    prec = pd.read_csv(PREC)
    sev = pd.read_csv(SEV)
    return df, desc, prec, sev

# Build Vocab
def build_vocab(df):
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    vocab = sorted(set(
        s.strip().lower().replace(" ", "_")
        for row in df[symptom_cols].fillna("").values
        for s in row if isinstance(s, str) and s.strip() != ""
    ))
    return vocab

# Vectorize
def vectorize(symptoms_selected, vocab):
    sset = set([s.strip().lower().replace(" ", "_") for s in symptoms_selected if s])
    return np.array([1 if v in sset else 0 for v in vocab], dtype=int)

# Train or Load
def train_or_load(df, vocab):
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        model = joblib.load(MODEL_PATH)
        saved_vocab = joblib.load(VOCAB_PATH)
        le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
        if saved_vocab == vocab:
            return model, le
    # Train fresh
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    X, y = [], []
    for _, row in df.iterrows():
        syms = [str(row[c]).strip() for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip()]
        X.append(vectorize(syms, vocab))
        y.append(row["Disease"])
    X = np.vstack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y_enc)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vocab, VOCAB_PATH)
    joblib.dump(le, os.path.join(DATA_DIR, "label_encoder.pkl"))
    return model, le

# Prediction Helper
def get_topk(model, x_vec, le, k=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([x_vec])[0]
        idx = np.argsort(proba)[::-1][:k]
        return [(le.inverse_transform([i])[0], float(proba[i])) for i in idx]
    pred_idx = model.predict([x_vec])[0]
    return [(le.inverse_transform([pred_idx])[0], 1.0)]

# UI
df, desc_df, prec_df, sev_df = load_csvs()
vocab = build_vocab(df)

num_symptoms = 12

# Dropdowns
symptom_options = ["-- Select a Symptom --"] + vocab
symptom_icons = ["🤒", "🤕", "🥵", "🤢", "😷", "🥴"]
user_syms = []

with st.container(border=True):
    cols = st.columns(3)
    for i in range(num_symptoms):
        with cols[i % 3]:
            icon = symptom_icons[i % len(symptom_icons)]
            selected_symptom = st.selectbox(
                f"{icon} Symptom {i+1}",
                symptom_options,
                key=f"s{i}"
            )
            if selected_symptom != "-- Select a Symptom --":
                user_syms.append(selected_symptom)

# Prediction
if st.button("Predict"):
    if not user_syms:
        st.warning("Please select at least 1 symptom.")
        st.stop()

    model, le = train_or_load(df, vocab)
    x_vec = vectorize(user_syms, vocab)

    # Always fetch top-3
    results = get_topk(model, x_vec, le, k=3)

    # Best match
    best_disease, best_prob = results[0]

    # Details
    desc = desc_df.set_index("Disease").to_dict().get("Description", {}).get(best_disease, "No description available.")
    precr = prec_df[prec_df["Disease"] == best_disease]
    prec_list = []
    if not precr.empty:
        row = precr.iloc[0]
        for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                prec_list.append(str(row[c]))

    # Main result — wrapped in a card so it doesn't look like bare text
    st.markdown(f"""
    <div class="result-card">
        <h2 class="result-title">🩺 Prediction: {best_disease}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(1.0, max(0.0, best_prob)))
    st.caption(f"Confidence: {best_prob:.1%}")

    st.markdown(f"""<div class="result-card">{desc}</div>""", unsafe_allow_html=True)

    st.write("**Precautions / Advice:**")
    if prec_list:
        for p in prec_list:
            st.markdown(f"- {p}")
    else:
        st.markdown("- Drink more water\n- Rest well\n- Consult a doctor if symptoms persist")

    #  Always show Top-3 list
    if len(results) > 1:
        st.write("---")
        st.subheader("🔮 Top 3 Predictions")
        for d, p in results:
            st.write(f"- {d}: {p:.2%}")

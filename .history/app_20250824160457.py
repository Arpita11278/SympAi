import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib, os

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    /* Label (Symptom 1, Symptom 2, ...) */
    label[data-baseweb="label"] {
        font-size: 16px !important;   /* label thoda bada */
        font-weight: 500 !important;  /* halka bold */
        color: inherit !important;    /* default Streamlit color */
    }

    /* Dropdown ke andar ka selected text */
    div[data-baseweb="select"] > div {
        font-size: 12px !important;   /* chhota font */
        color: black !important;      /* black text */
    }

    /* Dropdown khulte time list ke options */
    ul[role="listbox"] li {
        font-size: 12px !important;   /* chhota font */
        color: black !important;      /* black text */
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Virtual Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- File Paths ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_datasets")
DATASET = os.path.join(DATA_DIR, "dataset.csv")
DESC = os.path.join(DATA_DIR, "symptom_description.csv")
PREC = os.path.join(DATA_DIR, "symptom_precaution.csv")
SEV = os.path.join(DATA_DIR, "symptom_severity.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
VOCAB_PATH = os.path.join(DATA_DIR, "symptom_vocab.pkl")

# ---------- Load CSVs ----------
@st.cache_data
def load_csvs():
    df = pd.read_csv(DATASET)
    desc = pd.read_csv(DESC)
    prec = pd.read_csv(PREC)
    sev = pd.read_csv(SEV)
    return df, desc, prec, sev

# ---------- Build Vocab ----------
def build_vocab(df):
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    vocab = sorted(set(
        s.strip().lower().replace(" ", "_")
        for row in df[symptom_cols].fillna("").values
        for s in row if isinstance(s, str) and s.strip() != ""
    ))
    return vocab

# ---------- Vectorize ----------
def vectorize(symptoms_selected, vocab):
    sset = set([s.strip().lower().replace(" ", "_") for s in symptoms_selected if s])
    return np.array([1 if v in sset else 0 for v in vocab], dtype=int)

# ---------- Train or Load ----------
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

# ---------- Prediction Helper ----------
def get_topk(model, x_vec, le, k=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([x_vec])[0]
        idx = np.argsort(proba)[::-1][:k]
        return [(le.inverse_transform([i])[0], float(proba[i])) for i in idx]
    pred_idx = model.predict([x_vec])[0]
    return [(le.inverse_transform([pred_idx])[0], 1.0)]

# ---------- UI ----------
st.title("🩺 Virtual Disease Predictor ")

df, desc_df, prec_df, sev_df = load_csvs()
vocab = build_vocab(df)

num_symptoms = 12
show_probs = st.toggle("Show top-3 predictions", value=True)

# Dropdowns
symptom_options = ["-- Select a Symptom --"] + vocab
cols = st.columns(3)
user_syms = []

for i in range(num_symptoms):
    with cols[i % 3]:
        selected_symptom = st.selectbox(
            f"Symptom {i+1}",
            symptom_options,
            key=f"s{i}"
        )
        if selected_symptom != "-- Select a Symptom --":
            user_syms.append(selected_symptom)

# ---------- Prediction ----------
if st.button("Predict"):
    if not user_syms:
        st.warning("Please select at least 1 symptom.")
        st.stop()

    model, le = train_or_load(df, vocab)
    x_vec = vectorize(user_syms, vocab)   # ✅ FIX: use user_syms
    results = get_topk(model, x_vec, le, k=3 if show_probs else 1)
    best_disease, best_prob = results[0]

    desc = desc_df.set_index("Disease").to_dict().get("Description", {}).get(best_disease, "No description available.")
    precr = prec_df[prec_df["Disease"] == best_disease]
    prec_list = []
    if not precr.empty:
        row = precr.iloc[0]
        for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                prec_list.append(str(row[c]))

    st.subheader(f"Prediction: {best_disease}")
    st.progress(min(1.0, max(0.0, best_prob)))
    st.write(desc)
    st.write("**Precautions / Advice:**")
    if prec_list:
        for p in prec_list:
            st.markdown(f"- {p}")
    else:
        st.markdown("- Drink more water\n- Rest well\n- Consult a doctor if symptoms persist")

    if show_probs and len(results) > 1:
        st.write("---")
        st.write("**Top-3 predictions:**")
        for d, p in results:
            st.write(f"- {d}: {p:.2%}")

# ---------- Footer ----------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        text-align: center;
        color: gray;
        font-size: 13px;
        padding: 5px;
    }
    </style>
    <div class="footer">
        Replace sample CSVs in <code>sample_datasets/</code> with your own larger dataset.
    </div>
    """,
    unsafe_allow_html=True
)

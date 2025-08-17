import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from googletrans import Translator

# -------------------
# Load FAISS + verses
# -------------------
@st.cache_resource
def load_index():
    index = faiss.read_index("gita_index.faiss")
    with open("gita_verses.json", "r", encoding="utf-8") as f:
        verses = json.load(f)
    return index, verses

index, verses = load_index()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = InferenceClient(token="YOUR_HF_TOKEN")   # <-- replace with your HF token
translator = Translator()

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🕉️ Krishna's Guidance", page_icon="🪔", layout="centered")

st.markdown("<h1 style='text-align: center; color: gold;'>🪔 Krishna's Guidance</h1>", unsafe_allow_html=True)
st.write("Ask your question, Arjuna...")

user_input = st.text_area("Enter your problem here:", "")
language = st.selectbox("Choose response language:", ["English", "Hindi", "Telugu"])
submit = st.button("⚔️ Ask Krishna")

if submit and user_input.strip():
    # Find nearest verse
    query_emb = embedder.encode([user_input])
    D, I = index.search(np.array(query_emb).astype("float32"), k=1)
    verse_data = verses[I[0][0]]

    sanskrit_text = verse_data["sanskrit"]
    english_text = verse_data["english"]

    # LLM prompt
    prompt = f"""
    You are Lord Krishna, guiding Arjuna in the Mahabharata.

    User’s Problem:
    {user_input}

    Relevant Verse:
    Sanskrit:
    {sanskrit_text}

    Translation:
    {english_text}

    Task:
    1. Explain the verse in the context of the user's problem.
    2. Speak like Krishna — compassionate, wise, mentor-like.
    3. Make it epic, as if on the battlefield of Kurukshetra.
    """

    with st.spinner("🕉️ Krishna is speaking..."):
        response = client.text_generation(
            model="HuggingFaceH4/zephyr-7b-beta",
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.7
        )
        krishna_response = response

        # Translate if needed
        lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
        if language != "English":
            krishna_response = translator.translate(krishna_response, dest=lang_map[language]).text

    st.subheader("📜 Sanskrit Verse")
    st.markdown(f"<div style='font-size:18px; color:orange;'>{sanskrit_text}</div>", unsafe_allow_html=True)

    st.subheader("🪔 Krishna's Guidance")
    st.markdown(f"<div style='font-size:18px;'>{krishna_response}</div>", unsafe_allow_html=True)

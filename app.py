import streamlit as st
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator

# -------------------
# Load FAISS + verses
# -------------------
@st.cache_resource
def load_resources():
    index = faiss.read_index("gita_index.faiss")
    with open("gita_verses.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return index, data, model

index, data, model = load_resources()

# Hugging Face client
HF_TOKEN = os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN"))
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token is missing! Please set it in secrets.toml or env.")
else:
    client = InferenceClient(token=HF_TOKEN)

# -------------------
# Utility: search verses
# -------------------
def find_relevant_verses(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [data[i] for i in indices[0]]

# -------------------
# Divine UI Setup
# -------------------
st.set_page_config(page_title="🕉️ Krishna's Guidance", page_icon="🪔", layout="centered")

# Custom CSS for divine styling
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top, #1a1a1a, #000000);
        color: #FFD700;
        font-family: 'Georgia', serif;
    }
    h1, h2, h3, h4 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px #000000;
    }
    .krishna-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .krishna-header img {
        max-width: 250px;
        border-radius: 50%;
        box-shadow: 0px 0px 30px #FFD700;
    }
    .divine-box {
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0px;
        background: rgba(255, 215, 0, 0.05);
        box-shadow: 0px 0px 20px rgba(255, 215, 0, 0.3);
    }
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #FFD700, transparent);
        margin: 30px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Krishna header image (can be replaced with gif/png)
st.markdown(
    """
    <div class="krishna-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Krishna_with_Flute.jpg/440px-Krishna_with_Flute.jpg" alt="Lord Krishna">
        <h1>🪔 Krishna's Divine Guidance 🪔</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------
# Input Section
# -------------------
name = st.text_input("🙏 What is your name?")
query = st.text_area("⚔️ Share your problem (as Arjuna did in Kurukshetra):")

language = st.selectbox("🌐 Choose response language:", ["English", "Hindi", "Telugu"])
submit = st.button("✨ Ask Krishna ✨")

# -------------------
# Main Logic
# -------------------
if submit and name.strip() and query.strip():
    results = find_relevant_verses(query)

    prompt = f"""
You are Lord Krishna, guiding the {name} as a mentor and teacher in the Bhagavad Gita. 

The {name} has come to you with a personal problem:

{name}’s Problem:
{query}

Your task:
1. Print the Sanskrit sloka first.
2. Use the teachings of the Bhagavad Gita to respond.
3. Provide Sanskrit, translation, and a personalised explanation tied to {name}'s problem.
4. Use this knowledge base:
{results}

Tone:
- Compassionate, wise, and mentor-like.
- Make it divine, epic, as if on Kurukshetra.
"""

    with st.spinner("🕉️ Krishna is speaking..."):
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": "You are Lord Krishna, guiding with compassion and wisdom."},
                {"role": "user", "content": prompt}
            ]
        )
        krishna_response = response.choices[0].message["content"]

        # Translate if needed
        lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
        if language != "English":
            krishna_response = GoogleTranslator(source="auto", target=lang_map[language]).translate(krishna_response)

    # -------------------
    # Display Answer
    # -------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📜 Krishna's Divine Words")
    st.markdown(f"<div class='divine-box'>{krishna_response}</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("🔎 Relevant Verses")
    for verse in results:
        st.markdown(
            f"""
            <div class="divine-box">
            <b>🕉️ Chapter {verse['chapter']}, Verse {verse['verse']}</b><br>
            <i>📜 Sanskrit:</i> {verse['sanskrit']}<br>
            <i>🔹 English:</i> {verse['english']}
            </div>
            """,
            unsafe_allow_html=True
        )

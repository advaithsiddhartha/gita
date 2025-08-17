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
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # light & stable
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
# Streamlit UI
# -------------------
st.set_page_config(page_title="🕉️ Krishna's Guidance", page_icon="🪔", layout="centered")

st.markdown("<h1 style='text-align: center; color: gold;'>🪔 Krishna's Guidance</h1>", unsafe_allow_html=True)

name = st.text_input("What is your name?")
query = st.text_area("Describe your problem (like Arjuna on the battlefield):")

language = st.selectbox("Choose response language:", ["English", "Hindi", "Telugu"])
submit = st.button("⚔️ Ask Krishna")

if submit and name.strip() and query.strip():
    results = find_relevant_verses(query)

    # Build prompt
    prompt = f"""
You are Lord Krishna, guiding the {name} as a mentor and teacher in the Bhagavad Gita. 

The {name} has come to you with a personal problem:

{name}’s Problem:
{query}

Your task:
1. Read the problem carefully.
2. Print the sloka in sanskrit first
2. Use the teachings of the Bhagavad Gita, especially relevant Slokas, to respond. 
3. When quoting a sloka, provide:
   - The Sanskrit verse (or transliteration if needed).
   - A simple English translation.
   - A personalised explanation, tying it directly to the user's problem.
4. Use the following knowledge base to support your answer:
{results}

Tone:
- Compassionate, wise, and mentor-like (as Krishna speaking directly to Arjuna).
- Personalise the response so the user feels Krishna is guiding them personally.
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
    # Display
    # -------------------
    st.subheader("📜 Krishna's Guidance")
    st.markdown(f"<div style='font-size:18px;'>{krishna_response}</div>", unsafe_allow_html=True)

    st.subheader("🔎 Relevant Verses")
    for verse in results:
        st.markdown(f"**🕉️ Chapter {verse['chapter']}, Verse {verse['verse']}**")
        st.markdown(f"📜 Sanskrit: {verse['sanskrit']}")
        st.markdown(f"🔹 English: {verse['english']}")

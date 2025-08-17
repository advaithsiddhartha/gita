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
    # ✅ Fixed model identifier
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, data, model

index, data, model = load_resources()

# -------------------
# Hugging Face client
# -------------------
HF_TOKEN = os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN", None))
client = None
if HF_TOKEN:
    client = InferenceClient(token=HF_TOKEN)
else:
    st.warning("⚠️ Hugging Face token is missing! LLM responses will not work, but search will.")

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

    krishna_response = ""
    if client:
        # Build prompt
        prompt = f"""
You are Lord Krishna, guiding {name} as a mentor and teacher in the Bhagavad Gita. 

{name}'s Problem:
{query}

Your task:
1. Read the problem carefully.
2. Quote relevant slokas (Sanskrit first).
3. Provide a simple English translation.
4. Give a personalised explanation, tying it directly to {name}'s problem.
5. Use the following knowledge base to support your answer:
{results}

Tone: Compassionate, wise, and mentor-like (as Krishna speaking directly to Arjuna).
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
    else:
        krishna_response = "🙏 Please add a valid Hugging Face token to get Krishna's divine guidance."

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

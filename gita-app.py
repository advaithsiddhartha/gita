import json
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from googletrans import Translator
from streamlit_lottie import st_lottie
import requests

# -------------------
# Load FAISS index + verses
# -------------------
index = faiss.read_index("gita_index.faiss")

with open("gita_verses.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face LLM client
client = InferenceClient(token="YOUR_HF_TOKEN")

# Translator
translator = Translator()

# -------------------
# Custom Epic CSS
# -------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpaperaccess.com/full/2087385.jpg");
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1 {
    font-family: 'Georgia', serif;
    color: gold !important;
    text-shadow: 2px 2px 8px black;
}
h2, h3 {
    color: #FFD700 !important;
    text-shadow: 1px 1px 6px black;
}
div.stButton > button {
    background-color: #FFD700;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    font-weight: bold;
    box-shadow: 0px 4px 8px black;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------
# Lottie Animation
# -------------------
def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

krishna_anim = load_lottie("https://assets9.lottiefiles.com/packages/lf20_2glqweqs.json")

# -------------------
# Streamlit UI
# -------------------
st_lottie(krishna_anim, height=300, key="krishna")
st.title("🕉️ महाभारत Epic — Krishna's Guidance 🪔")

user_input = st.text_area("🙏 Arjuna, tell me your problem:")
chosen_lang = st.selectbox("🌐 Choose your language:", ["English", "Hindi", "Telugu"])

if st.button("⚔️ Ask Krishna"):
    if user_input.strip():
        # Retrieve verse
        query_emb = embedder.encode([user_input])
        D, I = index.search(np.array(query_emb).astype("float32"), k=1)
        verse_data = verses[I[0][0]]

        sanskrit_text = verse_data["sanskrit"]
        english_text = verse_data["english"]

        # Prompt
        prompt = f"""
        You are Lord Krishna on the battlefield of Kurukshetra, guiding Arjuna.

        User’s Problem:
        {user_input}

        Relevant Verse:
        Sanskrit:
        {sanskrit_text}

        Translation:
        {english_text}

        Task:
        1. Explain the verse in context of the user's problem.
        2. Speak like Krishna — compassionate, wise, mentor-like.
        3. Make it epic, as if it is the Mahabharata dialogue.
        """

        # LLM response
        response = client.text_generation(
            model="HuggingFaceH4/zephyr-7b-beta",
            prompt=prompt,
            max_new_tokens=450,
            temperature=0.7
        )
        krishna_response = response

        # Translate
        lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
        if chosen_lang != "English":
            translated = translator.translate(krishna_response, dest=lang_map[chosen_lang]).text
        else:
            translated = krishna_response

        # Display
        st.markdown("## 📜 Sanskrit Verse")
        st.markdown(f"<div style='font-size:20px; color:white; background:rgba(0,0,0,0.6); padding:15px; border-radius:10px;'>{sanskrit_text}</div>", unsafe_allow_html=True)

        st.markdown("## 🪔 Krishna's Guidance")
        st.markdown(f"<div style='font-size:18px; color:#FFD700; background:rgba(0,0,0,0.7); padding:15px; border-radius:10px;'>{translated}</div>", unsafe_allow_html=True)

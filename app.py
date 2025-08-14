import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai
import chromadb
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import asyncio
from dotenv import load_dotenv
load_dotenv()
# Fix for 'no current event loop' in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# ==============================
# CONFIGURATION (ENV VARS)
# ==============================
CURSEFORGE_API_KEY = os.getenv("CURSEFORGE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not CURSEFORGE_API_KEY or not GOOGLE_API_KEY:
    st.error("Please set CURSEFORGE_API_KEY and GOOGLE_API_KEY as environment variables.")
    st.stop()

headers = {
    "Accept": "application/json",
    "x-api-key": CURSEFORGE_API_KEY
}

mods = [
    "journeymap", "jei", "sodium", "AdditionalStructures", "adventurez",
    "deeperdarker-fabric", "DistantHorizons-fabric", "better-end1",
    "paradise-lost", "twilightforest-fabric", "The_Graveyard3.1(FABRIC)_for",
    "immersive_weathering", "Pehkui"
]

# ==============================
# FETCH MOD DATA (Run Once)
# ==============================
@st.cache_data
def fetch_mod_data():
    mod_data_list = []
    for mod_entry in mods:
        clean_name = mod_entry.split("-")[0].replace("_", " ").split("(")[0]
        search_url = f"https://api.curseforge.com/v1/mods/search?gameId=432&searchFilter={clean_name}"
        search_res = requests.get(search_url, headers=headers)

        if search_res.status_code == 200 and search_res.json().get("data"):
            mod_info = search_res.json()["data"][0]
            mod_id = mod_info["id"]
            mod_name = mod_info["name"]
            mod_slug = mod_info["slug"]

            desc_url = f"https://api.curseforge.com/v1/mods/{mod_id}/description"
            desc_res = requests.get(desc_url, headers=headers)

            if desc_res.status_code == 200:
                raw_html = desc_res.json().get("data", "No description found")
                desc = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n", strip=True)
            else:
                desc = "Description fetch failed."

            mod_data_list.append({
                "Your Entry": mod_entry,
                "Mod Name": mod_name,
                "CurseForge URL": f"https://www.curseforge.com/minecraft/mc-mods/{mod_slug}",
                "Description": desc
            })
        else:
            mod_data_list.append({
                "Your Entry": mod_entry,
                "Mod Name": "Not Found",
                "CurseForge URL": "N/A",
                "Description": "No description available."
            })

        time.sleep(0.5)

    df = pd.DataFrame(mod_data_list)
    df.to_csv("Minecraft_mod.csv", index=False, encoding="utf-8-sig")
    return df

df = fetch_mod_data()

# ==============================
# CREATE EMBEDDINGS & STORE
# ==============================
@st.cache_resource
def setup_chroma():
    genai.configure(api_key=GOOGLE_API_KEY)
    texts = df["Description"].tolist()
    embeddings = [
        genai.embed_content(model="models/embedding-001", content=text)["embedding"]
        for text in texts
    ]
    client = chromadb.PersistentClient(path="my_chroma_db")
    collection = client.get_or_create_collection(name="minecraft_mods")
    ids = [f"doc_{i}" for i in range(len(df))]
    metadatas = [{"mod_name": row["Mod Name"]} for _, row in df.iterrows()]
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chroma_db = Chroma(persist_directory="my_chroma_db", embedding_function=embedding_function)
    return chroma_db

chroma_db = setup_chroma()
retriever = chroma_db.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# ==============================
# RAG FUNCTION
# ==============================
def get_rag_answer(query: str) -> str:
    retrieved_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are an expert Minecraft mod analyst. You will be provided with textual descriptions of multiple Minecraft mods.

INSTRUCTIONS:
1. Search the provided context carefully and find the most relevant Minecraft mod(s) that best answer the question.
2. Extract detailed information about these mods, focusing on accuracy and relevance.
3. For each mod you find relevant, provide a structured JSON object with:
   - "Name of the mod"
   - "Version"
   - "Description"
   - "Mod Loader"
   - "Available Mod Loaders"
   - "Additional Notes"
4. Only reply with JSON. If no relevant mod is found, return [].

Context:
{context_text}

Question:
{query}

Answer with JSON only:
"""
    return llm.predict(prompt)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Minecraft Mods Q&A", page_icon="🎮")
st.title("🎮 Minecraft Mods Q&A Dashboard")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask about any Minecraft mod...")

if user_question:
    with st.spinner("Fetching answer..."):
        answer = get_rag_answer(user_question)
    st.session_state.chat_history.append({"question": user_question, "answer": answer})

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

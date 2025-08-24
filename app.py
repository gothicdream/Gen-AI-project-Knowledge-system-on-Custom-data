# app.py — Streamlit CSV → Gemini RAG (ChromaDB)

import os
import io
import uuid
import pandas as pd
import streamlit as st
import google.generativeai as genai
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Minecraft Mods Q&A", layout="wide")
st.title("🎮 Minecraft Mods Q&A with Gemini + CSV Upload")

# Google API Key input
api_key = st.text_input("🔑 Enter your Google API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

# ChromaDB options
persist_path = st.text_input("ChromaDB path", value="my_chroma_db")
collection_name = st.text_input("Collection name", value="minecraft_mods")
reset_collection = st.checkbox("Reset collection before indexing", value=False)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your Minecraft Mods CSV", type=["csv"]) 

# Helper: robust CSV loader
def load_csv_resilient(file) -> pd.DataFrame:
    raw = file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    seps = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in seps:
            try:
                buf = io.BytesIO(raw)
                df = pd.read_csv(buf, encoding=enc, sep=sep)
                if not df.empty and len(df.columns) >= 1:
                    return df
            except Exception:
                continue
    for enc in encodings:
        for sep in seps:
            try:
                buf = io.BytesIO(raw)
                df = pd.read_csv(buf, encoding=enc, sep=sep, header=None)
                if not df.empty:
                    df.columns = [f"col_{i}" for i in range(len(df.columns))]
                    return df
            except Exception:
                continue
    raise ValueError("❌ Could not parse CSV. Please save it as UTF-8 and try again.")

# -------------------------------
# Main App
# -------------------------------
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False

if uploaded_file and api_key:
    try:
        df = load_csv_resilient(uploaded_file)
    except Exception as e:
        st.error(f"❌ CSV parse error: {e}")
        st.stop()

    st.write("### 👀 Preview of Uploaded CSV")
    st.dataframe(df.head())

    # Pick text column for embeddings
    default_col = "Description" if "Description" in df.columns else df.columns[0]
    text_column = st.selectbox("Select column to embed", options=list(df.columns), index=list(df.columns).index(default_col))

    # Optional ID column
    id_default = "Mod Name" if "Mod Name" in df.columns else None
    id_column = st.selectbox("Optional: ID/Name column (metadata)", options=["<none>"] + list(df.columns), index=(0 if id_default is None else list(["<none>"] + list(df.columns)).index(id_default)))

    if st.button("🚀 Create / Update Embeddings"):
        client = chromadb.PersistentClient(path=persist_path)

        # Reset collection if chosen
        if reset_collection:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(name=collection_name)

        with st.spinner("Generating embeddings and indexing into ChromaDB... ⏳"):
            texts = df[text_column].dropna().astype(str).tolist()
            valid_idx = df[text_column].dropna().index.tolist()

            # Build embeddings
            embeddings = []
            for t in texts:
                try:
                    res = genai.embed_content(model="models/embedding-001", content=t)
                    embeddings.append(res["embedding"])
                except Exception as e:
                    if embeddings:
                        embeddings.append([0.0] * len(embeddings[0]))
                    else:
                        raise e

            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(texts))]

            # Metadata
            if id_column and id_column != "<none>":
                metas = [{"name": str(df.loc[idx, id_column])} for idx in valid_idx]
            else:
                metas = [{"row_index": int(idx)} for idx in valid_idx]

            # Fetch existing IDs to avoid duplicates
            try:
                existing_ids = set(collection.get()["ids"])
            except Exception:
                existing_ids = set()

            new_ids, new_texts, new_metas, new_embeddings = [], [], [], []
            for i, id_ in enumerate(ids):
                if id_ not in existing_ids:
                    new_ids.append(id_)
                    new_texts.append(texts[i])
                    new_metas.append(metas[i])
                    new_embeddings.append(embeddings[i])

            if new_ids:
                collection.add(ids=new_ids, documents=new_texts, metadatas=new_metas, embeddings=new_embeddings)

        st.session_state.collection_ready = True
        st.success("✅ Embeddings stored in ChromaDB successfully!")

# -------------------------------
# Q&A Section
# -------------------------------
if st.session_state.collection_ready and api_key:
    st.divider()
    st.subheader("💬 Ask Questions About Your Mods")

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(name=collection_name)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    question = st.text_input("Type your question here:")
    top_k = st.slider("Number of docs to retrieve", 1, 10, 3)

    if question:
        with st.spinner("Thinking... 🤔"):
            q_emb = genai.embed_content(model="models/embedding-001", content=question)["embedding"]

            results = collection.query(query_embeddings=[q_emb], n_results=top_k)
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            context_chunks = []
            for i, doc in enumerate(docs):
                meta = metadatas[i] if i < len(metadatas) else {}
                name = meta.get("name") or f"Row {meta.get('row_index', i)}"
                context_chunks.append(f"Source {i+1} - {name}:\n{doc}")
            context_text = "\n\n".join(context_chunks) if context_chunks else ""

            prompt = f"""
You are an expert Minecraft mod analyst. Use ONLY the provided context to answer the question.  
If the answer cannot be determined from the context, simply reply with: "I don’t know based on the given information."

Context:
{context_text}

Question:
{question}

Answer clearly and directly in plain text.  
"""
            try:
                answer = llm.predict(prompt)
                st.write(answer)
            except Exception as e:
                st.error(f"LLM error: {e}")

            with st.expander("🔎 Retrieved context (sources)"):
                for i, doc in enumerate(docs):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    name = meta.get("name") or f"Row {meta.get('row_index', i)}"
                    st.markdown(f"**Source {i+1}: {name}**")

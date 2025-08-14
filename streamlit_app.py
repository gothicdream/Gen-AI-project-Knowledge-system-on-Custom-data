# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import streamlit as st

dir_path = Path(__file__).parent


# Note that this needs to be in a method so we can have an e2e playwright test.
def run() -> None:
    page = st.navigation(
        {
            "Pages": [
                st.Page(
                    dir_path / "hello.py", title="Hello", icon=":material/waving_hand:"
                ),
                st.Page(
                    dir_path / "dataframe_demo.py",
                    title="DataFrame demo",
                    icon=":material/table:",
                ),
                st.Page(
                    dir_path / "plotting_demo.py",
                    title="Plotting demo",
                    icon=":material/show_chart:",
                ),
                st.Page(
                    dir_path / "mapping_demo.py",
                    title="Mapping demo",
                    icon=":material/public:",
                ),
                st.Page(
                    dir_path / "animation_demo.py",
                    title="Animation demo",
                    icon=":material/animation:",
                ),
            ]
        }
    )
    page.run()


if __name__ == "__main__":
    run()
"""
Streamlit Q&A Dashboard for Minecraft Mods (RAG + Gemini)
- Uses your existing RAG backend (Chroma + Gemini) under the hood
- Simple "API via JSON file" for storing & retrieving chat history (no DB needed)
- Clean chat UI powered by st.chat_message

How to run:
1) pip install -U streamlit langchain-chroma langchain-google-genai chromadb google-generativeai
2) Ensure your Chroma DB exists at PERSIST_DIRECTORY (default: ./my_chroma_db)
3) Set GOOGLE_API_KEY in environment or Streamlit secrets
4) streamlit run streamlit_app.py
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# ==============================
# BASIC CONFIG
# ==============================
st.set_page_config(
    page_title="Minecraft Mod Q&A (RAG + Gemini)",
    page_icon="🎮",
    layout="wide",
)

TITLE = "🎮 Minecraft Mod Q&A (RAG + Gemini)"
HISTORY_JSON = "history.json"  # <- our tiny "API" storage
DEFAULT_PERSIST_DIR = "my_chroma_db"

# ==============================
# UTIL: JSON history as a tiny API
# ==============================

def _load_history(path: str = HISTORY_JSON) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(items: List[Dict[str, Any]], path: str = HISTORY_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _append_history(entry: Dict[str, Any], path: str = HISTORY_JSON) -> None:
    items = _load_history(path)
    items.append(entry)
    _save_history(items, path)


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("Settings")

    persist_dir = st.text_input(
        "ChromaDB directory",
        value=DEFAULT_PERSIST_DIR,
        help="Folder where your prebuilt Chroma DB is stored.",
    )

    k_value = st.slider("Top-k documents", 1, 8, 2, help="How many chunks to retrieve.")
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("API Keys (stored only in session)")
    google_api_key = st.text_input("GOOGLE_API_KEY", type="password")

    st.markdown("---")
    save_to_history = st.checkbox("Save Q&A to history.json", value=True)
    if st.button("Clear history.json"):
        _save_history([])
        st.toast("history.json cleared.")

# ==============================
# RAG BACKEND (wrapping your code)
# ==============================
@st.cache_resource(show_spinner=False)
def _build_components(persist_directory: str, api_key: str, temperature: float):
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Provide it in the sidebar or via env/secrets."
        )

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=temperature,
        google_api_key=api_key,
    )

    return chroma_db, llm


def _force_json(s: str) -> Any:
    """Best-effort to coerce LLM output into JSON (list or object)."""
    # Try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract between first '[' and last ']'
    if "[" in s and "]" in s:
        try:
            return json.loads(s[s.index("[") : s.rindex("]") + 1])
        except Exception:
            pass
    # Try to extract single object
    if "{" in s and "}" in s:
        try:
            return json.loads(s[s.index("{") : s.rindex("}") + 1])
        except Exception:
            pass
    return []


def get_rag_answer(query: str, k: int, persist_directory: str, api_key: str, temperature: float):
    chroma_db, llm = _build_components(persist_directory, api_key, temperature)

    # Build retriever with current k
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    context_text = "\n\n".join(doc.page_content for doc in docs)

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

    # Some LangChain chat models use .invoke, some support .predict
    try:
        raw = llm.predict(prompt)  # type: ignore
    except Exception:
        resp = llm.invoke(prompt)  # type: ignore
        raw = getattr(resp, "content", resp)

    parsed = _force_json(raw)

    # Collect lightweight source info
    sources = [
        {
            "metadata": getattr(d, "metadata", {}),
            "preview": (d.page_content[:200] + "…") if d.page_content else "",
        }
        for d in docs
    ]

    return {
        "answer": parsed,
        "raw": raw,
        "sources": sources,
    }


# ==============================
# HEADER
# ==============================
st.title(TITLE)
left, right = st.columns([3, 1])
with left:
    st.write(
        "Ask about Minecraft mods you have indexed into your local Chroma database."
    )
with right:
    hist = _load_history()
    st.metric("Answers saved", len(hist))

# ==============================
# CHAT UI
# ==============================
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{role, content}]

# Show previous messages
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            st.json(msg["content"])  # pretty JSON display
        else:
            st.write(msg["content"])

query = st.chat_input("Type your question about a mod…")

if query:
    # Show the user message
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Validate prerequisites
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please provide GOOGLE_API_KEY in the sidebar or via env/secrets.")
    elif not os.path.isdir(persist_dir):
        with st.chat_message("assistant"):
            st.error(
                f"ChromaDB directory '{persist_dir}' not found. Point to your existing DB in the sidebar."
            )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching mods and generating answer…"):
                result = get_rag_answer(
                    query=query,
                    k=k_value,
                    persist_directory=persist_dir,
                    api_key=api_key,
                    temperature=temperature,
                )

                # Show JSON answer
                st.json(result["answer"])  # pretty view

                # Expanders for raw + sources
                with st.expander("Raw model output"):
                    st.code(result["raw"], language="json")

                with st.expander("Sources (top-k chunks)"):
                    if not result["sources"]:
                        st.write("No sources returned.")
                    else:
                        for i, s in enumerate(result["sources"], start=1):
                            st.markdown(f"**Chunk {i}**")
                            md = s.get("metadata", {})
                            if md:
                                st.write(md)
                            st.write(s.get("preview", ""))
                            st.markdown("---")

                # Append to chat memory
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": result["answer"],
                })

                # Save to history JSON if enabled
                if save_to_history:
                    _append_history(
                        {
                            "id": str(uuid.uuid4()),
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "question": query,
                            "answer": result["answer"],
                            "raw": result["raw"],
                            "k": k_value,
                            "persist_directory": persist_dir,
                        }
                    )

# ==============================
# HISTORY PANEL
# ==============================
st.markdown("---")
st.subheader("📜 History (JSON API store)")

hist_cols = st.columns([1, 1])
with hist_cols[0]:
    if st.button("Reload history.json"):
        st.rerun()
with hist_cols[1]:
    # Offer a download of the entire history file
    items = _load_history()
    st.download_button(
        label="Download history.json",
        data=json.dumps(items, ensure_ascii=False, indent=2),
        file_name="history.json",
        mime="application/json",
    )

# Show a compact table-like list of previous Q&A
history = _load_history()
if history:
    for item in reversed(history[-50:]):  # show last 50
        with st.expander(f"Q: {item.get('question', '')[:80]}…  |  {item.get('ts', '')}"):
            st.markdown("**Answer (parsed)**")
            st.json(item.get("answer", {}))
            st.markdown("**Raw**")
            st.code(item.get("raw", ""), language="json")
else:
    st.caption("No history yet. Ask something to create entries.")

# ==============================
# FOOTER TIP
# ==============================
st.caption(
    "Tip: This app keeps things simple—answers and metadata are saved to a local JSON file, \n"
    "which acts as a tiny API for your project or resume demo. You can later replace it with FastAPI or a DB without changing the UI."
)

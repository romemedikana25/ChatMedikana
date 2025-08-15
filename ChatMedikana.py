# User Interface Imports #
import streamlit as st

# ---- Google Drive Sync (FAISS) ----
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import io
from datetime import datetime, timezone
from drive_utils import get_drive_service

# AI Imports #
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

# Data Handling Imports #
import pandas as pd

# functions from local files #
from utils import is_table_or_numerical_query, normalize_text, extract_country_from_query
from global_vars import AMERICAS

# File Handling Imports #
from pathlib import Path
import os
import time
from PIL import Image
import tempfile
import json

# Custom Retriever with Country Filtering
from custom_retriever_class import CountryFilteredRetriever

BASE_DIR = Path(__file__).resolve().parent

# Icon lives in the repo root next to ChatMedikana.py
ICON_PATH = BASE_DIR / "Medikana v2-1.png"

# Put FAISS cache in the OS temp dir (works locally & on Streamlit Cloud)
INDEX_PATH = Path(tempfile.gettempdir()) / "chatmedikana_faiss_index"

# Streamlit Session State
if "state" not in st.session_state:
    st.session_state.state = 'Password'
if "messages" not in st.session_state:
    st.session_state.messages = [] # [{"role": "user"/"assistant", "content": str, "sources": [...] }]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # list of (user, assistant) tuples for CRC memory

# ---- Password Protection (local only) ----
if st.session_state.state == 'Password':
    st.title("🔒 Medikana Knowledge Base")
    st.caption("Please enter the password to access the knowledge base.")
    auth_pass = st.text_input("🔒 Enter password to access", type="password")
    if auth_pass != st.secrets['PASSWORD']:
        st.stop()
    else:
        st.session_state.state = 'Authenticated'
        st.rerun()

# ---- CONFIG ----
from config import get_openai_key, mask
OPENAI_API_KEY = get_openai_key()
st.caption(f"🔑 OpenAI key loaded: {mask(OPENAI_API_KEY)}")

# REQUIRED: put these in Streamlit secrets or hardcode (IDs are best; names are ambiguous)
SHARED_DRIVE_ID   = st.secrets["SHARED_DRIVE_ID"]        # e.g. "0AAbcDEF...PVA"
INDEX_FOLDER_ID   = st.secrets["INDEX_FOLDER_ID"]        # ID of the *faiss_index* folder in Drive
KB_FOLDER_ID      = st.secrets["KB_FOLDER_ID"]                   # name of kb folder
TABLE_CATALOG_ID  = st.secrets["TABLE_CATALOG_ID"]               # get the table catlog folder id
JSON_FILE_ID      = st.secrets["JSON_FILE_ID"]

# ---- Define English QA/Condense Prompt ----
def english_qa_prompt():
    """ Load the English QA prompt template. """
    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=(
            "You are a precise regulatory/market analyst for Medikana. Answer using ONLY the provided context.\n"
            "Rules:\n"
            "1) Jurisdiction: If the question mentions a country/region, restrict your answer to that jurisdiction's documents ONLY.\n"
            "   - If any snippet appears to be from a different jurisdiction, ignore it.\n"
            "2) Source sufficiency: If the context does not contain enough evidence to answer confidently,\n"
            "   ask ONE clarifying question and stop. Do not guess or import outside knowledge.\n"
            "3) Document types: If the question is regulatory, prefer official dispositions, annexes, articles, and definitions.\n"
            "   If the question is market/insights, summarize only what is in the provided context.\n"
            "4) Citations: If you state specific rules, deadlines, or definitions, anchor them to the provided context (file/annex names\n"
            "   if present in the snippets). Do NOT fabricate references.\n"
            "5) Annex/article Guardrail:\n"
            "- If the user cites an annex/article that does NOT match what the Context shows, "
            "explicitly say so and point to the correct annex/article in the Context before answering.\n\n"
            "6) Timeline guardrail:\n"
            "- Never invent timelines, days, or deadlines. If no specific timeline is in the Context, say: "
            "\"No specific timeline is stated in the provided documents.\"\n\n"
            "7) Language: Respond in English only.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "If needed, the chat history may resolve referents (e.g., which country), but do not pull new facts from it:\n"
            "{chat_history}\n"
        )
    )

def condense_prompt():
    """ Load the Condense prompt template for rewriting user questions when they are following up on previous queries. """
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Rewrite the user question as a standalone query in English.\n"
            "Use the chat history ONLY to recover missing referents (e.g., country/jurisdiction, product names, annex/article numbers,\n"
            "dates, company names). Do NOT add new facts or change the user's intent.\n"
            "If the country is ambiguous, include the ambiguity explicitly in the standalone query (e.g., 'in Argentina?').\n\n"
            "Chat history:\n{chat_history}\n\n"
            "User question: {question}\n\n"
            "Standalone query:"
        ),
    )

def clarifier_prompt():
    return PromptTemplate(
        input_variables=["question"],
        template=(
            "You do not have enough context to answer confidently.\n"
            "Ask ONE short clarifying question that will help you answer:\n\n"
            "User question: {question}\n"
            "Clarifying question:"
        )
    )

# ---- Load or Create FAISS DB ----
def _list_index_files(service):
    """List files inside the Drive index folder and return mapping of name -> file metadata."""
    resp = service.files().list(
        q=f"'{INDEX_FOLDER_ID}' in parents and trashed=false",
        corpora="drive",
        driveId=SHARED_DRIVE_ID,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id,name,mimeType,modifiedTime,size)"
    ).execute()
    files = resp.get("files", [])
    return {f["name"]: f for f in files}

def _parse_z(dt_str):
    # '2025-08-10T12:34:56.000Z' -> aware datetime
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

@st.cache_resource(show_spinner="Loading vector DB...")
def load_vector_db(openai_key: str):
    """
    Sync index.faiss/index.pkl from Drive INDEX_FOLDER_ID → local INDEX_PATH, then load FAISS.
    """
    # 1) Find the two index files in Drive
    svc = get_drive_service()
    remote = _list_index_files(svc)
    required = {"index.faiss", "index.pkl"}
    if not required.issubset(remote.keys()):
        st.error("Drive index folder is missing index.faiss and/or index.pkl. Rebuild the index first.")
        return None

    # make local storage for files in streamlit
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    local_faiss = INDEX_PATH / "index.faiss"
    local_pkl   = INDEX_PATH / "index.pkl"

    def _download_to(path: Path, file_id: str):
        req = svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        path.write_bytes(buf.getvalue())

    # Always download from Drive
    _download_to(local_faiss, remote["index.faiss"]["id"])
    _download_to(local_pkl,   remote["index.pkl"]["id"])
    st.info("Downloaded FAISS index from Drive.")

    embeddings = OpenAIEmbeddings(
        api_key=openai_key,
        model="text-embedding-3-small",  # ensure matches what you built with
    )

    # Fail fast: prove the key actually works *before* loading FAISS.
    try:
        _ = embeddings.embed_query("ping")
    except Exception as e:
        st.error("❌ OpenAI embeddings auth failed. Check `OPENAI_API_KEY` in Streamlit secrets (no quotes) and project access.")
        raise

    try:
        # makes instance of FAISS DB for similarity search
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# ---- Table Catalog -----
def _find_tables_json(service) -> str | None:
    """Find the file id for 'tables_catalog.json' inside the Table Catalog folder."""
    if not (SHARED_DRIVE_ID and TABLE_CATALOG_FOLDER_ID):
        return None
    resp = service.files().list(
        q=f"'{TABLE_CATALOG_FOLDER_ID}' in parents and trashed=false and name='tables_catalog.json'",
        corpora="drive",
        driveId=SHARED_DRIVE_ID,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id,name)",
        pageSize=1,
    ).execute()
    items = resp.get("files", [])
    return items[0]["id"] if items else None

def _download_file_bytes(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

@st.cache_data(show_spinner="Loading table catalog…")
def load_table_catalog() -> list[dict]:
    """
    Load the prebuilt tables_catalog.json from Drive.
    Prefers TABLES_JSON_FILE_ID; otherwise, discovers it in TABLE_CATALOG_FOLDER_ID.
    """
    svc = get_drive_service()

    # prefer direct file id if present; else discover by folder
    tables_file_id = JSON_FILE_ID or _find_tables_json(svc)
    if not tables_file_id:
        st.warning("Could not locate 'tables_catalog.json'. Provide JSON_FILE_ID or TABLE_CATALOG_FOLDER_ID in secrets.")
        return []

    data = json.loads(_download_file_bytes(svc, tables_file_id).decode("utf-8"))
    return data.get("tables", [])

# ---- Table Selection ----
def choose_table_for_query(query: str) -> dict | None:
    """
    Vector-search over table descriptions/previews and return the best table entry (dict).
    Expects catalog schema:
      {"file_id","file_name","mimeType","modifiedTime","path","description","preview"}
    """
    if not TAB_DATA:
        return None

    docs = []
    for i, t in enumerate(TAB_DATA):
        desc = t.get("description", "")
        prev = t.get("preview", "")
        docs.append(Document(page_content=f"{desc}\nPreview:\n{prev}", metadata={"i": i}))

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vector_db = FAISS.from_documents(docs, embeddings)

    hits = vector_db.similarity_search(query, k=1)
    if not hits:
        return None

    idx = hits[0].metadata.get("i")
    if idx is None or idx >= len(TAB_DATA):
        return None

    return TAB_DATA[idx]  # return the catalog entry itself

# If you already have these constants, reuse them:
MT_GSHEET = "application/vnd.google-apps.spreadsheet"

@st.cache_data(show_spinner=False)
def export_gsheet_csv(file_id: str) -> bytes:
    """Export a Google Sheet to CSV bytes."""
    svc = get_drive_service()
    req = svc.files().export_media(fileId=file_id, mimeType="text/csv")
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def download_bytes(file_id: str) -> bytes:
    """Download any Drive file's raw bytes using helper from before."""
    # You already have _download_file_bytes(service, file_id); reuse it:
    svc = get_drive_service()
    return _download_file_bytes(svc, file_id)

def table_entry_to_dataframe(table: dict) -> pd.DataFrame:
    file_id   = table["file_id"]
    file_name = table.get("file_name", "")
    mime_type = table.get("mimeType", "")
    ext = Path(file_name).suffix.lower()

    if mime_type == MT_GSHEET:
        raw = export_gsheet_csv(file_id)  # CSV bytes from Drive
        return pd.read_csv(io.BytesIO(raw), engine="python")
    else:
        raw = download_bytes(file_id)     # raw file bytes from Drive
        if ext == ".csv":
            return pd.read_csv(io.BytesIO(raw), engine="python", encoding="utf-8-sig")
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        else:
            raise ValueError(f"Unsupported table type: {mime_type} / {ext}")
            
# ---- Handle Table Query ----
def handle_tabular_query(query: str, table_entry: dict):
    """
    Handle a table question by pulling the file from Drive and answering using only that table.
    """
    try:
        df = table_entry_to_dataframe(table_entry) # try to make df from helper
    except Exception as e:
        return f"Error reading table: {e}", None

    if df.empty:
        return "The selected table appears to be empty.", [{
            "file_name": table_entry.get("file_name"),
            "file_path": table_entry.get("path"),
            "last_modified": table_entry.get("modifiedTime"),
        }]

    markdown_table = df.to_markdown(index=False)

    prompt = PromptTemplate(
        input_variables=["query", "table"],
        template="""
        You are an expert at answering questions from tabular data.
        Here is the table:
        {table}

        Answer the following question using only the data from the table.

        Question: {query}

        - If the answer is a count, provide the exact number.
        - If the answer is a list, list the items clearly.
        - If numeric calculations are needed, compute them directly.
        - If unsure, say "I cannot answer from this table."
        Respond in English only.
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"query": query, "table": markdown_table})

    # Return metadata dict (your UI already supports dict sources)
    src = [{
        "file_name": table_entry.get("file_name"),
        "file_path": table_entry.get("path"),           # the Drive UI path-like, for display only
        "last_modified": table_entry.get("modifiedTime"),
    }]
    return response["text"].strip(), src

# ---- Build Conversational Chain (with memory) ----
def build_crc(db, country):
    """ Build a Conversational Retrieval Chain with memory for handling user queries. """
    search_kwargs = {
        "k": 10,  # Number of documents to retrieve
        "fetch_k": 40,  # Number of documents to fetch before filtering
        "lambda_mult": 0.8,  # MMR lambda multiplier
    }

    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    custom_retriever = CountryFilteredRetriever(base_retriever=base_retriever, extract_country_fn=extract_country_from_query)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=custom_retriever,
        condense_question_prompt=condense_prompt(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": english_qa_prompt()},
        return_generated_question=True,
    )

# ---- Streamlit UI ----

if st.session_state.state == 'Authenticated':

    st.title("Medikana Knowledge Assistant")
    st.caption("Ask questions about regulatory docs and get answers with context.")

    # Load DB
    db = load_vector_db(OPENAI_API_KEY)
    if not db:
        st.error("Failed to load vector database. Please check the logs.")
        st.stop()

    TAB_DATA = load_table_catalog()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Settings")
        st.write("Vector index:", f"`{INDEX_PATH}`")
        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Render chat history
    # Loops through past messages in session state
    for msg in st.session_state.messages:
        # Display user and assistant messages
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"]) # renders markdown content
            if msg["role"] == "assistant" and msg.get("sources"): # if reeply is from assistant and has sources
                with st.expander("📎 Sources", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        if isinstance(src, Path):
                            st.markdown(f"**{i}.** `{src.name}`")
                            st.caption(f"Path: `{str(src)}`")
                        else:
                            meta = src or {}
                            st.markdown(f"**{i}.** `{meta.get('file_name')}`")
                            st.caption(
                                f"Path: `{meta.get('file_path')}` · Last Modified: {meta.get('last_modified', 'Unknown')}"
                            )

    # Chat input
    user_prompt = st.chat_input("Ask your question…")
    if user_prompt:
        # append user message (display original, not normalized)
        st.session_state.messages.append({"role": "user", "content": user_prompt}) # store user input

        # 1) Table intent path (bypass retrieval)
        if is_table_or_numerical_query(user_prompt):
            table_entry = choose_table_for_query(user_prompt)
            if table_entry:
                answer, table_sources = handle_tabular_query(user_prompt, table_entry)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if table_sources:
                        with st.expander("📎 Sources", expanded=False):
                            for i, meta in enumerate(table_sources, 1):
                                st.markdown(f"**{i}.** `{meta.get('file_name')}`")
                                st.caption(
                                    f"Path: `{meta.get('file_path')}` · Last Modified: {meta.get('last_modified', 'Unknown')}"
                                )
                # update memory
                st.session_state.chat_history.append((user_prompt, answer))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": table_sources}
                )
                st.rerun()
            # fallthrough to retrieval if no relevant table found
        
        # 2) Conversational retrieval path
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Normalize for retrieval (keep original for display)
                countries = extract_country_from_query(user_prompt) # extract countries from query
                norm_q = normalize_text(user_prompt) # normalize user input
                chain = build_crc(db, countries) # build conversational chain with country filter
                result = chain.invoke(
                    {"question": norm_q, "chat_history": st.session_state.chat_history}
                )
                answer = result["answer"] # get answer from chain
                docs = result.get("source_documents", []) or []

                st.markdown(answer)
                if countries:
                    st.caption(f"Country Filter Applied: {', '.join(countries)}")
                src_payload = []
                if docs:
                    # Display sources if available
                    with st.expander("📎 Sources", expanded=False):
                        for i, d in enumerate(docs, 1):
                            meta = d.metadata or {}
                            src_payload.append(
                                {
                                    "file_name": meta.get("file_name"),
                                    "file_path": meta.get("file_path"),
                                    "countries": meta.get("countries"),
                                    "last_modified": meta.get("last_modified"),
                                }
                            )
                            st.markdown(f"**{i}.** `{meta.get('file_name')}`")
                            st.caption(
                                f"Path: `{meta.get('file_path')}` · Last Modified: {meta.get('last_modified', 'Unknown')}"
                            )

        # update memory + history
        st.session_state.chat_history.append((user_prompt, answer))
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": src_payload}
        )
        st.rerun()



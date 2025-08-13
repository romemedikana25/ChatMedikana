# %%

# Path Locator Import
from pathlib import Path
import mimetypes
import os
from datetime import datetime

# File Handling Imports
from utils import (
    extract_pdf_text,
    extract_pptx_text,
    extract_docx_text,
    extract_xlsx_text,
    infer_counties_from_file,
    normalize_text
)

# Open AI and LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Import Google Drive Helpers
from folder_utils import get_folder_id_by_name, download_file
from drive_utils import get_drive_service
from google.oauth2.credentials import Credentials

RUN_SYNC = True

# # Get GOOGLE DRIVE SERVICE and find folders (TESTING PURPOSES)
if RUN_SYNC:
    service = get_drive_service()
    knowledge_base_id = get_folder_id_by_name(service, "Knowledge Base")
    regulatory_folder_id = get_folder_id_by_name(service, "Regulatory", parent_id=knowledge_base_id)
    # Download files from the Regulatory folder
    download_file(service, regulatory_folder_id, local_base_path='test_files')

# ---- CONFIG ----
DATA_FOLDER = "test_files"  # Local folder for your test files
TAB_DATA = []
OPENAI_API_KEY = st.secrets['OPENAI']

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it using `export OPENAI_API_KEY=your_key`.")

# ----------------------------------------------------
# Extract text from different file types
# ----------------------------------------------------
def extract_text(file_path):
    ext = file_path.suffix.lower()

    if ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".pptx":
        return extract_pptx_text(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_xlsx_text(file_path)
    elif ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext in [".txt", ".csv"]:
        return file_path.read_text(errors="ignore")
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return ""
    
# ----- TABLE CATALOG -----
def build_table_catalog(folder_path=DATA_FOLDER):
    for filepath in Path(folder_path).rglob("*"):
        if filepath.suffix.lower() in [".xlsx", ".xls", ".csv"]:
            try:
                if filepath.suffix.lower() in [".xlsx", ".xls"]:
                    df = pd.read_excel(filepath, nrows=5, engine="openpyxl")
                else:
                    df = pd.read_csv(filepath, nrows=5)

                description = f'Table from file {filepath.name} with columns: {", ".join(df.columns)}'
                preview = df.head(3).to_csv(index=False) # First 3 rows as preview

                TAB_DATA.append({
                    "file_path": str(filepath),
                    "description": description,
                    "preview": preview
                })
            except Exception as e:
                print(f"Error reading table {filepath}: {e}")


# ---- VECTOR DB CREATION ----
def load_chunk_files(folder_path=DATA_FOLDER):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    for filepath in Path(DATA_FOLDER).rglob("*"):
        text = extract_text(filepath)
        if not text.strip():
            continue

        text = normalize_text(text)

        filename = filepath.name
        countries = infer_counties_from_file(text)
        filetype = mimetypes.guess_type(filename)[0] or "unknown"
        last_modified = datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()

        chunks = splitter.split_text(text)
        # print(len(chunks), "chunks created for", filename)
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_name": filename,
                        "file_path": str(filepath),
                        "gdrive_path": str(filepath.relative_to(DATA_FOLDER)),
                        "countries": countries,
                        "filetype": filetype,
                        "last_modified": last_modified
                    }
                )
            )

    return docs


def build_faiss_db(docs, index_path='faiss_index'):
    # Create embeddings and FAISS DB
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vec_db = FAISS.from_documents(docs, embeddings)
    return vec_db

# ---- MAIN ----
if __name__ == "__main__":
    print("Loading files and creating vector DB...")
    docs = load_chunk_files(DATA_FOLDER)
    print(f"Indexed {len(docs)} document chunks.")

    # Build table catalog for structured data
    build_table_catalog(DATA_FOLDER)
    print(f"Indexed {len(TAB_DATA)} tables for tabular queries.")

    db = build_faiss_db(docs)
    db.save_local("faiss_index")
    print("Created/Saved FAISS index to 'faiss_index/' folder. Time to query!")

    

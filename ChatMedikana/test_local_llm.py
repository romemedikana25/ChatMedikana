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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# Tabular Data Handling Imports
import pandas as pd
from utils import is_table_or_numerical_query

# Import Google Drive Helpers
from folder_utils import get_folder_id_by_name, download_file
from drive_utils import get_drive_service

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
OPENAI_API_KEY = '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'  # Load from environment variables

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

# ----- HANDLE TABLUAR QUERIES -----
def choose_table_for_query(query):
    """Pick the most relevant table for the query based on description & preview."""
    if not TAB_DATA:
        return None
    
    table_texts = [f"{t['description']}\nPreview:\n{t['preview']}" for t in TAB_DATA]
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_db = FAISS.from_texts(table_texts, embeddings)

    docs = vector_db.similarity_search(query, k=1)
    best_match = docs[0].page_content if docs else None

    for table in TAB_DATA:
        if table['description'] in best_match:
            return table['file_path']
    return None

def handle_tabular_query(query: str, table_path: str):
    """
    Handle queries related to tabular data.
    This is a placeholder function for future implementation.
    """
    try:
        df = pd.read_excel(table_path, engine="openpyxl") if table_path.endswith('.xlsx') else pd.read_csv(table_path)
    except FileNotFoundError:
        return f"File not found: {table_path}"
    except Exception as e:
        return f"Error reading table data: {e}"
    
    # Convert DataFrame to markdown for LLM processing
    markdown_table = df.to_markdown(index=False)

    #create a prompt for the LLM
    prompt = PromptTemplate(
        input_variables=["query", "table"],
        template="""
        You are an expert at answering questions from tabular data.
        Here is the table:
        {table}

        Answer the following question using only the data from the table:

        Question: {question}

        If the answer is a count, provide the exact number.
        If the answer is a list, list the items clearly.
        If numeric calculations are needed, compute them directly.
        If unsure, say "I cannot answer from this table."
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    response = chain.invoke({"table": markdown_table, "query": query})
    return response.content.strip()


def build_faiss_db(docs, index_path='faiss_index'):
    # Create embeddings and FAISS DB
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vec_db = FAISS.from_documents(docs, embeddings)
    return vec_db

# ---- QUERY FUNCTION ----
def query_knowledge_base(query, db):

    if is_table_or_numerical_query(query):
        table_filepath = choose_table_for_query(query)
        if table_filepath:
            return handle_tabular_query(query, table_filepath)
        else:
            print("[INFO] No relevant table found, falling back to vector-based search...")

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
        retriever=retriever
    )
    response = qa.invoke({"query": query})
    return response

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

    # print("Knowledge base created! Type 'exit' to quit.")
    # while True:
    #     query = input("Ask a question: ")
    #     if query.lower() == 'exit':
    #         break
    #     print(query_knowledge_base(query, db))
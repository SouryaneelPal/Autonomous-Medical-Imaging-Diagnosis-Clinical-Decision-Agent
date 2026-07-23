from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- Changed this import to the pure Python PDF loader ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("medagent.rag.ingest")

def ingest_documents(directory_path: str) -> None:
    """
    Ingests clinical documents from a directory, splits them into manageable
    chunks, and saves them into a persistent FAISS vector database.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    logger.info("Loading documents from %s...", directory_path)
    
    # --- Changed the loader to PyPDFDirectoryLoader ---
    loader = PyPDFDirectoryLoader(str(dir_path))
    documents = loader.load()

    if not documents:
        logger.warning("No documents found in %s. Exiting ingestion.", directory_path)
        return

    logger.info("Loaded %d pages. Splitting into chunks...", len(documents))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split documents into %d chunks.", len(chunks))

    logger.info("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    output_dir = Path("vectorstore/faiss_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving FAISS index to %s...", output_dir)
    vectorstore.save_local(str(output_dir))
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest_documents(sys.argv[1])
    else:
        print("Usage: python src/medagent/rag/ingest.py <directory_path_to_documents>")
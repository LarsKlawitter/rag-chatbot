import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def process_pdf(source):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(source)
    documents = loader.load()

    # Filter out scanned pages
    unscanned_documents = [doc for doc in documents if doc.page_content.strip() != ""]

    scanned_pages = len(documents) - len(unscanned_documents)
    if scanned_pages > 0:
        logging.info(f"Omitted {scanned_pages} scanned page(s) from the PDF.")

    if not unscanned_documents:
        raise ValueError(
            "All pages in the PDF appear to be scanned. Please use a PDF with text content."
        )

    return split_documents(unscanned_documents)


def process_image(source):
    from langchain_community.document_loaders.parsers.pdf import extract_from_images_with_rapidocr
    # Extract text from image using OCR
    with open(source, "rb") as image_file:
        image_bytes = image_file.read()

    extracted_text = extract_from_images_with_rapidocr([image_bytes])
    documents = [Document(page_content=extracted_text, metadata={"source": source})]
    return split_documents(documents)


def split_documents(documents):
    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def process_document(source):
    # Determine file type and process accordingly
    if source.lower().endswith(".pdf"):
        return process_pdf(source)
    elif source.lower().endswith((".png", ".jpg", ".jpeg")):
        return process_image(source)
    else:
        raise ValueError(f"Unsupported file type: {source}")


def process_repository(repo_path):
    """
    Process the QuantConnect LEAN repository to extract Python files.
    """
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    documents.append(Document(page_content=f.read(), metadata={"source": file}))
    return split_documents(documents)


def save_embeddings_for_repository(repo_path):
    """
    Process the QuantConnect LEAN repository and save its embeddings to FAISS.
    """
    documents = process_repository(repo_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("lean_embeddings")

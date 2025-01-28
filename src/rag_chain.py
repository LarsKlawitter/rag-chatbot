from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

RAG_PROMPT_TEMPLATE = """
You are a helpful coding assistant that can answer questions about the provided context. The context includes documents from the QuantConnect LEAN framework repository. Use specific code examples from the repository when answering.

If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
"""
PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(chunks=None):
    # Load embeddings for QuantConnect LEAN repository
    if chunks:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        doc_search = FAISS.from_documents(chunks, embeddings)
    else:
        # Otherwise, load precomputed embeddings for the LEAN repository
        doc_search = FAISS.load_local("lean_embeddings")

    retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Load CodeLlama model
    llm = pipeline(
        "text-generation",
        model="codellama/CodeLlama-3B",
        device=0  # Use GPU if available
    )

    # Define the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain

from fastapi import FastAPI, UploadFile, BackgroundTasks, Query
import os
import uuid
import time
import json
import pdfplumber
import boto3

# ---------------- LangChain Community Imports ----------------
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings  # Using LangChain's BedrockEmbeddings for Titan
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document  # For creating docs
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = FastAPI(title="Candidate Interview API", version="1.0")

persist_directory = "./data/chroma"
os.makedirs(persist_directory, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=persist_directory)

collection_name = "pdf_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")

bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"  # Your embedding model
    )

llm = Bedrock(
    client=bedrock_client,
    model_id="amazon.titan-text-express-v1",  # Adjust to your Titan LLM model (e.g., titan-text-lite-v1)
    model_kwargs={"temperature": 0.7, "maxTokenCount": 512}  # Tune as needed
    )

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF extraction error for {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size=150, chunk_overlap=80):  # 150 words, 80 overlap
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

async def embed_and_store(chunks, filename):
    # Create LangChain Document objects for each chunk
    docs = [
        Document(page_content=chunk, metadata={"filename": filename, "chunk": idx})
        for idx, chunk in enumerate(chunks)
    ]
    
    # Use LangChain's Chroma vectorstore to add docs (this handles embeddings internally)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )
    vectorstore.add_documents(docs)
    print(f"✅ Stored {len(chunks)} chunks for {filename}")

async def process_pdf(file_path, filename):
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        print(f"No text to process for {filename}")
        return
    chunks = chunk_text(text)
    print(f"Total chunks to embed: {len(chunks)}")
    await embed_and_store(chunks, filename)

@app.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    os.makedirs("./data/pdfs", exist_ok=True)
    safe_filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join("./data/pdfs", safe_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(process_pdf, file_path, safe_filename)

    return {
        "message": "PDF uploaded successfully. Embedding will be processed in the background.",
        "filename": safe_filename,
        "pdf_path": file_path
    }



@app.get("/verify_embeddings")
def verify_embeddings():
    """
    Verify whether embeddings exist in the Chroma collection
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    results = collection.get(include=["documents", "metadatas", "embeddings"])
    
    total_docs = len(results["documents"])
    embeddings_exist = False
    
    for i in range(total_docs):
        emb = results["embeddings"][i]
        if emb is not None and len(emb) > 0:
            embeddings_exist = True
            break

    return {
        "total_documents": total_docs,
        "embeddings_stored": embeddings_exist
    }


@app.get("/inspect_chunks")
def inspect_chunks():
    collection = chroma_client.get_or_create_collection(name=collection_name)
    results = collection.get(include=["documents", "metadatas", "embeddings"])

    table = []
    for i in range(len(results["documents"])):
        doc_text = results["documents"][i]
        metadata = results["metadatas"][i]

        # embeddings might be numpy arrays or None
        emb = results["embeddings"][i]
        emb_preview = emb[:10].tolist() if emb is not None and len(emb) > 0 else []

        table.append({
            "filename": metadata.get("filename", "unknown"),
            "chunk_index": metadata.get("chunk", i),
            "text_snippet": " ".join(doc_text.split()[:150]),
            "embedding_preview": emb_preview
        })

    return {
        "total_chunks": len(results["documents"]),
        "chunks": table
    }



@app.delete("/delete_pdf/{filename}")
def delete_pdf(filename: str):
    """
    Delete all chunks and embeddings related to a specific PDF file permanently.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Fetch all metadata and IDs (IDs are always included, even without specifying in include)
    results = collection.get(include=["metadatas"])  # Removed "ids" from include
    all_metadatas = results.get("metadatas", [])
    all_ids = results.get("ids", [])  # IDs are still here

    # Filter IDs for this filename
    ids_to_delete = [
        str(all_ids[idx])  # Ensure it's a string
        for idx, meta in enumerate(all_metadatas)
        if meta.get("filename") == filename
    ]

    if not ids_to_delete:
        return {"message": f"No chunks found for filename '{filename}'."}

    # Delete embeddings from Chroma using string IDs
    collection.delete(ids=ids_to_delete)

    # Delete the actual PDF file
    pdf_path = os.path.join("./data/pdfs", filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return {
        "message": f"Deleted PDF '{filename}' and {len(ids_to_delete)} associated chunks permanently."
    }






@app.get("/chat")
def chat_with_nibav(user_query: str = Query(..., description="User's question about Nibav Lifts")):
    """
    Chat API that answers questions about Nibav Lifts using the uploaded PDF documents.
    """

    # Initialize Chroma vectorstore
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )

    # Convert to retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Nibav-specific prompt
    prompt_template = """
    You are Nibav Lifts' official AI assistant.
    Use ONLY the information from the provided Nibav PDF documents
    (like brochures, technical manuals, safety guides, or FAQs)
    to answer questions accurately and clearly.

    Rules:
    - If the answer isn't found in the documents, say: "I couldn’t find that information please ask another question."
    - Keep the tone professional and confident.
    - Don't make up or assume details.

    User Question: {question}

    Context from Nibav PDFs:
    {context}

    AI Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    # Create RetrievalQA chain with Titan LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # 🟢 Step 1: Retrieve matching documents before running the model
    retrieved_docs = retriever.get_relevant_documents(user_query)

    # 🛑 Step 2: If no documents were found, skip the LLM call
    if not retrieved_docs or len(retrieved_docs) == 0:
        return {
            "query": user_query,
            "answer": "I couldn’t find that information please ask another question.",
            "sources": [],
            "chunks_used": []
        }

    # 🟢 Step 3: Run the Titan LLM only if we found relevant chunks
    result = qa_chain.invoke({"query": user_query})

    raw_answer = result.get("result", "No answer generated.")

    def clean_text(text: str) -> str:
        import re
        # Remove duplicate lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        unique_lines = []
        for line in lines:
            if line.lower() not in [l.lower() for l in unique_lines]:
                unique_lines.append(line)
        text = ' '.join(unique_lines)
        # Remove repeated phrases
        text = re.sub(r'\b(\w+(?:, \w+)*)(?:, \1\b)+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        return text

    cleaned_answer = clean_text(raw_answer)

    # Build final response
    response = {
        "query": user_query,
        "answer": cleaned_answer,
        "sources": list(set(doc.metadata.get("filename", "Unknown") for doc in result.get("source_documents", []))),
        "chunks_used": [doc.page_content[:200] + "..." for doc in result.get("source_documents", [])]
    }

    return response



# @app.get("/chat")
# def chat_with_nibav(user_query: str = Query(..., description="User's question about Nibav Lifts")):
#     """
#     Chat API that answers questions about Nibav Lifts using the uploaded PDF documents.
#     """

#     # Initialize Chroma vectorstore
#     vectorstore = Chroma(
#         client=chroma_client,
#         collection_name=collection_name,
#         embedding_function=bedrock_embeddings
#     )

#     # Convert to retriever
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#     # Nibav-specific prompt
#     prompt_template = """
#     You are Nibav Lifts' official AI assistant.
#     Use ONLY the information from the provided Nibav PDF documents
#     (like brochures, technical manuals, safety guides, or FAQs)
#     to answer questions accurately and clearly.

#     Rules:
#     - If the answer isn't found in the documents, say: "I couldn’t find that information please ask another question."
#     - Keep the tone professional and confident.
#     - Don't make up or assume details.

#     User Question: {question}

#     Context from Nibav PDFs:
#     {context}

#     AI Answer:
#     """

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["question", "context"]
#     )

#     # Create RetrievalQA chain with Titan LLM
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )

#     # Run the query
#     result = qa_chain.invoke({"query": user_query})  # ✅ Use "query" key for consistency

#     raw_answer = result.get("result", "No answer generated.")

#     def clean_text(text: str) -> str:
#         import re
#         # Remove duplicate lines
#         lines = [line.strip() for line in text.split('\n') if line.strip()]
#         unique_lines = []
#         for line in lines:
#             if line.lower() not in [l.lower() for l in unique_lines]:
#                 unique_lines.append(line)
#         text = ' '.join(unique_lines)
#         # Remove repeated phrases (like "Bird-watching, Trekking, Camping" twice)
#         text = re.sub(r'\b(\w+(?:, \w+)*)(?:, \1\b)+', r'\1', text)
#         # Fix spacing and capitalization
#         text = re.sub(r'\s+', ' ', text).strip()
#         if text and not text[0].isupper():
#             text = text[0].upper() + text[1:]
#         return text

#     cleaned_answer = clean_text(raw_answer)

#     # Build response
#     response = {
#         "query": user_query,
#         "answer": cleaned_answer,
#         # "sources": [doc.metadata.get("filename", "Unknown") for doc in result.get("source_documents", [])],
#         "sources": list(set(doc.metadata.get("filename", "Unknown") for doc in result.get("source_documents", []))),
#         "chunks_used": [doc.page_content[:200] + "..." for doc in result.get("source_documents", [])]
#     }

#     return response

















# @app.delete("/delete_pdf/{filename}")
# def delete_pdf(filename: str):
#     """
#     Delete all chunks and embeddings related to a specific PDF file
#     """
#     collection = chroma_client.get_or_create_collection(name=collection_name)

#     results = collection.get(include=["metadatas", "documents"])
#     all_metadatas = results["metadatas"]
#     all_ids = results.get("ids", [])

#     # Convert IDs to str to avoid errors
#     all_ids = [str(i) for i in all_ids]

#     ids_to_delete = [
#         all_ids[idx]
#         for idx, meta in enumerate(all_metadatas)
#         if meta.get("filename") == filename
#     ]

#     if not ids_to_delete:
#         return {"message": f"No chunks found for filename '{filename}'."}

#     collection.delete(ids=ids_to_delete)

#     pdf_path = f"./data/pdfs/{filename}"
#     if os.path.exists(pdf_path):
#         os.remove(pdf_path)

#     return {"message": f"Deleted {len(ids_to_delete)} chunks and PDF file '{filename}'."}


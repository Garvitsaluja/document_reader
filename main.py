# main.py
# High-Performance API Server for HackRx 6.0
# Features: Async Processing, Caching, Startup Model Loading

import os
import asyncio
from typing import List, Dict, Any

# --- Core FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

# --- AI and ML Imports ---
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Utility Imports ---
import httpx  # For async HTTP requests to download files and call Gemini
import fitz  # PyMuPDF for reading PDF text
from dotenv import load_dotenv # To load environment variables from .env file

# --- Environment and Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Load your Gemini API Key from an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# This is the specific token required by the HackRx platform
HACKRX_AUTH_TOKEN = "e31e480650ef213ef618fe685acb61ff925d2780a2853e489b73eec846a6a0a7"

# --- Pydantic Models for Request and Response Validation ---
# These models ensure that the data sent to and from your API is in the correct format.

class HackRxRequest(BaseModel):
    """Defines the structure of the incoming request from the HackRx platform."""
    documents: HttpUrl  # The URL of the PDF document
    questions: List[str]  # A list of questions to answer

class HackRxResponse(BaseModel):
    """Defines the structure of the JSON response your API will send back."""
    answers: List[str]

# --- Global State and Caching ---
# We store the embedding model and the document cache in the global state.
# This ensures they are loaded only once and are available for all requests.

app_state: Dict[str, Any] = {
    "embedding_model": None,
    # Caching processed documents drastically improves speed for repeated requests
    # with the same document URL. The key is the document URL, and the value
    # is a dictionary containing the text chunks and their embeddings.
    "document_cache": {},
}

# --- FastAPI Application Setup ---
app = FastAPI(
    title="PolicyGuard AI for HackRx 6.0",
    description="An advanced, high-performance API for intelligent document analysis.",
    version="1.0.0"
)

# Security scheme for bearer token authentication
auth_scheme = HTTPBearer()

def check_auth_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """A dependency that checks the provided bearer token against the required token."""
    if credentials.scheme != "Bearer" or credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    This function runs when the server starts.
    It pre-loads the sentence-transformer model into memory to avoid a
    "cold start" delay on the first request.
    """
    print("Server starting up...")
    if not GEMINI_API_KEY:
        print("CRITICAL ERROR: GEMINI_API_KEY not found. Please set it in your .env file.")
        # In a real app, you might want to exit here, but for the hackathon, we'll let it run.
    print("Loading embedding model... (This may take a moment)")
    # 'all-MiniLM-L6-v2' is a great, lightweight model for this task.
    app_state["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully.")


# --- Core Logic Functions ---

async def get_document_chunks(document_url: str) -> List[str]:
    """
    Downloads a PDF from a URL, extracts text, and splits it into chunks.
    This function is async to handle network I/O without blocking.
    """
    print(f"Processing document from URL: {document_url}")
    try:
        # Use httpx for asynchronous HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, timeout=30.0)
            response.raise_for_status()  # Raise an exception for bad status codes
            pdf_data = response.content

        # Extract text using PyMuPDF (fitz)
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n\n"
        doc.close()

        if not full_text.strip():
            raise ValueError("Could not extract text from the PDF.")

        # Simple but effective chunking strategy
        paragraphs = full_text.split('\n\n')
        chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100] # Filter short/empty paragraphs
        print(f"Document successfully chunked into {len(chunks)} pieces.")
        return chunks

    except httpx.RequestError as e:
        print(f"Error downloading document: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download document from URL: {e}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF document: {e}")

async def get_answer_for_question(question: str, doc_chunks: List[str], doc_embeddings: np.ndarray) -> str:
    """
    Finds the most relevant chunks for a single question and gets an answer from Gemini.
    """
    model = app_state["embedding_model"]
    
    # 1. Embed the question
    question_embedding = model.encode(question, convert_to_tensor=False)

    # 2. Find relevant chunks using cosine similarity
    # We use dot product here because the embeddings are normalized.
    similarities = np.dot(doc_embeddings, question_embedding)
    
    # Get the indices of the top 5 most similar chunks
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [doc_chunks[i] for i in top_indices]
    context = "\n---\n".join(relevant_chunks)

    # 3. Construct the prompt for Gemini
    prompt = f"""
    You are an expert AI assistant. Your task is to answer the following question based ONLY on the provided context from a policy document. Do not use any external knowledge. If the answer is not found in the context, state that clearly.

    **CONTEXT FROM DOCUMENT:**
    {context}

    **QUESTION:**
    {question}

    **ANSWER:**
    """

    # 4. Call the Gemini API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the text from the Gemini response
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer.strip()

    except Exception as e:
        print(f"Error calling Gemini API for question '{question}': {e}")
        # Return a helpful error message in the final response
        return "Error: Could not get an answer from the AI model."


# --- API Endpoint ---

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(
    request: HackRxRequest,
    token: str = Security(check_auth_token)
):
    """
    This is the main endpoint that the HackRx platform will call.
    It orchestrates the entire process of document processing and question answering.
    """
    doc_url = str(request.documents)
    model = app_state["embedding_model"]
    cache = app_state["document_cache"]

    # --- Caching Logic ---
    if doc_url in cache:
        print("Cache hit! Using pre-processed document.")
        doc_chunks = cache[doc_url]["chunks"]
        doc_embeddings = cache[doc_url]["embeddings"]
    else:
        print("Cache miss. Processing new document.")
        # 1. Get and chunk the document text
        doc_chunks = await get_document_chunks(doc_url)
        
        # 2. Embed all chunks
        doc_embeddings = model.encode(doc_chunks, convert_to_tensor=False, show_progress_bar=True)
        
        # Store in cache for future requests
        cache[doc_url] = {"chunks": doc_chunks, "embeddings": doc_embeddings}

    # --- Asynchronous Question Answering ---
    # Create a list of concurrent tasks, one for each question.
    # This is the key to high performance and low latency.
    tasks = [get_answer_for_question(q, doc_chunks, doc_embeddings) for q in request.questions]
    
    print(f"Starting concurrent processing for {len(tasks)} questions...")
    # Run all tasks in parallel and wait for them all to complete.
    answers = await asyncio.gather(*tasks)
    print("All questions processed.")

    return HackRxResponse(answers=answers)

# To run this server for deployment, Render will use a command like:
# uvicorn main:app --host 0.0.0.0 --port 10000


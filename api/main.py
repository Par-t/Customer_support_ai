from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid, os
from pathlib import Path
from services.search_client import SEARCH
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_docs"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Support Copilot - Minimal")

# Serve static web files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web")), name="static")

@app.get("/")
async def serve_index():
    """Serve the main web interface"""
    return FileResponse(str(BASE_DIR / "web" / "index.html"))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint hit")
    return {"status": "healthy", "message": "API is running"}

@app.on_event("startup")
def startup_event():
    # load sample docs on startup
    logger.info(f"Starting up - Loading documents from: {SAMPLE_DIR}")
    SEARCH.load_from_folder(str(SAMPLE_DIR), tenant="demo")
    logger.info(f"Loaded {len(SEARCH.docs)} documents")

@app.post("/ingest")
async def ingest(tenant: str = Form(...), file: UploadFile = File(...)):
    # Save file and add to in-memory store
    logger.info(f"Ingesting file: {file.filename} for tenant: {tenant}")
    content = await file.read()
    doc_id = f"{uuid.uuid4().hex}_{file.filename}"
    path = UPLOAD_DIR / doc_id
    path.write_bytes(content)
    logger.info(f"Saved file to: {path}")
    
    # Try decode as text
    try:
        text = content.decode("utf-8")
        logger.info(f"Successfully decoded text, length: {len(text)} characters")
    except Exception as e:
        text = "(binary file — not parsed)"
        logger.warning(f"Failed to decode file as UTF-8: {e}")
    
    SEARCH.add_doc(doc_id, file.filename, text, tenant=tenant)
    logger.info(f"Added document to search index. Total docs: {len(SEARCH.docs)}")
    return {"status":"ok","id": doc_id, "tenant": tenant}

@app.get("/ingest_sample")
async def ingest_sample():
    # reload sample docs (useful after uploading)
    logger.info("Reloading sample documents")
    SEARCH.docs = []
    SEARCH.load_from_folder(str(SAMPLE_DIR), tenant="demo")
    logger.info(f"Reloaded {len(SEARCH.docs)} sample documents")
    return "Sample docs loaded."

@app.post("/query")
async def query(payload: dict):
    logger.info(f"POST /query endpoint hit with payload: {payload}")
    tenant = payload.get("tenant","demo")
    question = payload.get("question","")
    logger.info(f"Query received - Tenant: {tenant}, Question: '{question}'")
    
    if not question:
        logger.warning("Empty question received")
        return JSONResponse({"answer": "", "sources": []})
    
    # simple retrieval
    top = SEARCH.query(question, tenant=tenant, top_k=3)
    logger.info(f"Found {len(top)} relevant documents")
    
    # Log the search results
    for i, result in enumerate(top):
        logger.info(f"Result {i+1}: {result['title']} (score: {result['score']:.3f})")
    
    # very simple "compose": join top snippets as answer
    if top:
        answer = "\n\n".join([f"{r['title']}: {r['snippet']}" for r in top])
        logger.info(f"Generated answer with {len(answer)} characters")
    else:
        answer = "No relevant documents found."
        logger.info("No relevant documents found for query")
    
    return {"answer": answer, "sources": top}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

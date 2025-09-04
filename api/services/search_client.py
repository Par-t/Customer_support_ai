# Simple TF-IDF based in-memory retriever
from pathlib import Path
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleSearch:
    def __init__(self):
        self.docs = []            # list of dicts: {id, title, text, tenant}
        self.index = None         # fitted TF-IDF
        self.vectorizer = None

    def load_from_folder(self, folder: str, tenant: str = "demo"):
        p = Path(folder)
        logger.info(f"Loading documents from folder: {folder} for tenant: {tenant}")
        files_found = list(p.glob("*.md"))
        logger.info(f"Found {len(files_found)} .md files")
        
        for f in files_found:
            logger.info(f"Loading file: {f.name}")
            text = f.read_text(encoding="utf-8")
            doc = {"id": str(f.name), "title": f.name, "text": text, "tenant": tenant}
            self.docs.append(doc)
            logger.info(f"Added document: {f.name} ({len(text)} characters)")
        
        self._reindex()

    def add_doc(self, doc_id: str, title: str, text: str, tenant: str = "demo"):
        logger.info(f"Adding document: {title} (ID: {doc_id}) for tenant: {tenant}")
        self.docs.append({"id": doc_id, "title": title, "text": text, "tenant": tenant})
        logger.info(f"Total documents now: {len(self.docs)}")
        self._reindex()

    def _reindex(self):
        texts = [d["text"] for d in self.docs]
        logger.info(f"Reindexing with {len(texts)} documents")
        
        if len(texts) == 0:
            logger.info("No documents to index")
            self.index = None
            self.vectorizer = None
            return
        
        logger.info("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.index = self.vectorizer.fit_transform(texts)
        logger.info(f"Index created with shape: {self.index.shape}")

    def query(self, q: str, tenant: str = "demo", top_k: int = 3):
        logger.info(f"Searching for: '{q}' in tenant: {tenant}")
        
        if self.index is None:
            logger.warning("No index available for search")
            return []
        
        # restrict to tenant
        tenant_mask = [i for i,d in enumerate(self.docs) if d.get("tenant","demo")==tenant]
        logger.info(f"Found {len(tenant_mask)} documents for tenant: {tenant}")
        
        if not tenant_mask:
            logger.warning(f"No documents found for tenant: {tenant}")
            return []
        
        doc_vectors = self.index[tenant_mask]
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, doc_vectors)[0]
        
        logger.info(f"Similarity scores: {sims}")
        
        order = np.argsort(-sims)[:top_k]
        results = []
        
        for idx in order:
            doc = self.docs[tenant_mask[idx]]
            score = float(sims[idx])
            # simple snippet: first 200 chars containing any query word
            snippet = doc["text"][:200].replace("\n"," ")
            results.append({"id": doc["id"], "title": doc["title"], "snippet": snippet, "score": score})
            logger.info(f"Result: {doc['title']} (score: {score:.3f})")
        
        logger.info(f"Returning {len(results)} results")
        return results

# single global instance used by the app
SEARCH = SimpleSearch()

import os
from typing import TypedDict, List
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb

# 1. SETUP EMBEDDINGS & CHROMADB
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="zepto_docs")

# Load documents
docs = []
ids = []
for i in range(1, 9):
    filepath = f"docs/doc_{i:02d}.txt"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            docs.append(f.read())
            ids.append(f"doc_{i:02d}")

# Embed and store
if docs:
    embeddings = embedder.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids)
    print(f"Loaded {len(docs)} documents into ChromaDB.")

# 2. LANGGRAPH STATE
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_docs: List[str]
    answer: str
    sources: List[str]
    confidence: float

# 3. NODES
def classify_intent(state: GraphState):
    query = state["query"].lower()
    policy_keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    if any(kw in query for kw in policy_keywords):
        return {"intent": "policy_question"}
    return {"intent": "general_question"}

def retrieve_and_answer(state: GraphState):
    query = state["query"]
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    
    retrieved_chunks = results['documents'][0]
    sources = results['ids'][0]
    
    # MOCK LLM LOGIC (Graded Baseline)
    if os.getenv("MOCK_LLM", "1") == "1":
        top_chunk_snippet = retrieved_chunks[0][:200] if retrieved_chunks else "No context found."
        answer = f"Based on the retrieved context: {top_chunk_snippet}..."
        confidence = 1.0
    else:
        answer = "Real LLM call placeholder" 
        confidence = 0.9
        
    return {"answer": answer, "sources": sources, "confidence": confidence}

def direct_answer(state: GraphState):
    if os.getenv("MOCK_LLM", "1") == "1":
        return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
    return {"answer": "Real LLM general answer", "sources": [], "confidence": 0.9}

# 4. BUILD LANGGRAPH
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    lambda x: x["intent"],
    {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"}
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
app_graph = workflow.compile()

# 5. FASTAPI WRAPPER
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(request: QueryRequest):
    initial_state = {"query": request.query, "intent": "", "retrieved_docs": [], "answer": "", "sources": [], "confidence": 0.0}
    result = app_graph.invoke(initial_state)
    return QueryResponse(answer=result["answer"], sources=result["sources"], confidence=result["confidence"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

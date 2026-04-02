from src.data_loader import process_all_documents, split_documents
from src.embeddings import EmbManagaer
from src.vectorstore import VectorStore
from src.search import RAGRetriever
from src.model import GroqLLM, AdvancedRAGPipline
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Data dir
dir = "data/pdf_files"
logging.info(f'Data directory: {dir}')

# Loading documents
all_documents = process_all_documents(dir)
logging.info(f"Documents Loaded.")

# Chunking documents
docs_chunked = split_documents(all_documents, chunk_size=1000, overlap_size=100)
logging.info(f"Documents Chunked.")

# init the embeddings manager
emb_manager = EmbManagaer("all-MiniLM-L6-v2")
logging.info(f"Embedding model loaded.")

# Init the vector store
vector_store = VectorStore(collection_name="pdfs", persist_directory="../data/vector_store")
logging.info("Vector store initialized or loaded.")

# Extracting the texts from the texts
texts = [chunk.page_content for chunk in docs_chunked]
# Passing text to the emb model to generate embeddings
embeddings = emb_manager.generate_embeddings(texts)
logging.info("Embeddings generated for documents")
# Storing embeddings with thier documents
vector_store.add_document(docs_chunked, embeddings)
logging.info("Embeddings and documents_chunks are stored")

# Seting up the retriever
rag_retriever = RAGRetriever(vector_store, emb_manager)
logging.info("Retriever is sitten up")

# Loading the model
logging.info("Loading the LLM")
try:
    groq_llm = GroqLLM(api_key=os.getenv("groq_api_key"))
    logging.info("Groq LLM initialized succesfully")

except Exception as e:
    logging.info("LLM is not loaded, please provide the API_KEY as env var")
    raise


groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)
# Setting up the Pipline
adv_rag = AdvancedRAGPipline(rag_retriever, llm)
result = adv_rag.query("how ESRGAN work ? in super resolution?", 5, 0.0, False, True)
print("\nFinal Answer:", result['answer'])
print("Summary:", result['summary'])
print("History:", result['history'][-1])
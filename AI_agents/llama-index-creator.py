import json
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
import asyncio
import chromadb

llm = Ollama(model="llama3:latest", request_timeout=60)
response = llm.complete("What is the capital of France?")
print(response)

reader = SimpleDirectoryReader(
    input_dir="./data",
)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents.")

db = chromadb.PersistentClient(path="./story_chroma_db")
chroma_collection = db.get_or_create_collection("story")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

async def main():
    nodes = await pipeline.arun(documents=[Document.example()])
    return nodes

nodes = asyncio.run(main())


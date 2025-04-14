from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.ollama import Ollama
import asyncio
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.core.workflow import Context


# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = Ollama(model="qwen2.5:latest", request_timeout=60)

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

# Create a context
tx = Context(agent)

db = chromadb.PersistentClient(path="./story_chroma_db")
chroma_collection = db.get_or_create_collection("story")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="story_query_engine",
    description="A tool to query the chidren's story database.",
    return_direct=False,
)

query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful story telling assistant that has access to a database containing children's stories. "
)


async def main():
    while True:
        # Run everything in a single event loop
        print("Enter your query:")
        x = input()
        if x == "q":
            break
        if "story" in x:
            x+="After the end of each paragraph add just the keyword to highlight the most important emotion in the story so far. The keywords are: [happy, sad, fear, anger, disgust]"
        
        print(f"Running query engine agent...{x}")
        # Run the query engine agent
        response = await query_engine_agent.run(x, ctx=tx)
        print(response)

asyncio.run(main())
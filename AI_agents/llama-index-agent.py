from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.evaluation import AnswerRelevancyEvaluator
from TTS.api import TTS
import sounddevice as sd
import torch

db = chromadb.PersistentClient(path="./story_chroma_db")
chroma_collection = db.get_or_create_collection("story")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = Ollama(model="llama3:latest", request_timeout=60)
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
response = query_engine.query("Tell me a children's story about a kid who has ADHD.")

print(str(response))
# # query index
# evaluator = AnswerRelevancyEvaluator(llm=llm)
# eval_result = evaluator.evaluate_response(response=response)
# print(eval_result.passing)

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph")
device = "cuda" if torch.cuda.is_available() else "cpu"
wav = tts.tts(text=str(response), speaker_wav="audio/0.wav", device=device, progress_bar=True)

# Play the waveform
sd.play(wav, samplerate=22050)
sd.wait()  # Wait until sound has finished playing
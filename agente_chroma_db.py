import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

READER_DIR = "C:/Codigo/Agents/LlamaIndex-agent/Reader"
DB_PATH = "./agente_basico_asyncio_chroma_db"
COLLECTION = "agente_basico_asyncio"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
HF_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

def build_llm():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    print("Loading LLM...")
    llm = HuggingFaceInferenceAPI(model_name=HF_MODEL_ID, temperature=0.7, max_tokens=100, token=token, provider="auto")
    print("LLM ready")
    return llm

def load_documents():
    print("Loading documents...")
    reader = SimpleDirectoryReader(input_dir=READER_DIR)
    docs = reader.load_data()
    if not docs:
        raise RuntimeError("No documents found in Reader")
    print(f"Documents: {len(docs)}")
    print(f"Sample: {docs[0].text[:100]}...")
    return docs

def build_vector_store():
    print("Preparing ChromaDB...")
    db = chromadb.PersistentClient(path=DB_PATH)
    col = db.get_or_create_collection(COLLECTION)
    vs = ChromaVectorStore(chroma_collection=col)
    print("ChromaDB ready")
    return db, col, vs

def build_pipeline(vector_store):
    print("Building ingestion pipeline...")
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
    pipe = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20), embed],
        vector_store=vector_store,
    )
    print("Pipeline ready")
    return pipe, embed

async def run_ingestion(pipe, documents):
    print("Running ingestion...")
    nodes = await pipe.arun(documents=documents)
    print(f"Nodes: {len(nodes)}")

def build_index(vector_store, embed_model):
    print("Building index...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    print("Index ready")
    return index

def build_query_engine(index, llm):
    print("Building query engine...")
    qe = index.as_query_engine(llm=llm, similarity_top_k=3, response_mode="tree_summarize")
    print("Query engine ready")
    return qe

def evaluate_answer(llm, response_obj):
    print("Evaluating response...")
    evaluator = FaithfulnessEvaluator(llm=llm)
    result = evaluator.evaluate_response(response=response_obj)
    print(f"Faithful: {result.passing}")
    return result.passing

def build_tools(index, llm):
    print("Registering tools...")
    def query_docs(q: str) -> str:
        qe = index.as_query_engine(llm=llm, similarity_top_k=3, response_mode="tree_summarize")
        r = qe.query(q)
        return str(r)
    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny"
    t_query = FunctionTool.from_defaults(fn=query_docs, name="query_docs", description="Query the local document index")
    t_weather = FunctionTool.from_defaults(fn=get_weather, name="weather", description="Get weather by city")
    print("Tools ready")
    return t_query, t_weather

async def main():
    llm = build_llm()
    docs = load_documents()
    db, col, vs = build_vector_store()
    pipe, embed = build_pipeline(vs)
    await run_ingestion(pipe, docs)
    print(f"Chroma count: {col.count()}")
    index = build_index(vs, embed)
    qe = build_query_engine(index, llm)
    ans = qe.query("what is the meaning of life?")
    print(ans)
    _ = evaluate_answer(llm, ans)
    t_query, t_weather = build_tools(index, llm)
    print(t_weather.call("Barcelona"))
    print(t_query.call("Summarize the docs"))

    print("\nMini agent ready")
    print("=" * 50)

    while True:
        q = input("\nAsk (exit to quit): ").strip()
        if q.lower() in {"exit", "quit", "salir"}:
            print("Bye")
            break

        if "weather" in q.lower() or "clima" in q.lower() or "tiempo" in q.lower():
            city = q.replace("weather", "").replace("clima", "").replace("tiempo", "").strip() or "Barcelona"
            print("Tool:", t_weather.call(city))
        else:
            print("Tool:", t_query.call(q))

if __name__ == "__main__":
    asyncio.run(main())

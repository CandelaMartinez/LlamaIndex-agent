import asyncio
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.tools import FunctionTool
from llama_index.tools.google import GmailToolSpec

async def main():
    
    # 1. CONFIGURACIÓN BÁSICA
    print("🚀 INICIANDO VERIFICACIÓN...")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    print("✅ .env cargado")

    # 2. PROBAR LLM
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.7,
        max_tokens=100,
        token=hf_token,
        provider="auto"
    )
    response = llm.complete("Responde 'OK' si funcionas:")
    print(f"✅ LLM: {response}")

    # 3. CARGAR DOCUMENTOS
    reader = SimpleDirectoryReader(input_dir="C:/Codigo/Agents/LlamaIndex-agent/Reader")
    documents = reader.load_data()
    print(f"✅ Documentos cargados: {len(documents)}")
    print(f"📄 Contenido: {documents[0].text[:100]}...")

    # 4. CONFIGURAR CHROMADB
    db = chromadb.PersistentClient(path="./agente_basico_asyncio_chroma_db")
    chroma_collection = db.get_or_create_collection("agente_basico_asyncio")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print("✅ ChromaDB configurado")

    # 5. PROCESAR DOCUMENTOS
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=20),
            embed_model,
        ],
        vector_store=vector_store,
    )

    nodes = await pipeline.arun(documents=documents)
    print(f"✅ Nodos creados: {len(nodes)}")

    # 6. CREAR ÍNDICE
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    print("✅ Índice creado")

    # 7. VERIFICACIONES FINALES
    count = chroma_collection.count()
    print(f"📊 Documentos en ChromaDB: {count}")
    
    #querying a vectorstoreindex with prompts and llms
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
    meaning_of_life = query_engine.query("what is the meaning of life?")
    print(meaning_of_life)

    #evaluation and observability
    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=meaning_of_life)
    evaluation_res = eval_result.passing
    print(evaluation_res)

    #using tools in llamaIndex
    def get_weather(location:str) -> str:
        """usefuk gor getting the weather for a given location"""
        print(f"Getting weather for {location}")
        return f"the weather in {location} is sunny"
    
    tool = FunctionTool.from_defaults(
        get_weather,
        name="my_weather_tool",
        description="Useful for getting the weather for a given location"
    )
    tool.call("Barcelona")

    #pip install llama-index-tools-google
    tool_spec = GmailToolSpec()
    tool_spec_list = tool_spec.to_tool_list()
    [(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
    print(f"Tool name and tool description: {tool.metadata.name,tool.metadata.description}")

   





if __name__ == "__main__":
    asyncio.run(main())
import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vector_chat")

# Carga de env
load_dotenv()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
DIRECTORIO_DOCS = os.getenv("DIRECTORIO_DOCS")
CHROMA_PATH     = os.getenv("CHROMA_PATH")
if not all([OPENAI_API_KEY, DIRECTORIO_DOCS, CHROMA_PATH]):
    logger.critical("Faltan variables de entorno.")
    sys.exit(1)

def cargar_documentos(directorio):
    logger.info(f"Cargando docs de {directorio}")
    docs = []
    for f in os.listdir(directorio):
        ruta = os.path.join(directorio, f)
        if f.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(ruta).load())
        elif f.lower().endswith(".txt"):
            docs.extend(TextLoader(ruta).load())
    logger.info(f"{len(docs)} documentos cargados.")
    return docs

def trocear_documentos(docs):
    logger.info("Troceando docs en chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    trozos = splitter.split_documents(docs)
    logger.info(f"{len(trozos)} chunks generados.")
    return trozos

def crear_vectorstore(trozos):
    logger.info("Creando vectorstore con Chroma")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(
        trozos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    # ← NO persist()
    return vectordb

def cargar_vectorstore():
    logger.info("Cargando vectorstore existente")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def responder_pregunta(pregunta, vectordb):
    docs_sim = vectordb.similarity_search(pregunta, k=4)
    contexto = "\n".join(d.page_content for d in docs_sim)
    modelo = ChatOpenAI(
        model_name="gpt-4.1",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2
    )
    prompt = (
        "Responde SOLO con base en este contexto:\n"
        f"{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
    )
    try:
        out = modelo.invoke(prompt)
        return out.content
    except Exception as e:
        logger.error(f"Error al consultar el modelo: {e}")
        return "❌ Error al generar respuesta."

if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    if not os.listdir(CHROMA_PATH):
        # Si está vacío, indexamos
        docs = cargar_documentos(DIRECTORIO_DOCS)
        trozos = trocear_documentos(docs)
        crear_vectorstore(trozos)
        logger.info("Indexación completada.")
    else:
        logger.info("Índice ya existe, omitiendo indexación.")

    vectordb = cargar_vectorstore()
    logger.info("Chat listo. Ctrl+C para salir.")
    try:
        while True:
            q = input(">>> ").strip()
            if not q: continue
            print("\n" + responder_pregunta(q, vectordb) + "\n")
    except KeyboardInterrupt:
        logger.info("Saliendo.")

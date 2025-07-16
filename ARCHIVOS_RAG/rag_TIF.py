import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import tkinter as tk
from tkinter import scrolledtext, ttk

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

def lanzar_gui(vectordb):
    # Colores institucionales suaves
    BG_COLOR = "#f5f7fa"
    BOT_COLOR = "#1a73e8"
    USER_COLOR = "#34a853"
    TITLE_COLOR = "#22223b"
    FONT = ("Segoe UI", 11)
    TITLE_FONT = ("Segoe UI", 16, "bold")

    def preguntar(event=None):
        pregunta = entry.get()
        if not pregunta.strip():
            return
        chat_area.config(state='normal')
        chat_area.insert(tk.END, f"Tú: {pregunta}\n", 'user')
        chat_area.config(state='disabled')
        chat_area.see(tk.END)
        entry.delete(0, tk.END)
        root.update()
        respuesta = responder_pregunta(pregunta, vectordb)
        chat_area.config(state='normal')
        chat_area.insert(tk.END, f"Diplomatura IA: {respuesta}\n\n", 'bot')
        chat_area.config(state='disabled')
        chat_area.see(tk.END)

    root = tk.Tk()
    root.title("Diplomatura en IA Aplicada a Entornos Digitales de Gestión FCE-UB")
    root.configure(bg=BG_COLOR)
    root.resizable(False, False)

    # Título institucional
    title_frame = tk.Frame(root, bg=BG_COLOR)
    title_frame.pack(padx=10, pady=(10, 0), fill=tk.X)
    title_label = tk.Label(
        title_frame,
        text="🤖 Diplomatura en IA Aplicada a Entornos Digitales de Gestión\nFCE-UB",
        font=TITLE_FONT,
        fg=TITLE_COLOR,
        bg=BG_COLOR,
        justify="center"
    )
    title_label.pack()

    # Área de chat
    chat_area = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, width=80, height=22, state='disabled',
        font=FONT, bg="white", relief=tk.FLAT, bd=2
    )
    chat_area.tag_config('user', foreground=USER_COLOR, font=FONT)
    chat_area.tag_config('bot', foreground=BOT_COLOR, font=FONT)
    chat_area.pack(padx=10, pady=(10, 5), fill=tk.BOTH, expand=True)

    # Frame de entrada y botón
    input_frame = tk.Frame(root, bg=BG_COLOR)
    input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

    entry = tk.Entry(input_frame, width=70, font=FONT, relief=tk.GROOVE, bd=2)
    entry.pack(side=tk.LEFT, padx=(0, 8), pady=2, expand=True, fill=tk.X)
    entry.focus()

    send_btn = ttk.Button(input_frame, text="Enviar", command=preguntar)
    send_btn.pack(side=tk.LEFT, padx=(0, 0), pady=2)

    root.bind('<Return>', preguntar)

    # Mensaje de bienvenida
    chat_area.config(state='normal')
    chat_area.insert(
        tk.END,
        "Bienvenido/a al asistente de la Diplomatura en IA Aplicada a Entornos Digitales de Gestión (FCE-UB).\n"
        "Escribe tu consulta sobre los materiales y recibirás una respuesta basada en los documentos cargados.\n\n",
        'bot'
    )
    chat_area.config(state='disabled')

    root.mainloop()

if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    if not os.listdir(CHROMA_PATH):
        docs = cargar_documentos(DIRECTORIO_DOCS)
        trozos = trocear_documentos(docs)
        crear_vectorstore(trozos)
        logger.info("Indexación completada.")
    else:
        logger.info("Índice ya existe, omitiendo indexación.")

    vectordb = cargar_vectorstore()
    logger.info("Chat listo. Lanzando interfaz gráfica.")
    lanzar_gui(vectordb)
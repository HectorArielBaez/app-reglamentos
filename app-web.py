import streamlit as st
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from google.oauth2 import service_account
import json
import os

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente de Reglamentos",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Asistente de Reglamentos UTN")

# ---------------------------------------------------------------------------
# CARGA Y CONFIGURACIÓN DE CREDENCIALES
# ---------------------------------------------------------------------------

try:
    # Cargar las credenciales desde st.secrets
    creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)

    # Guardar temporalmente el archivo en /tmp
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "w") as f:
        f.write(st.secrets["GCP_CREDENTIALS"])

    project_id = creds_dict["project_id"]
except Exception as e:
    st.error(f"Error al cargar credenciales: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE EMBEDDINGS Y MODELO DE LENGUAJE
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Inicializando embeddings y LLM de Vertex AI...")
def init_models():
    """Inicializa los modelos de embeddings y LLM."""
    embeddings = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002",
        credentials=credentials
    )
    llm = VertexAI(
        model_name="gemini-1.5-flash",
        location="us-central1",
        credentials=credentials,
        temperature=0.2
    )
    return embeddings, llm


embeddings, llm = init_models()

# ---------------------------------------------------------------------------
# CARGA Y PROCESAMIENTO DE DOCUMENTOS PDF
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando y procesando documentos PDF...")
def load_vectorstore():
    """Carga documentos PDF y construye la base vectorial FAISS."""
    loader = PyPDFDirectoryLoader("docs")  # Carpeta con tus reglamentos
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


vectorstore = load_vectorstore()

# ---------------------------------------------------------------------------
# CREACIÓN DE LA CADENA DE RETRIEVAL QA
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Creando cadena de consulta...")
def init_chain():
    """Crea la cadena de RAG (Retrieval-Augmented Generation)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=False
    )
    return qa_chain


rag_chain = init_chain()

# ---------------------------------------------------------------------------
# INTERFAZ DE USUARIO
# ---------------------------------------------------------------------------

st.write("💬 Preguntá sobre los reglamentos de la UTN y obtené respuestas contextualizadas.")

query = st.text_input("Ingresá tu pregunta:")

if query:
    with st.spinner("Buscando respuesta..."):
        try:
            result = rag_chain.invoke(query)
            st.success(result["result"])
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")

import streamlit as st
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from google.oauth2 import service_account
import json
import os

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente de Reglamentos UTN",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Asistente de Reglamentos UTN")

# ---------------------------------------------------------------------------
# CREDENCIALES DE GOOGLE CLOUD
# ---------------------------------------------------------------------------
try:
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
# MODELOS VERTEX AI
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando modelos Vertex AI...")
def init_models():
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
# CARGA DEL TEXTO DESDE UN ARCHIVO TXT
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando texto...")
def load_text_vectorstore(txt_path="reglamento.txt"):
    if not os.path.exists(txt_path):
        st.warning("⚠️ No se encontró el archivo 'reglamento.txt'. Subilo con el cargador de abajo.")
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        st.warning("⚠️ El archivo está vacío.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ---------------------------------------------------------------------------
# INTERFAZ DE USUARIO PARA CARGAR EL TXT
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("📄 Subí el archivo de texto (formato .txt)", type=["txt"])

if uploaded_file is not None:
    temp_path = "/tmp/reglamento.txt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("✅ Archivo cargado correctamente.")
    vectorstore = load_text_vectorstore(temp_path)
else:
    vectorstore = load_text_vectorstore("reglamento.txt")

if not vectorstore:
    st.stop()

# ---------------------------------------------------------------------------
# CADENA RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Creando cadena de consulta...")
def init_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=False
    )
    return qa_chain

rag_chain = init_chain(vectorstore)

# ---------------------------------------------------------------------------
# INTERFAZ DE CONSULTA
# ---------------------------------------------------------------------------
st.write("💬 Ingresá tu pregunta sobre el reglamento cargado:")

query = st.text_input("Pregunta:")

if query:
    with st.spinner("Analizando el texto..."):
        try:
            result = rag_chain.invoke(query)
            st.success(result["result"])
        except Exception as e:
            st.error(f"Error al generar respuesta: {e}")

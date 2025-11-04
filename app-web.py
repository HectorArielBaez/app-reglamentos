# app.py
import streamlit as st
import os
import json
from google.oauth2 import service_account
from google.cloud import aiplatform

# --- Configuración de Autenticación para Google Cloud ---
PROJECT_ID = "rag-v0"
LOCATION = "us-central1"

# 1️⃣ Autenticación según entorno
if "GCP_CREDENTIALS" in st.secrets:
    # Si estamos en Streamlit Cloud: cargar las credenciales desde el secreto
    creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
else:
    # Si estamos en local: usar las credenciales por defecto del entorno
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

# --- Librerías de LangChain / Vertex ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- FUNCIÓN DE CARGA Y CACHÉ ---
@st.cache_resource
def load_rag_chain():
    status_text = st.empty()
    status_text.info("Leyendo documento...")

    loader = TextLoader("CSU_ORD__0__1549_OCR.txt", encoding="utf-8")
    documents = loader.load()

    status_text.info("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    status_text.info("Inicializando modelo de embeddings...")
    embeddings_model = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

    status_text.info("Creando base de datos vectorial con FAISS...")
    vector_store = FAISS.from_documents(document_chunks, embeddings_model)

    status_text.info("Inicializando LLM (Gemini)...")
    llm = VertexAI(
        model_name="gemini-2.0-flash-lite-001",
        location=LOCATION,
        temperature=0.2
    )

    status_text.info("Creando la cadena RAG...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    status_text.success("¡Aplicación lista! Ya puedes hacer tu pregunta.")
    return rag_chain


# --- INTERFAZ DE USUARIO ---
st.title("🤖 Chatbot de Reglamentos")
st.caption("Haz una pregunta sobre el reglamento de la UTN (CSU_ORD__0__1549_OCR.txt)")

try:
    rag_chain = load_rag_chain()

    query = st.text_input("Escribe tu pregunta:", placeholder="Ej: ¿Cuántos días de vacaciones tengo por año?")

    if query:
        with st.spinner("Buscando en el reglamento y generando respuesta..."):
            result = rag_chain.invoke(query)
            st.success("**Respuesta:**")
            st.write(result['result'])

            with st.expander("Ver las fuentes utilizadas"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Fuente:** `{doc.metadata.get('source', 'N/A')}`")
                    st.info(doc.page_content)

except Exception as e:
    st.error("Ha ocurrido un error al inicializar la aplicación:")
    st.exception(e)
    st.warning("Verifica tus credenciales o el ID del modelo en el Model Garden.")

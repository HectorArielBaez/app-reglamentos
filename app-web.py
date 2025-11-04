# app.py
import streamlit as st
import os

# --- Configuración de Autenticación para Google Cloud ---

# Revisa si estamos corriendo en Streamlit Cloud (donde el secret "GCP_CREDENTIALS" existe)
if "GCP_CREDENTIALS" in st.secrets:
    # Si estamos desplegados, toma el JSON del secret
    creds_json_str = st.secrets["GCP_CREDENTIALS"]

    # Escribe el JSON en un archivo temporal (que está en .gitignore)
    with open("gcp_key.json", "w") as f:
        f.write(creds_json_str)

    # Setea la variable de entorno para que las bibliotecas de Google
    # encuentren y usen este archivo de clave.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# Si no estamos en Streamlit Cloud, (estamos en local),
# las bibliotecas usarán tu autenticación por defecto ('gcloud auth ...')

# Setea el proyecto (esto lo necesitamos en ambos casos)
os.environ["GCLOUD_PROJECT"] = "rag-v0"

import vertexai

# Replace "your-project-id" with your actual Project ID
vertexai.init(project="Rag-v0", location="us-central1")


# app.py - Tu frontend con Streamlit
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA

# Configura tu proyecto de Google Cloud
# (Asegúrate de que esta variable de entorno esté configurada
# o que te hayas autenticado con 'gcloud auth application-default login')
#os.environ["GCLOUD_PROJECT"] = "rag-v0" 

# --- FUNCIÓN DE CARGA Y CACHÉ ---
# Esta función se ejecutará UNA SOLA VEZ gracias a @st.cache_resource
# Carga los modelos y crea el RAG chain, luego lo guarda en memoria.
@st.cache_resource
def load_rag_chain():
    # Creamos un placeholder para mostrar mensajes de estado en la UI
    status_text = st.empty()

    # --- 1. Cargar el Documento ---
    status_text.info("leyendo")
    loader = TextLoader("CSU_ORD__0__1549_OCR.txt", encoding="utf-8")
    documents = loader.load()

    # --- 2. Dividir el Texto (Chunking) ---
    status_text.info("Dividiendo documentos en fragmentos...") 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    document_chunks = text_splitter.split_documents(documents)

    # --- 3. Inicializar Modelo de Embeddings ---
    status_text.info("Inicializando modelo de embeddings...")
    # Asegúrate de usar el nombre que te funcionó:
    embeddings_model = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002" 
    )

    # --- 4. Crear Base de Datos Vectorial ---
    status_text.info("Creando base de datos vectorial con FAISS...")
    vector_store = FAISS.from_documents(document_chunks, embeddings_model)

    # --- 5. Inicializar el LLM ---
    status_text.info("Inicializando LLM (Gemini)...")
    # !!! IMPORTANTE: Usa el ID del modelo que te funcionó !!!
    # (El que encontraste en el Model Garden)
    llm = VertexAI(
        model_name="gemini-2.0-flash-lite-001", # ej: "gemini-2.0-flash-lite-001"
        location="us-central1",
        temperature=0.2
    )

    # --- 6. Crear la Cadena RAG ---
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

# --- INTERFAZ DE USUARIO PRINCIPAL ---

st.title("🤖 Chatbot de Reglamentos")
st.caption("Haz una pregunta sobre el reglamento de la UTN (CSU_ORD__0__1549_OCR.txt)")

try:
    # Carga la cadena RAG (usará la caché si ya está cargada)
    rag_chain = load_rag_chain()
    
    # --- 7. Caja de texto para la pregunta ---
    query = st.text_input("Escribe tu pregunta:", placeholder="Ej: ¿Cuántos días de vacaciones tengo por año?")

    # Si el usuario escribió una pregunta (y presionó Enter)
    if query:
        # Muestra un indicador de "pensando..."
        with st.spinner("Buscando en el reglamento y generando respuesta..."):
            
            # --- 8. Ejecutar la Cadena RAG ---
            result = rag_chain.invoke(query)
            
            # --- 9. Mostrar Resultados ---
            st.success("**Respuesta:**")
            st.write(result['result'])
            
            # --- 10. Mostrar Fuentes (opcional pero recomendado) ---
            with st.expander("Ver las fuentes utilizadas"):
                st.write("Fragmentos del documento que usó el LLM para responder:")
                for doc in result["source_documents"]:
                    st.markdown(f"**Fuente:** `{doc.metadata.get('source', 'N/A')}`")
                    st.info(doc.page_content)

except Exception as e:
    st.error(f"Ha ocurrido un error al inicializar la aplicación:")
    st.error(e)
    st.warning("Verifica que el `model_name` del LLM en el script `app.py` sea el ID de modelo correcto que encontraste en el Model Garden.")
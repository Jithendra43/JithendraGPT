import os
import io
import streamlit as st
import dotenv
import unicodedata
from typing import Optional
import pickle

# ---- Try importing speech recognition with fallback ----
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    st.warning("Speech recognition not available. Install required packages for speech functionality.")

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("Text-to-speech not available. Install gtts for speech output.")

# ---- OpenAI Imports ----
from langchain_openai import ChatOpenAI

# ---- LangChain & Related Imports ----
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Try to import vector stores with fallback priority
VECTOR_STORE_TYPE = None
try:
    from langchain_community.vectorstores import FAISS
    VECTOR_STORE_TYPE = "FAISS"
except ImportError:
    try:
        from langchain_chroma import Chroma
        VECTOR_STORE_TYPE = "CHROMA"
    except ImportError:
        st.error("❌ No vector store available. Please check your dependencies.")
        st.stop()

# ---- Load environment variables ----
dotenv.load_dotenv()

# Set up API keys with Streamlit secrets fallback
def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment or Streamlit secrets"""
    # First try environment variables
    key = os.getenv(key_name)
    if key:
        return key
    
    # Then try Streamlit secrets
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        return None

# Set up API keys
groq_api_key = get_api_key("GROQ_API_KEY")
openai_api_key = get_api_key("OPENAI_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# -------------------------------------
# 1. Define Functions
# -------------------------------------
def setup_vectorstore(persist_directory: str = "vector_db_dir"):
    """Setup vector store with FAISS priority and ChromaDB fallback"""
    if "vectorstore" not in st.session_state:
        try:
            # Initialize embeddings with better error handling
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': False
                    },
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as embedding_error:
                # Fallback to basic initialization if advanced options fail
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            if VECTOR_STORE_TYPE == "FAISS":
                # Try to load existing FAISS index
                faiss_index_path = os.path.join(persist_directory, "faiss_index")
                if os.path.exists(faiss_index_path + ".faiss") and os.path.exists(faiss_index_path + ".pkl"):
                    try:
                        st.session_state.vectorstore = FAISS.load_local(
                            faiss_index_path, 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                    except Exception as e:
                        # Create empty FAISS index if loading fails
                        st.session_state.vectorstore = None
                else:
                    # Create empty FAISS index - will be populated when documents are uploaded
                    st.session_state.vectorstore = None
            
            elif VECTOR_STORE_TYPE == "CHROMA":
                # Fallback to ChromaDB
                try:
                    from langchain_chroma import Chroma
                    st.session_state.vectorstore = Chroma(
                        persist_directory=persist_directory, 
                        embedding_function=embeddings
                    )
                except Exception as chroma_error:
                    # Try alternative ChromaDB initialization
                    import chromadb
                    from chromadb.config import Settings
                    
                    client = chromadb.PersistentClient(
                        path=persist_directory,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    st.session_state.vectorstore = Chroma(
                        client=client,
                        embedding_function=embeddings
                    )
                
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return None
    
    return st.session_state.vectorstore

def create_conversational_chain(vectorstore, model_choice, temperature=0.5, provider="groq"):
    """
    Creates a ConversationalRetrievalChain using ChatGroq or ChatOpenAI as LLM,
    with a ConversationBufferMemory.
    """
    try:
        if provider == "openai":
            if not openai_api_key:
                st.error("OpenAI API key not found. Please add it to your secrets.")
                return None
            llm = ChatOpenAI(model_name=model_choice, temperature=temperature)
        else:
            if not groq_api_key:
                st.error("Groq API key not found. Please add it to your secrets.")
                return None
            llm = ChatGroq(model=model_choice, temperature=temperature)
        
        if vectorstore is None:
            st.warning("⚠️ No vector store available. Please upload documents first.")
            return None
            
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            verbose=True,
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def vectorize_new_documents(files, persist_directory="vector_db_dir"):
    """Vectorize uploaded documents with FAISS priority"""
    temp_folder = "temp_uploads"
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        saved_file_paths = []
        for file in files:
            file_path = os.path.join(temp_folder, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            saved_file_paths.append(file_path)
        
        all_docs = []
        for file_path in saved_file_paths:
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        text_chunks = text_splitter.split_documents(all_docs)
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False
                },
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as embedding_error:
            # Fallback to basic initialization if advanced options fail
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        if VECTOR_STORE_TYPE == "FAISS":
            # Create or update FAISS index
            faiss_index_path = os.path.join(persist_directory, "faiss_index")
            
            if st.session_state.vectorstore is None:
                # Create new FAISS index
                vectordb = FAISS.from_documents(text_chunks, embeddings)
            else:
                # Add to existing FAISS index
                vectordb = st.session_state.vectorstore
                vectordb.add_documents(text_chunks)
            
            # Save FAISS index
            vectordb.save_local(faiss_index_path)
            return vectordb
            
        elif VECTOR_STORE_TYPE == "CHROMA":
            # Fallback to ChromaDB
            try:
                from langchain_chroma import Chroma
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            except Exception as chroma_error:
                st.error(f"ChromaDB initialization failed: {str(chroma_error)}")
                # Try alternative ChromaDB initialization
                import chromadb
                from chromadb.config import Settings
                
                client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                vectordb = Chroma(
                    client=client,
                    embedding_function=embeddings
                )
            
            vectordb.add_documents(text_chunks)
            return vectordb
            
    except Exception as e:
        st.error(f"Error vectorizing documents: {str(e)}")
        return None

def clear_chat_history():
    """Clear chat history"""
    st.session_state.chat_history = []
    st.success("Chat history deleted.")

def clean_text(text):
    """Remove problematic Unicode characters to prevent encoding errors."""
    return "".join(c for c in text if unicodedata.category(c)[0] != "C")

def transcribe_speech():
    """Transcribe speech with error handling"""
    if not SPEECH_AVAILABLE:
        st.error("Speech recognition is not available. Please install required packages.")
        return None
    
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
    except sr.WaitTimeoutError:
        st.error("Listening timed out. Please speak louder or check your microphone.")
    except Exception as e:
        st.error(f"Error during speech recognition: {e}")
    return None

# -------------------------------------
# 2. Streamlit UI Configuration
# -------------------------------------
st.set_page_config(
    page_title="JithendraGPT", 
    page_icon="📚", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>JithendraGPT: AI-Powered Document Insight Engine</h1>", unsafe_allow_html=True)
st.markdown("#### How can I assist you today?")

# Vector store is ready - no need to display status messages

# Check for API keys
if not groq_api_key and not openai_api_key:
    st.error("⚠️ No API keys found! Please add your GROQ_API_KEY or OPENAI_API_KEY to Streamlit secrets.")
    st.info("Go to your Streamlit Cloud dashboard → App settings → Secrets to add your API keys.")
    st.stop()

# -------------------------------------
# 3. Sidebar - Configuration Panel
# -------------------------------------
st.sidebar.header("Configuration Panel")

temperature = st.sidebar.slider("Response Complexity (Temperature)", 0.0, 1.0, 0.5, 0.1)

# --- Model Selection ---
available_providers = []
if groq_api_key:
    available_providers.append("groq")
if openai_api_key:
    available_providers.append("openai")

if not available_providers:
    st.error("No API keys available. Please add your API keys to continue.")
    st.stop()

model_provider = st.sidebar.selectbox("Choose Model Provider", available_providers)

if model_provider == "openai":
    model_choice = st.sidebar.selectbox("OpenAI Model", [
        "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"
    ])
else:
    model_choice = st.sidebar.selectbox("Groq Model", [
        "llama-3.3-70b-versatile", 
        "llama-3.1-8b-instant", 
        "deepseek-r1-distill-llama-70b", 
        "mixtral-8x7b-32768", 
        "qwen-2.5-32b"
    ])

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, CSV)", 
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "csv"]
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        result = vectorize_new_documents(uploaded_files, persist_directory="vector_db_dir")
        if result:
            st.sidebar.success("Documents successfully vectorized!")
            st.session_state.vectorstore = result
            st.session_state.pop("conversational_chain", None)

if st.sidebar.button("Clear Chat History"):
    clear_chat_history()

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Jithendra Pavuluri**")

# -------------------------------------
# 4. Input/Output Mode Selection
# -------------------------------------
col1, col2 = st.columns(2)
with col1:
    input_modes = ["Text"]
    if SPEECH_AVAILABLE:
        input_modes.append("Speech")
    input_mode = st.selectbox("Choose Input Mode", input_modes)

with col2:
    output_modes = ["Text"]
    if TTS_AVAILABLE:
        output_modes.append("Speech")
    output_mode = st.selectbox("Choose Output Mode", output_modes)

# -------------------------------------
# 5. Initialize Vectorstore & Conversational Chain
# -------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore(persist_directory="vector_db_dir")

# Only create conversational chain if we have a vectorstore with documents
if st.session_state.vectorstore is not None and (
    "conversational_chain" not in st.session_state 
    or st.session_state.get("current_temp") != temperature
    or st.session_state.get("selected_model") != model_choice
    or st.session_state.get("selected_provider") != model_provider
):
    with st.spinner("Initializing AI model..."):
        st.session_state.conversational_chain = create_conversational_chain(
            st.session_state.vectorstore,
            model_choice,
            temperature=temperature,
            provider=model_provider
        )
        st.session_state.current_temp = temperature
        st.session_state.selected_model = model_choice
        st.session_state.selected_provider = model_provider

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------
# 6. Display Chat History
# -------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------
# 7. User Input Handling
# -------------------------------------
user_input = None

if input_mode == "Text":
    user_input = st.chat_input("Ask AI...")

elif input_mode == "Speech" and SPEECH_AVAILABLE:
    if st.button("🎙️ Start Speaking"):
        user_input = transcribe_speech()

if user_input:
    # Check if we have a conversational chain
    if st.session_state.vectorstore is None:
        st.error("📄 Please upload documents first to start chatting!")
        st.stop()
    
    if "conversational_chain" not in st.session_state or st.session_state.conversational_chain is None:
        st.error("🤖 AI model not initialized. Please check your API keys and try again.")
        st.stop()
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversational_chain({"question": user_input})
                assistant_answer = clean_text(response["answer"])
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_answer})
                st.markdown(assistant_answer)
                
                if output_mode == "Speech" and TTS_AVAILABLE:
                    try:
                        tts = gTTS(assistant_answer)
                        buf = io.BytesIO()
                        tts.write_to_fp(buf)
                        buf.seek(0)
                        st.audio(buf, format="audio/mp3")
                    except Exception as e:
                        st.warning(f"Could not generate speech: {str(e)}")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.chat_history.append({"role": "assistant", "content": "I apologize, but I encountered an error while processing your request. Please try again."})

# Show clean welcome message only if no documents uploaded
if st.session_state.vectorstore is None and not uploaded_files:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px; margin: 1rem 0;">
        <h3>🚀 Welcome to JithendraGPT!</h3>
        <p>Upload documents using the sidebar to start chatting with your AI assistant.</p>
    </div>
    """, unsafe_allow_html=True)
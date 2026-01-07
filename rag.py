import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------- CONFIG --------------------

load_dotenv()
st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("📄 RAG Q&A with Multiple PDFs + Chat History")

with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")

api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

# -------------------- LLM + EMBEDDINGS --------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# -------------------- PDF UPLOAD --------------------

uploaded_files = st.file_uploader(
    "📄Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin.")
    st.stop()

all_docs = []
tmp_paths = []

for pdf in uploaded_files:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = pdf.name

    all_docs.extend(docs)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs.")

# -------------------- CLEAN TEMP FILES --------------------

for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

# -------------------- CHUNKING --------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

docs_chunks = text_splitter.split_documents(all_docs)

# -------------------- VECTOR STORE --------------------

INDEX_DIR = "chroma_db"

@st.cache_resource
def get_vectorstore():
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        return Chroma(
            persist_directory=INDEX_DIR,
            embedding_function=embeddings
        )

    return Chroma.from_documents(
        documents=docs_chunks,
        embedding=embeddings,
        persist_directory=INDEX_DIR
    )

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# -------------------- PROMPTS --------------------

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question into a standalone search query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using the context below.\n"
        "If the answer is not present, say you don't know.\n\n"
        "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are analyzing a PDF document.\n"
        "Based on the user's request, generate:\n"
        "- summaries\n"
        "- important points\n"
        "- first or last headings\n"
        "- topic-wise explanations\n\n"
        "Use the document content below.\n\n"
        "Document:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------- CHAT HISTORY --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

def get_chat_history(session_id: str):
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

# -------------------- USER INPUT --------------------

session_id = st.text_input("🆔 Session ID", value="default")
user_question = st.text_input("❓ Ask a question")

chat_history = get_chat_history(session_id)

if not user_question:
    st.stop()

# -------------------- INTENT DETECTION --------------------

summary_keywords = [
    "summary", "summarize", "overview",
    "important", "key points",
    "first", "last", "headings",
    "topics", "explain the pdf",
    "page summary"
]

is_summary = any(k in user_question.lower() for k in summary_keywords)

# -------------------- DOCUMENT SELECTION --------------------

if is_summary:
    # Use ALL chunks in correct order
    selected_docs = sorted(
        docs_chunks,
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0))
    )
else:
    rewrite_msgs = rewrite_prompt.format_messages(
        input=user_question,
        chat_history=chat_history.messages
    )
    standalone_query = llm.invoke(rewrite_msgs).content.strip()
    selected_docs = retriever.invoke(standalone_query)

if not selected_docs:
    st.chat_message("assistant").write("❗ No relevant content found.")
    st.stop()

# -------------------- BUILD CONTEXT --------------------

def join_docs(docs, max_chars=20000):
    text, total = [], 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        text.append(f"[Page {d.metadata.get('page', '')}] {d.page_content}")
        total += len(d.page_content)
    return "\n\n".join(text)

context = join_docs(selected_docs)

# -------------------- FINAL ANSWER --------------------

prompt = summary_prompt if is_summary else qa_prompt

final_msgs = prompt.format_messages(
    input=user_question,
    context=context,
    chat_history=chat_history.messages
)

answer = llm.invoke(final_msgs).content.strip()

# -------------------- DISPLAY --------------------

st.chat_message("user").write(user_question)
st.chat_message("assistant").write(answer)

chat_history.add_user_message(user_question)
chat_history.add_ai_message(answer)

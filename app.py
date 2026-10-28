import os
import tempfile
import requests
import streamlit as st
from rag_engine import LocalRAGPipeline

st.set_page_config(
    page_title="Local RAG & Document Q&A Studio", page_icon="📄", layout="wide"
)

# Initialize Session State
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = LocalRAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# Helper function to fetch local Ollama models dynamically via HTTP API
def get_available_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m["model"] for m in data.get("models", [])]
            if models:
                return models
    except Exception:
        pass

    # Fallback default multi-model list if API fails
    return ["llama3.2:3b", "qwen3.5:2b", "qwen3.5:0.8b"]


# Title & Description
st.title("📄 Local RAG & Document Q&A Bot")
st.caption(
    "Powered by ChromaDB, LangChain, SentenceTransformers, and Local LLMs (Ollama)"
)

# Sidebar Configuration
with st.sidebar:
    st.header("🛠️ 1. Chunking & Indexing Config")
    chunk_strategy = st.selectbox(
        "Chunking Strategy", ["Recursive", "Sentence", "Fixed Size"]
    )
    chunk_size = st.slider(
        "Chunk Size (characters)", min_value=100, max_value=2000, value=500, step=50
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)", min_value=0, max_value=500, value=50, step=10
    )

    st.divider()
    st.header("🔍 2. Retrieval Parameters")
    search_type = st.selectbox(
        "Search Strategy", ["Hybrid (Dense + BM25)", "Dense Vector"]
    )
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=4)
    hybrid_alpha = st.slider(
        "Hybrid Weight (Dense vs BM25)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )

    st.divider()
    st.header("🤖 3. Generator Model")

    # Dynamic Model Picker Dropdown
    available_models = get_available_ollama_models()
    llm_model = st.selectbox(
        "Choose Ollama Model",
        options=available_models,
        index=0,
    )

    st.divider()
    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        st.session_state.rag_pipeline.clear_database()
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.success("Database cleared!")

# Main Tabs
tab_chat, tab_docs, tab_debug = st.tabs(
    ["💬 Document Q&A", "📁 Upload & Index Studio", "🔬 RAG Debug Inspector"]
)

# TAB 1: UPLOAD & INDEX
with tab_docs:
    st.subheader("Upload Documents into Local Vector Database")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Process & Index Files", type="primary"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Write file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Process file
                    stats = st.session_state.rag_pipeline.process_and_index_file(
                        file_path=tmp_path,
                        chunk_strategy=chunk_strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    os.remove(tmp_path)
                    st.session_state.indexed_files.append(uploaded_file.name)

                    st.success(
                        f"Successfully indexed `{uploaded_file.name}` into ChromaDB!"
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Generated Chunks", stats["total_chunks"])
                    col2.metric("Total DB Chunks", stats["total_documents_indexed"])
                    col3.metric("Indexing Time", f"{stats['indexing_time_sec']}s")

                    with st.expander("Preview Sample Chunk"):
                        st.code(stats["sample_chunk"])
        else:
            st.warning("Please select at least one document to upload.")

    st.subheader("Indexed Documents")
    if st.session_state.indexed_files:
        for fname in set(st.session_state.indexed_files):
            st.write(f"• 📄 `{fname}`")
    else:
        st.info("No documents indexed yet.")

# TAB 2: CHAT & RETRIEVAL
with tab_chat:
    st.subheader("Ask Questions Against Document Store")

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(
                            f"**Source {idx+1}** (Page {src.metadata.get('page', 'N/A')}):"
                        )
                        st.info(src.page_content)

    # Chat Input
    if user_query := st.chat_input("Ask a question about your uploaded documents..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner(
                f"Retrieving context & generating answer using {llm_model}..."
            ):
                result = st.session_state.rag_pipeline.answer_query(
                    query=user_query,
                    top_k=top_k,
                    search_type=search_type,
                    model_name=llm_model,
                )

                answer = result["answer"]
                sources = result["context_sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Cited Sources"):
                        for idx, src in enumerate(sources):
                            st.markdown(
                                f"**Source {idx+1}** (Page {src.metadata.get('page', 1)}):"
                            )
                            st.info(src.page_content)

                st.caption(
                    f"🤖 Model: {llm_model} | ⏱️ Latency: {result['latency_sec']}s | Chunks Retrieved: {result['retrieved_count']}"
                )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "debug": result,
            }
        )

# TAB 3: DEBUG INSPECTOR
with tab_debug:
    st.subheader("RAG Pipeline & Prompt Debugger")

    if st.session_state.chat_history:
        last_assistant_msgs = [
            m for m in st.session_state.chat_history if m["role"] == "assistant"
        ]
        if last_assistant_msgs and "debug" in last_assistant_msgs[-1]:
            last_debug = last_assistant_msgs[-1]["debug"]

            st.markdown("### Exact Prompt Sent to Local LLM")
            st.code(last_debug["formatted_prompt"], language="markdown")

            st.markdown("### Retrieved Raw Context Objects")
            for idx, doc in enumerate(last_debug["context_sources"]):
                st.write(f"**Chunk {idx+1} Metadata:**", doc.metadata)
                st.text_area(
                    f"Chunk {idx+1} Text",
                    value=doc.page_content,
                    height=100,
                    key=f"debug_chunk_{idx}",
                )
        else:
            st.info("No active query debug data available.")
    else:
        st.info(
            "Run a query in the Chat tab to inspect context generation and prompts."
        )

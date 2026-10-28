

This is a **Local RAG (Retrieval-Augmented Generation) & Document Q&A Studio**. It is a fully self-hosted application designed to let you upload documents (PDFs and text files), index them into a local vector database, and chat with them using local Large Language Models (LLMs) running via **Ollama**.

![Summary](https://github.com/user-attachments/assets/b6695879-e1ec-4f5b-9a8e-f3e05f4fe4ed)

Because everything runs locally—from embeddings and vector storage to text generation—your data never leaves your machine, making it completely private and offline-capable.

---

## Project Structure

```text
local-rag-studio/
├── app.py              # Main Streamlit UI & Chat Logic
├── rag_engine.py       # Chunking, Vector Store, Retrieval & LLM Chain
├── requirements.txt    # Dependencies
└── README.md           # Instructions to run

```

---

## How the Two Files Work Together

The project is cleanly split into two architectural layers:

1. **`app.py` (The Presentation & Control Layer):** Built with **Streamlit**, it provides the interactive web interface, sidebar parameter controls, file upload handlers, chat history state management, and debugging tabs.
2. **`rag_engine.py` (The Core Backend Engine):** Built around **LangChain**, **ChromaDB**, and **HuggingFace**, it handles all heavy-lifting: document loading, text chunking, vector embedding, hybrid retrieval (Dense + BM25), and Ollama LLM prompting.

---

## Deep Dive: File-by-File Breakdown

### 1. `app.py` (Frontend & UI Controller)

* **Session State Management:** Persists your RAG pipeline instance, chat history, and uploaded file lists across UI reruns so you don't lose your session when clicking around.
* **Dynamic Model Discovery:** Automatically pings your local Ollama instance (`http://localhost:11434/api/tags`) to populate the sidebar dropdown with whatever models you currently have installed locally.
* **Interactive Sidebar Controls:** Lets you tune RAG hyperparameters on the fly—such as choosing chunking strategies, adjusting chunk sizes and overlaps, switching between dense vector search and hybrid search, tweaking top-$k$ counts, and clearing the database.
* **Three-Tab Architecture:**
* **Upload & Index Tab:** Accepts PDF/TXT files, writes them temporarily to disk, passes them to the backend engine for processing, and displays ingestion telemetry (chunk counts, timing, and sample previews).
* **Document Q&A Tab:** Renders the chat message interface, captures your prompts, calls the backend engine, and displays the generated answer alongside expandable source citations and performance metrics.
* **RAG Debug Inspector Tab:** Pulls the latest turn's debug payload to let you inspect the **exact prompt string** sent to the LLM and review the raw text chunks and metadata retrieved from the vector store.



### 2. `rag_engine.py` (Backend RAG Pipeline)

* **Initialization (`LocalRAGPipeline`):** Sets up a local HuggingFace embedding model (`all-MiniLM-L6-v2` running on CPU) and initializes a persistent local Chroma vector database (`./chroma_db`).
* **Ingestion & Chunking (`process_and_index_file`):**
* Loads raw files using `PyPDFLoader` or `TextLoader`.
* Splits text using your chosen strategy (**Recursive** for natural boundaries, **Fixed Size** for uniform character cuts, or **Sentence** to keep sentences intact).
* Embeds and stores the chunks in ChromaDB while simultaneously building/updating a keyword-based `BM25Retriever`.


* **Hybrid Retrieval (`retrieve_context`):** Executes either pure Dense Vector Search or a combined **Ensemble Retriever** (Dense Vector + BM25 keyword search) weighted by your alpha parameter to capture both semantic meaning and exact keyword matches.
* **Grounded Generation (`answer_query`):**
* Retrieves the top-$k$ chunks and formats them into a clean context string with metadata (source index and page numbers).
* Constructs a **strict anti-hallucination prompt template** that forces the local Ollama LLM to answer *only* using the provided context (or state it cannot find the answer).
* Invokes the chosen local Ollama model at a low temperature (`0.1`) for factual, deterministic responses and packages generation telemetry for debugging.


* **Database Reset (`clear_database`):** Wipes the active Chroma collection and resets internal document/BM25 state.

---

### 🛠️ Execution Instructions

1. **Install Prerequisites**:
Ensure Python 3.10+ is installed on your local machine.
2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


3. **Install & Start Ollama (Local LLM)**:
* Download Ollama from [ollama.com](https://ollama.com).
* Pull your preferred model:
```bash
ollama pull llama3

```




4. **Run the Streamlit Application**:
```bash
streamlit run app.py

```

import os
import time
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM as Ollama


class LocalRAGPipeline:
    def __init__(
        self,
        collection_name: str = "local_rag_docs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
    ):
        """Initializes Embeddings and Vector Store."""
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.documents: List[Document] = []
        self.bm25_retriever = None

    def process_and_index_file(
        self,
        file_path: str,
        chunk_strategy: str = "Recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """Loads file, performs chunking strategy, and indexes into ChromaDB & BM25."""
        start_time = time.time()

        # 1. Load Document
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()

        # 2. Select Chunking Strategy
        if chunk_strategy == "Recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
        elif chunk_strategy == "Fixed Size":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" "
            )
        elif chunk_strategy == "Sentence":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[". ", "! ", "? ", "\n"],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        # 3. Split Documents
        chunks = splitter.split_documents(raw_docs)
        self.documents.extend(chunks)

        # 4. Update Vector DB (ChromaDB automatically persists in v0.4+)
        self.vector_store.add_documents(chunks)

        # 5. Build/Update BM25 Retriever for Hybrid Search
        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

        indexing_time = round(time.time() - start_time, 3)

        return {
            "total_chunks": len(chunks),
            "total_documents_indexed": len(self.documents),
            "indexing_time_sec": indexing_time,
            "sample_chunk": chunks[0].page_content if chunks else "",
        }

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        search_type: str = "Hybrid (Dense + BM25)",
        alpha: float = 0.5,
    ) -> List[Tuple[Document, float]]:
        """Retrieves documents using Dense Vector Search, BM25, or Hybrid Search."""
        if search_type == "Dense Vector":
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            return results

        elif search_type == "Hybrid (Dense + BM25)":
            dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

            if self.bm25_retriever is None and self.documents:
                self.bm25_retriever = BM25Retriever.from_documents(self.documents)
                self.bm25_retriever.k = top_k

            if self.bm25_retriever:
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, self.bm25_retriever],
                    weights=[alpha, 1.0 - alpha],
                )
                docs = ensemble_retriever.invoke(query)[:top_k]
                return [(doc, 0.0) for doc in docs]
            else:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=top_k
                )
                return results

        else:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            return results

    def answer_query(
        self,
        query: str,
        top_k: int = 4,
        search_type: str = "Hybrid (Dense + BM25)",
        model_name: str = "llama3",
    ) -> Dict[str, Any]:
        """Runs RAG pipeline: Context Retrieval -> Prompt Construction -> LLM Generation."""
        start_time = time.time()

        # 1. Fetch relevant contexts
        retrieved_items = self.retrieve_context(
            query, top_k=top_k, search_type=search_type
        )
        retrieved_docs = [item[0] for item in retrieved_items]

        # 2. Format Context
        context_str = "\n\n---\n\n".join(
            [
                f"[Source {i+1} - Page {doc.metadata.get('page', 1)}]:\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        # 3. Strict Grounded Prompt Construction
        system_prompt = """You are a precise Document Assistant. Answer the question STRICTLY using only the provided context below.
If the answer cannot be found in the context, explicitly state "I cannot find the answer based on the provided document context." Do not extrapolate or hallucinate.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        formatted_prompt = prompt.format(context=context_str, question=query)

        # 4. Generate response via local LLM (Ollama)
        try:
            llm = Ollama(model=model_name, temperature=0.1)
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            response = f"Local LLM Error (Ensure Ollama is running): {str(e)}"

        generation_time = round(time.time() - start_time, 3)

        return {
            "answer": response,
            "context_sources": retrieved_docs,
            "formatted_prompt": formatted_prompt,
            "latency_sec": generation_time,
            "retrieved_count": len(retrieved_docs),
        }

    def clear_database(self):
        """Resets the vector database."""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="local_rag_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.documents = []
        self.bm25_retriever = None

# Multi-PDF-RAG-Chatbot-using-Ollama-and-Llama3.2

A retrieval-augmented QA service that answers questions strictly from uploaded PDFs and TXT files. It loads documents, performs semantic chunking, builds a FAISS vector index using BGE embeddings, re-ranks with a CrossEncoder, and generates answers via an Ollama LLM with a strict grounding prompt.

### Overview
- Accepts PDF and TXT uploads and stores them under uploads/.
- Extracts text, performs semantic chunking, embeds chunks, and builds a FAISS index in memory.
- At query time, retrieves top similar chunks, re-ranks them with a cross-encoder, and answers using an LLM constrained to the provided context.
- Maintains a single in-memory vector store per running process via a global FileProcessor instance.

### How the Pipeline Works
- Loading: Uses PyPDFLoader and TextLoader to read and concatenate all page contents.
- Chunking: Uses SemanticChunker to split text by meaning, not fixed size, tuned via percentile breakpoints.
- Embeddings: Encodes chunks with BAAI/bge-large-en-v1.5 for high-quality semantic search.
- Vector store: Builds a FAISS index from encoded chunks for fast similarity search.
- Re-ranking: Applies a CrossEncoder to the top-k retrieved chunks to improve precision.
- Answering: Runs a LangChain QA chain with a strict prompt and an Ollama LLM to produce grounded answers only from the selected context.

### Code Structure and Responsibilities
- app: FastAPI application that exposes two endpoints: /process_files and /ask_question.
- UPLOAD_FOLDER: Directory for persisted uploads. Created if absent.
- QuestionRequest: Pydantic model defining the request body for questions.
- FileProcessor: Core orchestration class. A single global instance is shared.

### FileProcessor Class
- State
  - vector_store: FAISS index (in-memory).
  - text_chunks: List of chunked strings.
  - embedding_func: HuggingFaceEmbeddings for BGE Large.
  - cross_encoder: CrossEncoder for re-ranking document relevance.

- Methods
  - get_file_text(files)
    - Iterates over file paths.
    - For .pdf, uses PyPDFLoader; for .txt, uses TextLoader (UTF-8).
    - Concatenates page_content into a single string.
    - Returns an error string if unsupported type, empty content, or exceptions.
  - get_text_chunks(text)
    - Builds a SemanticChunker with embeddings=self.embedding_func.
    - Uses breakpoint_threshold_type="percentile" and breakpoint_threshold_amount=90.
    - Persists chunks to self.text_chunks and returns them, or an error string on failure.
  - create_vector_store(text_chunks)
    - Wraps chunks as LangChain Document objects.
    - Creates a FAISS index with the configured embedding function.
    - Sets self.vector_store and returns a status string.
  - get_conversational_chain()
    - Constructs a strict prompt:
      - Answers must be grounded in the provided context.
      - If the answer is not found, responds: “The answer is not in the provided context.”
      - If asked to summarize, return a detailed gist.
    - Initializes Ollama LLM with model="llama3.2", temperature=0.5.
    - Creates a PromptTemplate with variables context and question.
    - Returns a QA chain (chain_type="stuff").
  - user_input(user_question)
    - Validates vector store presence; else returns an error string.
    - Retrieves k=5 similar documents from FAISS.
    - Forms [question, doc] pairs and scores them with the CrossEncoder.
    - Sorts by score descending, selects top 3 documents.
    - Runs the QA chain with input_documents and question.
    - Returns chain output_text or a fallback “not in context” string, or an error string on failure.
  - process_files(files)
    - Calls get_file_text → get_text_chunks → create_vector_store.
    - Returns a success string when ready, or propagates a descriptive error string.

### FastAPI Endpoints (Behavior and Contracts)
- POST /process_files
  - Accepts multiple files (PDF, TXT).
  - Streams each file to uploads/ with a UUID prefix to avoid filename collisions.
  - Calls FileProcessor.process_files with the saved file paths.
  - Returns a JSON message with either success or descriptive error text.
  - Rejects unsupported file types with HTTP 400.

- POST /ask_question
  - Accepts JSON with user_question (string).
  - Calls FileProcessor.user_input to retrieve, re-rank, and answer.
  - Returns a JSON answer string.
  - If no vector store is available yet, returns a descriptive error string.

### Prompt and Grounding Strategy
- The prompt enforces document-grounded answering.
- When the content is unavailable in the selected context, the service explicitly returns “The answer is not in the provided context.”
- When asked to summarize, the chain returns a detailed gist of the processed documents.

### Retrieval and Re-ranking Details
- Initial retrieval: FAISS similarity_search with k=5 candidates.
- Re-ranking: CrossEncoder "cross-encoder/ms-marco-MiniLM-L-12-v2" scores [question, chunk] pairs.
- Final context: Top 3 re-ranked chunks are passed into the QA chain.

### Configuration Knobs
- Embedding model: FileProcessor.embedding_func uses "BAAI/bge-large-en-v1.5". Change model_name to trade accuracy vs. resources.
- Chunking:
  - breakpoint_threshold_amount in get_text_chunks controls aggressiveness; lower values produce smaller chunks.
- Retrieval:
  - k in similarity_search controls candidate breadth.
  - Slice [:3] controls the number of top re-ranked chunks provided to the LLM.
- LLM:
  - In get_conversational_chain, change model or temperature.
- Ollama host:
  - Configure with the OLLAMA_HOST environment variable if Ollama isn’t local/default.

### Error Handling and Messages
- File loading errors:
  - Unsupported file types are rejected early in /process_files (HTTP 400).
  - Empty or unreadable files return “Error:No text found in files.” or a read error string.
- Chunking, indexing, or retrieval issues:
  - Functions return “Error:” prefixed strings that propagate to the endpoint response’s message or answer fields.
- No index built:
  - Asking a question before processing files returns “Error:No vector store found.Please upload and process files first.”

### Dependencies
- FastAPI and Uvicorn for the web API.
- LangChain core plus community and experimental modules.
- sentence-transformers for BGE embeddings and CrossEncoder.
- FAISS for vector search.
- pypdf for PDF parsing.
- numpy for array operations.
- Ollama must be installed and running with the llama3.2 model available.

Example requirements.txt:

```txt
fastapi
uvicorn[standard]
pydantic>=2.0
langchain
langchain-community
langchain-experimental
sentence-transformers
faiss-cpu
pypdf
numpy
```

Notes:
- If you encounter import changes for embeddings, consider installing langchain-huggingface and updating imports accordingly.
- GPU acceleration for FAISS is possible with faiss-gpu.
- Initial model downloads occur on first run and are cached afterward.

### Running the Service
- Ensure Python 3.10+.
- Create a virtual environment and install dependencies.
- Ensure Ollama is installed, running, and the model "llama3.2" is available.
- Start the server with Uvicorn (e.g., uvicorn main:app --reload).
- An uploads/ directory will be created for incoming files.

### State, Concurrency, and Persistence
- The vector store is **in-memory** and **process-local**. It is lost on restart.
- A single global FileProcessor instance means state is shared across all requests and users in that process.
- Consider adding locks if you expect concurrent writes or high parallelism.
- For persistence across restarts, add FAISS save/load to disk.

### Performance Considerations
- BGE Large and CrossEncoder models are sizable and can be memory-intensive.
- SemanticChunker uses embeddings for segmentation; adjust the breakpoint percentile for speed vs. chunk quality.
- Re-ranking improves precision but adds latency; tune k and top-n.
- Use faiss-gpu and/or smaller models if latency or memory is critical.

### Security Notes
- No authentication or authorization is implemented.
- Files are written to disk under uploads/. Sanitize, scan, and/or auto-clean as required for your environment.
- Do not expose this service publicly without proper security hardening.

### For Demo refer:https://www.youtube.com/watch?v=ffW3iIVVqVU

### Project Structure

```txt
.
├─ app.py                 # FastAPI app and RAG pipeline
├─ requirements.txt       # Python dependencies
└─ README.md              # Documentation
```
### Extensibility
- Additional file types: Extend FileProcessor.get_file_text with new LangChain loaders (e.g., DOCX).
- Index persistence: Use FAISS save/load to maintain the vector store across restarts.
- Multi-user isolation: Instantiate per-session FileProcessor objects and map sessions to indexes.
- Model swaps: Change embedding models, cross-encoders, or Ollama LLMs as needed.
- Improved error handling: Replace “Error:” strings with structured exceptions and HTTP error codes.

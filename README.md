# 📚 Semantic Search & Intelligent Q&A System

An AI-powered **Semantic Search and Intelligent Question Answering (Q&A) System** that retrieves relevant information from research papers using **vector embeddings**, **semantic search**, **re-ranking**, and **Retrieval-Augmented Generation (RAG)**.

Unlike traditional keyword-based search, this system understands the **semantic meaning** of user queries and retrieves the most relevant information from a collection of research papers.

---

# 🚀 Features

* 📄 Multi-PDF document ingestion
* 📝 Automatic text extraction and preprocessing
* ✂️ Recursive text chunking
* 🧠 Sentence Transformer embeddings
* 🗂️ ChromaDB vector database
* 🔍 Semantic similarity search
* 📊 Top-K document retrieval
* 🎯 CrossEncoder re-ranking
* 🤖 Intelligent Q&A using Ollama (Llama 3.2)
* 💻 Interactive Streamlit interface
* 📈 Precision@K evaluation

---

# 🏗️ System Architecture

```text
                   Research Papers
                          │
                          ▼
                 PDF Text Extraction
                          │
                          ▼
                    Text Chunking
                          │
                          ▼
           SentenceTransformer Embeddings
                          │
                          ▼
                  ChromaDB Vector Store
                          │
                          ▼
               Semantic Retrieval (Top-K)
                          │
                          ▼
               CrossEncoder Re-ranking
                          │
                          ▼
                 Ollama (Llama 3.2)
                          │
                          ▼
                  Intelligent Answer
                          │
                          ▼
                  Streamlit Interface
```

---

# 📂 Project Structure

```text
SemanticSearchQA/

│── app.py
│── ingest.py
│── preprocess.py
│── embeddings.py
│── vectordb.py
│── retrieval.py
│── reranker.py
│── llm.py
│── evaluate.py
│── config.py
│── requirements.txt
│── README.md

├── data/
│      ├── paper1.pdf
│      ├── paper2.pdf
│      └── ...
│
├── chroma_db/
│
└── assets/
```

---

# ⚙️ Technologies Used

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| UI Framework         | Streamlit                                |
| PDF Processing       | PyPDF                                    |
| Embedding Model      | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database      | ChromaDB                                 |
| Re-ranking Model     | CrossEncoder (ms-marco-MiniLM-L-6-v2)    |
| Large Language Model | Ollama (Llama 3.2)                       |
| Similarity Search    | Vector Search (Cosine Similarity)        |

---

# 🔄 Workflow

1. Load research papers.
2. Extract text from PDFs.
3. Split documents into smaller chunks.
4. Generate semantic embeddings.
5. Store embeddings inside ChromaDB.
6. Convert the user query into an embedding.
7. Retrieve Top-K relevant chunks using semantic similarity.
8. Re-rank retrieved chunks using CrossEncoder.
9. Generate the final answer using Ollama.
10. Display the answer, sources, and retrieved chunks in Streamlit.

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/rohanrg2003/SecninjazAssignment.git

cd SecninjazAssignment
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🤖 Install Ollama

Download Ollama:

https://ollama.com/download

Pull the required model:

```bash
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

---

# ▶️ Run the Application

Generate embeddings (only the first time):

```bash
python ingest.py
python embeddings.py
python vectordb.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

# 💬 Example Queries

* What is Federated Learning?
* Explain Cyberbullying Detection.
* What are the applications of Federated Learning?
* Explain privacy-preserving machine learning.
* How is deep learning used for cyber security?

---

# 📊 Evaluation

The retrieval component can be evaluated using **Precision@K**, which measures the proportion of relevant documents retrieved within the top K search results.

---

# 📸 Sample Output

The application provides:

* AI-generated answers
* Top retrieved research papers
* Source page numbers
* Re-ranking scores
* Retrieved document chunks
* Response time

---
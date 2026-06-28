import time
import streamlit as st

from retrieval import SemanticRetriever
from reranker import ReRanker
from llm import LLMGenerator

st.set_page_config(
    page_title="Semantic Search & Intelligent Q&A",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return SemanticRetriever(), ReRanker(), LLMGenerator()

retriever, reranker, llm = load_pipeline()

# Sidebar
with st.sidebar:
    st.title("📊 Project Info")
    st.success("System Ready")
    st.write("📄 PDFs Loaded: 21")
    st.write("📃 Pages: 452")
    st.write("🧩 Chunks: 4254")
    st.write("🧠 Embedding: all-MiniLM-L6-v2")
    st.write("🗂️ Vector DB: ChromaDB")
    st.write("🤖 LLM: Llama 3.2")
    st.write("📈 Re-ranker: CrossEncoder")

st.title("📚 Semantic Search & Intelligent Q&A System")
st.caption("Semantic Search • ChromaDB • Sentence Transformers • Ollama")

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is Federated Learning?"
)

if st.button("🚀 Generate Answer", use_container_width=True):

    if question.strip():

        with st.spinner("Searching research papers..."):

            start = time.time()

            retrieval = retriever.search(question, top_k=10)

            ranked = reranker.rerank(question, retrieval)

            answer = llm.generate_answer(question, ranked)

            end = time.time()

        st.success("Answer Generated")

        st.subheader("🤖 AI Answer")
        st.write(answer)

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("📚 Sources")

            for item in ranked[:3]:
                st.info(
                    f"{item['metadata']['source']}\n\nPage: {item['metadata']['page']}"
                )

        with col2:

            st.subheader("📈 Re-ranking Scores")

            for i, item in enumerate(ranked[:3]):
                st.metric(
                    f"Chunk {i+1}",
                    f"{item['score']:.2f}"
                )

        st.divider()

        st.subheader("📄 Retrieved Chunks")

        for i, item in enumerate(ranked[:3]):

            with st.expander(
                f"Chunk {i+1} • Score {item['score']:.2f}"
            ):
                st.write(item["text"])

        st.success(
            f"⏱ Response Time: {end-start:.2f} seconds"
        )
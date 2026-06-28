import chromadb
from sentence_transformers import SentenceTransformer


class SemanticRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path="chroma_db")

        self.collection = self.client.get_collection(
            "research_papers"
        )

    def search(self, query, top_k=5):

        print("\nGenerating query embedding...")

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        results = self.collection.query(

            query_embeddings=[query_embedding.tolist()],

            n_results=top_k

        )

        return results
if __name__ == "__main__":

    retriever = SemanticRetriever()

    question = input("\nAsk a question:\n\n> ")

    results = retriever.search(question)

    print("\nTop Results\n")

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(docs)):

        print("=" * 70)

        print(f"Rank : {i+1}")

        print(f"Similarity Distance : {distances[i]}")

        print(f"Source : {metas[i]['source']}")

        print(f"Page : {metas[i]['page']}")

        print()

        print(docs[i][:500])

        print()
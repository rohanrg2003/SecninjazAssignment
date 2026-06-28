from sentence_transformers import CrossEncoder


class ReRanker:

    def __init__(self):

        print("Loading Cross Encoder...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, retrieval_results):

        docs = retrieval_results["documents"][0]
        metas = retrieval_results["metadatas"][0]

        pairs = []

        for doc in docs:
            pairs.append((query, doc))

        scores = self.model.predict(pairs)

        ranked = []

        for score, doc, meta in zip(scores, docs, metas):

            ranked.append({

                "score": float(score),

                "text": doc,

                "metadata": meta

            })

        ranked = sorted(
            ranked,
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked
from retrieval import SemanticRetriever

if __name__ == "__main__":

    retriever = SemanticRetriever()

    reranker = ReRanker()

    question = input("Question: ")

    results = retriever.search(question, top_k=10)

    ranked = reranker.rerank(question, results)

    print("\nTop Re-ranked Results\n")

    for i, item in enumerate(ranked[:5]):

        print("=" * 70)

        print(f"Rank {i+1}")

        print(f"Score : {item['score']:.4f}")

        print(item["metadata"])

        print()

        print(item["text"][:400])
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def generate_embeddings(self, chunked_documents):

        embeddings = []

        print("\nGenerating embeddings...\n")

        for doc in tqdm(chunked_documents):

            vector = self.model.encode(
                doc["text"],
                convert_to_numpy=True
            )

            embeddings.append(
                {
                    "embedding": vector,
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
            )

        return embeddings
from ingest import PDFIngestor
from preprocess import TextPreprocessor

if __name__ == "__main__":

    ingestor = PDFIngestor()
    docs = ingestor.load_pdfs()

    preprocessor = TextPreprocessor()
    chunks = preprocessor.create_chunks(docs)

    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(chunks)

    print("\nTotal Embeddings:", len(embeddings))

    print("\nEmbedding Dimension:")

    print(len(embeddings[0]["embedding"]))
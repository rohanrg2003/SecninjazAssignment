import chromadb
from chromadb.config import Settings


class VectorDatabase:

    def __init__(self):

        self.client = chromadb.PersistentClient(path="chroma_db")

        self.collection = self.client.get_or_create_collection(
            name="research_papers"
        )

    def add_documents(self, embeddings):

        print("\nStoring embeddings in ChromaDB...\n")

        for idx, item in enumerate(embeddings):

            self.collection.add(

                ids=[str(idx)],

                embeddings=[item["embedding"].tolist()],

                documents=[item["text"]],

                metadatas=[item["metadata"]]

            )

        print(f"\nStored {len(embeddings)} embeddings successfully!")

    def get_collection(self):

        return self.collection
from ingest import PDFIngestor
from preprocess import TextPreprocessor
from embeddings import EmbeddingGenerator

if __name__ == "__main__":

    ingestor = PDFIngestor()

    docs = ingestor.load_pdfs()

    preprocessor = TextPreprocessor()

    chunks = preprocessor.create_chunks(docs)

    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(chunks)

    db = VectorDatabase()

    db.add_documents(embeddings)
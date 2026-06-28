from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextPreprocessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def create_chunks(self, documents):
        chunked_documents = []

        for doc in documents:
            chunks = self.splitter.split_text(doc["text"])

            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "page": doc["metadata"]["page"],
                        "chunk": i + 1
                    }
                })

        return chunked_documents
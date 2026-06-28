from pathlib import Path
from pypdf import PdfReader


class PDFIngestor:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)

    def load_pdfs(self):
        documents = []

        pdf_files = list(self.data_path.glob("*.pdf"))

        if not pdf_files:
            print("❌ No PDF files found inside data folder.")
            return documents

        print(f"\n📚 Found {len(pdf_files)} PDF(s).\n")

        total_pages = 0

        for pdf in pdf_files:

            print(f"📄 Reading {pdf.name}")

            try:
                reader = PdfReader(pdf)

                for page_num, page in enumerate(reader.pages, start=1):

                    text = page.extract_text()

                    if text and text.strip():

                        documents.append(
                            {
                                "text": text,
                                "metadata": {
                                    "source": pdf.name,
                                    "page": page_num,
                                },
                            }
                        )

                        total_pages += 1

            except Exception as e:
                print(f"❌ Error reading {pdf.name}: {e}")

        print("\n==============================")
        print(f"✅ PDFs Loaded : {len(pdf_files)}")
        print(f"✅ Pages Read  : {total_pages}")
        print("==============================\n")

        return documents


from preprocess import TextPreprocessor

if __name__ == "__main__":

    ingestor = PDFIngestor()
    docs = ingestor.load_pdfs()

    preprocessor = TextPreprocessor()

    chunks = preprocessor.create_chunks(docs)

    print(f"\nTotal Pages Extracted : {len(docs)}")
    print(f"Total Chunks Created : {len(chunks)}")

    print("\nSample Chunk Metadata")
    print(chunks[0]["metadata"])

    print("\nSample Chunk\n")
    print(chunks[0]["text"])
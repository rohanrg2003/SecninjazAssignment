import ollama


class LLMGenerator:

    def __init__(self, model="llama3.2"):
        self.model = model

    def generate_answer(self, question, ranked_chunks):

        context = "\n\n".join(
            [chunk["text"] for chunk in ranked_chunks[:3]]
        )

        prompt = f"""
You are an AI Research Assistant.

Use ONLY the information present in the provided context.

Do NOT use your own knowledge.

If the context does not contain the answer, reply exactly:

"I couldn't find the answer in the provided research papers."

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
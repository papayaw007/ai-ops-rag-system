import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_anthropic import ChatAnthropic

# Load environment variables (.env)
load_dotenv()


class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        return len(chunks)

    def ask(self, question):
        if not self.vectorstore:
            return "No documents loaded."

        # 1. Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # 2. Load Claude with API key from env
        llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0
        )

        # 3. Prompt
        prompt = f"""
You are an enterprise AI assistant.

Use ONLY the context below to answer the question.

If the answer is not in the context, say: "Not found in document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # 4. Call model
        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [doc.page_content[:300] for doc in docs]
        }
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import parse_faq_text

load_dotenv()  # Load environment variables

embeddings = OpenAIEmbeddings()  # API key from environment

def get_vectorstore(documents):
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

if __name__ == "__main__":
    docs = parse_faq_text()
    vectorstore = get_vectorstore(docs)

    query = "What is the Anzara loan app?"
    results = vectorstore.similarity_search(query, k=2)

    for doc in results:
        # Remove the metadata access since it might not exist
        print("Found document:")
        print(doc.page_content[:100] + "...\n")
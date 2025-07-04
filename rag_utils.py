from langchain.schema import Document
from faq_data import faq_data

def preparate_documents():
    docs= []
    for item in faq_data.values():
        question = item['question']
        answer = item['answer']
        docs.append(Document(page_content = answer, metadata={"question": question}))
    return docs    

if __name__ == "__main__":
    docs = preparate_documents()
    for doc in docs:
        print(doc.page_content)
        print(doc.page_content[:100]+ "...\n")
from langchain.schema import Document
from faq_data import faq_text

def parse_faq_text():
    blocks = faq_text.split("\n\n")
    docs= []
    for block in blocks:
        if block.strip():
            docs.append(Document(page_content=block.strip()))
    return docs    

if __name__ == "__main__":
    docs = parse_faq_text()
    for doc in docs:
        print(doc.page_content)
        print(doc.page_content[:100]+ "...\n")
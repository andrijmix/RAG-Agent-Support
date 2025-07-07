import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from rag_utils import parse_faq_text
from vectorstore_utils import get_vectorstore
from langchain.callbacks import LangChainTracer

def get_qa_agent():
    docs = parse_faq_text()
    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tracer = LangChainTracer()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        callbacks=[tracer]
    )
    
    return chain

if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "true" 
    os.environ["LANGCHAIN_PROJECT"] = "anzara-faq"
    chain = get_qa_agent()
    
    while True:
        query = input("Enter your question: ")
        if query.lower() in ["exit", "quit", "bye"]:
            break

        try:
            result = chain.invoke(query)
            print("Answer: ", result["result"])
        except Exception as e:
            print(f"Error: {str(e)}")
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()  # Make sure this is called before using OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
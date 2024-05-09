from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

import os

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV=os.environ.get("PINECONE_API_ENV")

#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "medical-chatbot" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
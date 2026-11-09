from dotenv import load_dotenv
import os

load_dotenv()
print("Key loaded:", os.getenv("GOOGLE_API_KEY") is not None)

from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(len(emb.embed_query("hello world")), "dimensions")
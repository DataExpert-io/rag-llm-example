import os
from openai import OpenAI
from pinecone import Pinecone
import hashlib


# --- Configure your API keys ---
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# --- Initialize Pinecone ---
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)
INDEX_NAME = 'rag-example-index'  # Replace with your Pinecone index name
# Assuming the index has already been created in Pinecone
index = pinecone_client.Index(INDEX_NAME)

def upsert_to_pinecone(text, metadata, id=None):
    response = client.embeddings.create(input=text, model='text-embedding-ada-002')
    embedding = response.data[0].embedding
    # if the id is non, use the MD5 hash of the content
    if id is None:
        # Encode the text to bytes, then create the MD5 hash
        hash_object = hashlib.md5(text.encode())
        # Get the hexadecimal representation of the digest
        id = hash_object.hexdigest()
    index.upsert(
        vectors=[
            {
                "id": id,
                "values": embedding,
                "metadata": metadata
            }
        ]
    )
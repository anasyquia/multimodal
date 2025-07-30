import os
import json
import getpass
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore():
    # Get API keys
    print("Please enter your API keys:")
    openai_key = getpass.getpass("OpenAI API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_key

    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load scraped data
    print("Loading scraped data...")
    with open("scraped_ms_ads_data_v3.json", "r") as f:
        scraped_data = json.load(f)

    # Convert to documents
    print("Converting to documents...")
    documents = []
    for item in scraped_data:
        doc = Document(
            page_content=item["content"],
            metadata={
                "source": item["url"],
                "title": item["title"],
                "description": item.get("description")
            }
        )
        documents.append(doc)

    print(f"Created {len(documents)} documents")

    # Create FAISS store
    print("Creating FAISS vector store...")
    faiss_store = FAISS.from_documents(documents, embeddings)

    # Save vector store
    print("Saving vector store...")
    faiss_store.save_local("vectorstore")

    print("✅ Vector store created and saved successfully!")

if __name__ == "__main__":
    create_vectorstore() 
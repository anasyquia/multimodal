import os
import pickle
import getpass
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_test_vectorstore():
    # Get API keys
    print("Please enter your API keys:")
    openai_key = getpass.getpass("OpenAI API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_key

    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create test documents
    print("Creating test documents...")
    documents = [
        Document(
            page_content="The MS in Applied Data Science program at UChicago offers both in-person and online formats.",
            metadata={"source": "test", "title": "Program Overview"}
        ),
        Document(
            page_content="Tuition for the program is $6,384 per course, totaling $76,608 for the entire program.",
            metadata={"source": "test", "title": "Tuition Information"}
        ),
        Document(
            page_content="The program requires completion of 12 courses: 6 core courses, 4 electives, and 2 capstone courses.",
            metadata={"source": "test", "title": "Curriculum"}
        )
    ]

    print(f"Created {len(documents)} test documents")

    # Create FAISS store
    print("Creating FAISS vector store...")
    faiss_store = FAISS.from_documents(documents, embeddings)

    # Save to pickle file
    print("Saving vector store...")
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(faiss_store, f)

    print("✅ Test vector store created and saved successfully!")

if __name__ == "__main__":
    create_test_vectorstore() 
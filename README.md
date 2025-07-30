# UChicago MSADS Program Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) system that helps answer questions about the University of Chicago's MS in Applied Data Science program.

## Features

- 🔍 Semantic search across program materials
- ✨ Response validation and quality checks
- 📊 Source attribution and relevance scoring
- 🎯 Accurate and grounded responses
- 🚀 Real-time performance metrics

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
2. Type your question about the UChicago MSADS program in the text input
3. View the response, along with:
   - Validation metrics
   - Source documents
   - Performance statistics

## System Components

- **RAG System**: Core question-answering system using LangChain and ChromaDB
- **Response Validator**: Ensures high-quality, factual responses
- **Document Reranker**: Improves relevance of retrieved documents using Cohere
- **Streamlit Interface**: User-friendly web interface

## Notes

- The system requires both OpenAI and Cohere API keys to function
- The vector store (ChromaDB) contains pre-processed program materials
- All responses are grounded in official program documentation
- The system includes comprehensive validation to prevent hallucinations

## Contributing

Feel free to submit issues and enhancement requests! 
import os
import time
import json
import warnings
import streamlit as st
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import cohere

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
RETRIEVAL_K = 5

class DocumentReranker:
    """Enhanced reranker with retry logic and fallback"""
    
    def __init__(self, model_name: str = "rerank-english-v3.0"):
        self.model_name = model_name
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        self.reranker = cohere.Client(api_key=api_key)
        self.max_retries = 2
        self.retry_delay = 10  # seconds
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank with retry logic and better error handling"""
        if not documents:
            return documents
        
        for attempt in range(self.max_retries + 1):
            try:
                docs_for_rerank = [{"text": doc.page_content} for doc in documents]
                
                results = self.reranker.rerank(
                    model=self.model_name,
                    query=query,
                    documents=docs_for_rerank,
                    top_n=min(top_k, len(documents)),  # Don't exceed available docs
                    return_documents=True
                )
                
                reranked_docs = []
                for result in results.results:
                    original_doc = documents[result.index]
                    original_doc.metadata["rerank_score"] = round(result.relevance_score, 3)
                    reranked_docs.append(original_doc)
                
                return reranked_docs
                
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    return documents[:top_k]

class ResponseValidator:
    """Enhanced response validation with multiple checks"""
    
    def __init__(self):
        self.required_prefixes = [
            "Based on the program materials",
            "According to the program materials",
            "The program materials indicate",
            "The program materials state"
        ]
        
        self.hallucination_phrases = [
            "typically",
            "usually",
            "generally",
            "often",
            "in most cases",
            "commonly",
            "traditionally",
            "tends to",
            "approximately",
            "around"
        ]
        
        self.uncertainty_phrases = [
            "might",
            "may",
            "could",
            "possibly",
            "perhaps",
            "probably",
            "likely",
            "seems",
            "appears"
        ]
    
    def validate_source_attribution(self, response: str) -> Dict[str, Any]:
        """Check if response properly attributes sources"""
        has_attribution = any(
            response.lower().startswith(prefix.lower())
            for prefix in self.required_prefixes
        )
        
        return {
            "has_attribution": has_attribution,
            "attribution_score": 1.0 if has_attribution else 0.0
        }
    
    def check_hallucination_risk(self, response: str) -> Dict[str, Any]:
        """Check for phrases that might indicate hallucination"""
        found_phrases = [
            phrase for phrase in self.hallucination_phrases
            if phrase in response.lower()
        ]
        
        risk_score = len(found_phrases) / len(self.hallucination_phrases)
        
        return {
            "hallucination_risk": risk_score,
            "risky_phrases": found_phrases,
            "is_safe": risk_score < 0.2
        }
    
    def validate_uncertainty(self, response: str) -> Dict[str, Any]:
        """Check for uncertain language"""
        found_phrases = [
            phrase for phrase in self.uncertainty_phrases
            if phrase in response.lower()
        ]
        
        uncertainty_score = len(found_phrases) / len(self.uncertainty_phrases)
        
        return {
            "uncertainty_score": uncertainty_score,
            "uncertain_phrases": found_phrases,
            "is_confident": uncertainty_score < 0.2
        }
    
    def check_answer_length(self, response: str) -> Dict[str, Any]:
        """Validate answer length"""
        words = response.split()
        word_count = len(words)
        
        return {
            "word_count": word_count,
            "is_appropriate_length": 10 <= word_count <= 150,
            "length_score": 1.0 if 10 <= word_count <= 150 else 0.5
        }
    
    def verify_context_usage(self, response: str, context: str) -> Dict[str, Any]:
        """Verify that response uses information from context"""
        # Extract key phrases (3+ word sequences) from response
        response_words = response.lower().split()
        response_phrases = [
            ' '.join(response_words[i:i+3])
            for i in range(len(response_words)-2)
        ]
        
        # Check if phrases appear in context
        found_phrases = [
            phrase for phrase in response_phrases
            if phrase in context.lower()
        ]
        
        context_score = len(found_phrases) / len(response_phrases) if response_phrases else 0
        
        return {
            "context_usage_score": context_score,
            "found_phrases": found_phrases[:5],  # Show top 5 matches
            "is_grounded": context_score > 0.2
        }
    
    def validate_response(self, response: str, context: str = None) -> Dict[str, Any]:
        """Run all validation checks on a response"""
        try:
            # Input validation
            if not response or not isinstance(response, str):
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "attribution": {"has_attribution": False, "attribution_score": 0.0},
                    "hallucination_risk": {"hallucination_risk": 1.0, "risky_phrases": [], "is_safe": False},
                    "uncertainty": {"uncertainty_score": 1.0, "uncertain_phrases": [], "is_confident": False},
                    "length": {"word_count": 0, "is_appropriate_length": False, "length_score": 0.0},
                    "context_usage": {"context_usage_score": 0.0, "found_phrases": [], "is_grounded": False},
                    "error": "Invalid response input"
                }
            
            # Run all checks
            attribution = self.validate_source_attribution(response)
            hallucination = self.check_hallucination_risk(response)
            uncertainty = self.validate_uncertainty(response)
            length = self.check_answer_length(response)
            
            # Context usage check if context provided
            context_usage = (
                self.verify_context_usage(response, context)
                if context else {"context_usage_score": 1.0, "found_phrases": [], "is_grounded": True}
            )
            
            # Calculate overall quality score
            quality_score = np.mean([
                attribution["attribution_score"],
                1 - hallucination["hallucination_risk"],
                1 - uncertainty["uncertainty_score"],
                length["length_score"],
                context_usage["context_usage_score"]
            ])
            
            # Determine if response is valid
            is_valid = all([
                attribution["has_attribution"],
                hallucination["is_safe"],
                uncertainty["is_confident"],
                length["is_appropriate_length"],
                context_usage["is_grounded"]
            ])
            
            return {
                "is_valid": is_valid,
                "quality_score": quality_score,
                "attribution": attribution,
                "hallucination_risk": hallucination,
                "uncertainty": uncertainty,
                "length": length,
                "context_usage": context_usage
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "attribution": {"has_attribution": False, "attribution_score": 0.0},
                "hallucination_risk": {"hallucination_risk": 1.0, "risky_phrases": [], "is_safe": False},
                "uncertainty": {"uncertainty_score": 1.0, "uncertain_phrases": [], "is_confident": False},
                "length": {"word_count": 0, "is_appropriate_length": False, "length_score": 0.0},
                "context_usage": {"context_usage_score": 0.0, "found_phrases": [], "is_grounded": False},
                "error": f"Validation failed: {str(e)}"
            }

class RAGSystem:
    def __init__(self):
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.reranker = DocumentReranker()
        self.validator = ResponseValidator()
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=TEMPERATURE
        )
        
        # Create QA chain with enhanced prompt
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a precise information system for the University of Chicago's MS in Applied Data Science program.

CORE REQUIREMENTS:
1. ALWAYS start with "Based on the program materials..."
2. Include specific details from the context (dates, costs, contact info, URLs) as available
3. Be specific about program types (Online vs In-Person) when relevant
4. Use exact quotes and numbers from the context
5. If information seems incomplete, state what you found and note limitations

RESPONSE RULES:
- NO speculation beyond provided context
- NO approximations unless explicitly quoted
- NO hedging language (might, maybe, probably) unless in quotes
- If asked about visa sponsorship, be explicit about which programs are eligible
- If asked about appointments/advising, mention specific contact methods available

Context: {context}

Question: {question}

Complete and accurate answer:"""
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
    
    def format_sources(self, source_docs: List[Document]) -> str:
        """Format source documents with metadata"""
        if not source_docs:
            return "No sources available"
        
        sources = []
        for i, doc in enumerate(source_docs, 1):
            title = doc.metadata.get('title', 'Unknown Source')
            url = doc.metadata.get('source', 'No URL')
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            
            content_preview = (
                doc.page_content[:150] + "..." 
                if len(doc.page_content) > 150 
                else doc.page_content
            )
            
            sources.append(f"[{i}] {title}")
            sources.append(f"    Relevance Score: {rerank_score}")
            sources.append(f"    Source: {url}")
            sources.append(f"    Preview: {content_preview}\n")
        
        return "\n".join(sources)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question with comprehensive error handling and monitoring"""
        start_time = time.time()
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract components
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Prepare context for validation
            contexts = [doc.page_content for doc in source_docs]
            combined_context = " ".join(contexts)
            
            # Validate response
            validation = self.validator.validate_response(answer, combined_context)
            
            # Format sources
            sources_summary = self.format_sources(source_docs)
            
            response_time = time.time() - start_time
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources_summary,
                "validation": validation,
                "source_docs": source_docs,
                "success": True,
                "response_time": response_time
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error processing your question. Please try again or contact support if the issue persists.",
                "sources": None,
                "validation": None,
                "source_docs": None,
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time
            }

def main():
    st.set_page_config(
        page_title="UChicago MSADS Program Assistant",
        page_icon="🎓",
        layout="wide"
    )
    
    # Title and description
    st.title("🎓 UChicago MS in Applied Data Science Program Assistant")
    
    # Evaluation notice
    st.info("""
    ⏳ **Evaluation Version**
    
    This is a temporary deployment for DSI staff evaluation. The system uses the latest program materials 
    to provide accurate information about the MS in Applied Data Science program.
    """)
    
    # API Key Management
    with st.sidebar:
        st.header("API Key Configuration")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        cohere_key = st.text_input("Cohere API Key", type="password", key="cohere_key")
        
        if not openai_key or not cohere_key:
            st.warning("⚠️ Please enter both API keys to use the assistant.")
            return
        
        # Set API keys
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["COHERE_API_KEY"] = cohere_key
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner("🔄 Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem()
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            return
    
    st.markdown("""
    Welcome to the UChicago MSADS Program Assistant! I can help answer your questions about:
    - Program requirements and curriculum
    - Application process and deadlines
    - Tuition and financial aid
    - Program formats (Online vs In-Person)
    - And more!
    """)
    
    # User input
    user_question = st.text_input("💭 What would you like to know about the program?")
    
    if user_question:
        with st.spinner("🤔 Searching for information..."):
            result = st.session_state.rag_system.answer_question(user_question)
        
        # Display answer
        st.markdown("### 📝 Answer")
        st.markdown(result["answer"])
        
        # Display sources in expander
        with st.expander("📚 Sources"):
            st.markdown(result["sources"])

if __name__ == "__main__":
    main() 
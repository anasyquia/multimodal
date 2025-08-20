import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import tempfile
from typing import List, Tuple
import matplotlib.pyplot as plt

# Import our RAG backend and config
from rag_backend import MultimodalRAG
from config import Config

# Setup configuration
app_config = Config.get_app_config()
st.set_page_config(**app_config)

# Custom CSS
st.markdown(Config.get_css_styles(), unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def load_rag_system():
    """Load and initialize the RAG system"""
    try:
        with st.spinner("Loading RAG system... This may take a few minutes on first run."):
            rag = MultimodalRAG()
            return rag
    except Exception as e:
        st.error(f"Failed to load RAG system: {str(e)}")
        return None

def display_product_sources(sources: List[str]):
    """Display retrieved product sources in a nice format"""
    st.markdown('<div class="source-section">', unsafe_allow_html=True)
    st.subheader("📦 Retrieved Products")
    
    # Use full width container for products
    with st.container():
        for i, source in enumerate(sources[:3], 1):
            with st.expander(f"Product {i}", expanded=True):
                # Split the source into components for better formatting
                lines = source.split('\n')
                
                # Look for image URL and display it first
                image_url = None
                for line in lines:
                    if line.strip().startswith('Image: http'):
                        image_url = line.replace('Image: ', '').strip()
                        break
                
                # Display image if found
                if image_url:
                    try:
                        st.image(image_url, width=200, caption=f"Product {i}")
                    except Exception as e:
                        st.write(f"Image: {image_url}")
                
                # Parse and display product information cleanly
                product_title = None
                price = None
                description_parts = []
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('Image: http'):
                        continue
                    
                    # Extract product title (first meaningful line)
                    if product_title is None and line and not line.startswith(('Brand:', 'Price:', 'About:', '(Brand:')):
                        # Clean up title
                        clean_title = line.replace('|', '').replace('- -', '').strip()
                        if len(clean_title) > 5 and not clean_title.startswith(('-', '.')):
                            product_title = clean_title
                        continue
                    
                    # Extract price from format: "(Brand: X, Price: Y)" - ignore brand
                    if 'Price:' in line:
                        price_match = line.split('Price:')
                        if len(price_match) > 1:
                            price_part = price_match[1].replace(')', '').strip()
                            if price_part and '$' in price_part:
                                price = price_part
                        continue
                    
                    # Extract description - everything else that's meaningful
                    if line.startswith('About:'):
                        desc_text = line.replace('About:', '').strip()
                        if desc_text:
                            description_parts.append(desc_text)
                    elif product_title and not line.startswith(('Brand:', 'Price:', '(Brand:')):
                        # Additional description lines
                        clean_line = line.replace('|', ' ').replace('- -', '').strip()
                        if len(clean_line) > 3 and not clean_line.startswith(('-', '.', ':')):
                            description_parts.append(clean_line)
                
                # Display formatted information
                if product_title:
                    st.markdown(f"**{product_title}**")
                
                # Display brand and price if available
                col1, col2 = st.columns(2)
                with col1:
                    # Try to extract brand from the raw data
                    brand = None
                    for line in lines:
                        if '(Brand:' in line and 'Price:' in line:
                            parts = line.replace('(', '').replace(')', '').split(', ')
                            for part in parts:
                                if part.startswith('Brand:'):
                                    brand_value = part.replace('Brand:', '').strip()
                                    if brand_value and brand_value != 'nan' and brand_value != '':
                                        brand = brand_value
                                    break
                            break
                    
                    if brand:
                        st.markdown(f"**Brand:** {brand}")
                
                with col2:
                    if price:
                        st.markdown(f"**Price:** {price}")
                
                # Display clean description
                if description_parts:
                    st.markdown("**Description:**")
                    
                    # Join and clean all description text
                    full_desc = ' '.join(description_parts)
                    
                    # Remove boilerplate and clean up
                    boilerplate_phrases = [
                        "Make sure this fits by entering your model number",
                        "make sure this fits by entering your model number", 
                        "This fits your",
                        "this fits your"
                    ]
                    
                    for phrase in boilerplate_phrases:
                        full_desc = full_desc.replace(phrase, '')
                    
                    # Clean up punctuation and spacing issues
                    full_desc = full_desc.replace('||', '|').replace('|', ' | ')
                    full_desc = full_desc.replace('- -', ' ').replace('--', ' ')
                    full_desc = full_desc.replace('. .', '.').replace('..', '.')
                    
                    # Split into meaningful sentences
                    sentences = []
                    for sep in [' | ', '. ', '! ', '? ']:
                        if sep in full_desc:
                            sentences.extend([s.strip() for s in full_desc.split(sep)])
                            break
                    
                    if not sentences:
                        sentences = [full_desc.strip()]
                    
                    # Display only meaningful sentences
                    count = 0
                    for sentence in sentences:
                        sentence = sentence.strip()
                        # Skip meaningless fragments
                        if (len(sentence) > 15 and 
                            not sentence.startswith(('-', ':', '.', ' ')) and 
                            sentence.count(' ') > 2 and
                            not all(c in '- .:' for c in sentence[:5])):
                            st.markdown(f"• {sentence}")
                            count += 1
                            if count >= 3:  # Limit to 3 meaningful points
                                break
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🛍️ Multimodal Amazon Product RAG</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # CSV upload
        # API Key Input
        st.subheader("🔑 API Configuration")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key (Required for OpenAI models)",
            type="password",
            help="Enter your OpenAI API key to use GPT models. Get one at https://platform.openai.com/api-keys",
            placeholder="sk-..."
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        
        
        # Fixed model settings (simplified)
        st.subheader("🤖 System Configuration")
        st.info("**CLIP Model**: clip-ViT-B-32 (for multimodal embeddings)")
        st.info("**LLM Model**: openai/gpt-4o-mini (for answer generation)")
        
        # Use fixed models
        clip_model = "clip-ViT-B-32"
        llm_model = "openai/gpt-4o-mini"
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        # Load system button
        if st.button("🚀 Initialize RAG System", type="primary"):
            try:
                # Check if OpenAI model is selected but no key provided
                if llm_model.startswith("openai/") and not openai_key:
                    st.error("❌ OpenAI API key required for OpenAI models. Please enter your API key above.")
                    return
                
                # Initialize RAG system (will load from artifacts if available)
                rag = MultimodalRAG(
                    csv_path=None,  # Let it load from artifacts first
                    clip_model=clip_model,
                    llm_model=llm_model,
                    top_k=top_k
                )
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
                st.success("✅ RAG system loaded successfully from pre-built artifacts!")
            except Exception as e:
                # If artifacts don't exist, try with CSV as fallback
                csv_path = "real_amazon_data.csv"
                if os.path.exists(csv_path):
                    try:
                        st.info("📦 Artifacts not found, building from CSV data...")
                        rag = MultimodalRAG(
                            csv_path=csv_path,
                            clip_model=clip_model,
                            llm_model=llm_model,
                            top_k=top_k
                        )
                        st.session_state.rag_system = rag
                        st.session_state.data_loaded = True
                        st.success("✅ RAG system loaded successfully!")
                    except Exception as e2:
                        st.error(f"❌ Error loading system: {str(e2)}")
                else:
                    st.error(f"❌ Error: {str(e)}\n\nPlease run `python setup_data.py` first to build the artifacts.")

    # Main content area with wider container
    with st.container():
        if not st.session_state.data_loaded:
            st.info("👈 Please configure and initialize the RAG system in the sidebar to get started.")
            
            # Display example interface
            st.subheader("📝 Example: Text-based Query")
            st.text_input("Ask about products...", placeholder="e.g., What are the features of the Lego Minecraft Creeper?", disabled=True)
            
            st.subheader("🖼️ Example: Image-based Query") 
            st.file_uploader("Upload product image", type=['jpg', 'jpeg', 'png'], disabled=True)
            st.text_input("Ask about the image...", placeholder="e.g., Can you identify this product?", disabled=True)
            
            return
    
    # Main interface when system is loaded
    rag = st.session_state.rag_system
    
    # Create tabs for different query types
    tab1, tab2, tab3 = st.tabs(["💬 Text Query", "🖼️ Image Query", "📊 System Info"])
    
    with tab1:
        st.subheader("Ask about products using text")
        
        # Text input
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the features of the Lego Minecraft Creeper?",
            key="text_query"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_clicked = st.button("🔍 Search", key="text_search", type="primary")
        
        with col2:
            if st.button("🗑️ Clear", key="clear_text"):
                st.rerun()
        
        # Display results outside of columns for full width
        if search_clicked:
            if user_question:
                with st.spinner("Searching products..."):
                    try:
                        answer, sources = rag.answer_text_question(user_question)
                        
                        # Display answer in full width
                        st.markdown("### 🤖 Answer")
                        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                        with st.container():
                            st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        display_product_sources(sources)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'type': 'text',
                            'query': user_question,
                            'answer': answer,
                            'sources': sources
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.subheader("Ask about products using images")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=['jpg', 'jpeg', 'png'],
            key="image_upload"
        )
        
        if uploaded_image:
            # Display uploaded image
            image = Image.open(uploaded_image)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Question about image
        image_question = st.text_input(
            "Question about the image:",
            placeholder="e.g., Can you identify this product and describe its usage?",
            key="image_query"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_clicked = st.button("🔍 Analyze", key="image_search", type="primary")
        
        with col2:
            if st.button("🗑️ Clear", key="clear_image"):
                st.rerun()
        
        # Display results outside of columns for full width
        if analyze_clicked:
            if uploaded_image and image_question:
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    img_path = tmp_file.name
                
                with st.spinner("Analyzing image and searching..."):
                    try:
                        answer, sources = rag.answer_image_question(img_path, image_question)
                        
                        # Display answer in full width
                        st.markdown("### 🤖 Answer")
                        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                        with st.container():
                            st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        display_product_sources(sources)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'type': 'image',
                            'query': image_question,
                            'answer': answer,
                            'sources': sources,
                            'image': image
                        })
                        
                        # Clean up temp file
                        os.unlink(img_path)
                        
                    except Exception as e:
                        st.error(f"Error processing image query: {str(e)}")
                        os.unlink(img_path)
            elif not uploaded_image:
                st.warning("Please upload an image.")
            elif not image_question:
                st.warning("Please enter a question about the image.")
    
    with tab3:
        st.subheader("System Information")
        
        if rag:
            # Dataset info
            st.markdown("### 📊 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Products", len(rag.df))
            with col2:
                st.metric("Embedding Dimension", rag.emb_dim)
            with col3:
                st.metric("CLIP Model", rag.clip_model_name)
            
            # Model info
            st.markdown("### 🤖 Model Information")
            st.write(f"**LLM Model:** {rag.llm_model_name}")
            if rag.use_openai:
                st.write(f"**LLM Type:** OpenAI API ({rag.openai_model})")
            else:
                st.write(f"**LLM Type:** Local Model")
            st.write(f"**Device:** {rag.device}")
            
            # Sample products
            st.markdown("### 📦 Sample Products")
            sample_df = rag.df[['product_name', 'brand_name', 'selling_price']].head(10)
            st.dataframe(sample_df, use_container_width=True)
    
    # Chat history sidebar
    if st.session_state.chat_history:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📝 Recent Queries")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Query {len(st.session_state.chat_history)-i}"):
                    if item['type'] == 'image':
                        st.image(item['image'], width=100)
                    st.write(f"**Q:** {item['query'][:50]}...")
                    st.write(f"**A:** {item['answer'][:100]}...")

if __name__ == "__main__":
    main() 
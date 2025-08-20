import streamlit as st
import tempfile
import os
from rag_backend import MultimodalRAG
from config import Config
from typing import List

# Page config
st.set_page_config(
    page_title="Multimodal RAG for Amazon Products", 
    page_icon="🛒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
st.markdown(Config.get_css_styles(), unsafe_allow_html=True)

def clean_description(description: str) -> str:
    """Clean generic Amazon template phrases from product descriptions."""
    # Common generic phrases to filter out
    generic_phrases = [
        "Make sure this fits by entering your model number.",
        "Make sure this fits by entering your model number",
        "Please check the size chart before purchasing",
        "Customer satisfaction is our top priority",
        "If you have any questions, please contact us",
        "100% brand new and high quality",
        "Package includes:",
        "Note: Light shooting and different displays",
        "Due to the difference between different monitors"
    ]
    
    cleaned = description
    for phrase in generic_phrases:
        # Remove the phrase and any trailing/leading whitespace
        cleaned = cleaned.replace(phrase, "").strip()
        # Clean up any resulting double spaces or punctuation
        cleaned = " ".join(cleaned.split())
        # Remove leading/trailing punctuation that might be left over
        cleaned = cleaned.strip(".,;:-")
    
    return cleaned.strip() if cleaned.strip() else description

def display_product_sources(sources: List[str]):
    """Display retrieved product sources with proper image handling"""
    for i, source in enumerate(sources, 1):
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
            brand = None
            description_parts = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('Image: http'):
                    continue
                
                # Extract product title (first meaningful line with **)
                if product_title is None and line.startswith('- **') and '**' in line:
                    # Extract title between **
                    title_match = line.split('**')
                    if len(title_match) >= 3:
                        product_title = title_match[1].strip()
                    
                    # Extract brand and price from the same line
                    if '(Brand:' in line and 'Price:' in line:
                        paren_content = line.split('(')[1].split(')')[0]
                        parts = paren_content.split(', ')
                        for part in parts:
                            if part.startswith('Brand:'):
                                brand_value = part.replace('Brand:', '').strip()
                                if brand_value and brand_value != 'nan':
                                    brand = brand_value
                            elif part.startswith('Price:'):
                                price_value = part.replace('Price:', '').strip()
                                if price_value and price_value != 'N/A':
                                    price = price_value
                    continue
                
                # Extract description - lines starting with "About:"
                if line.startswith('About:'):
                    desc_text = line.replace('About:', '').strip()
                    if desc_text:
                        # Clean the description of generic phrases
                        cleaned_desc = clean_description(desc_text)
                        if cleaned_desc and len(cleaned_desc) > 10:  # Only add if substantial content remains
                            description_parts.append(cleaned_desc)
            
            # Display formatted information
            if product_title:
                st.markdown(f"**{product_title}**")
            
            # Display brand and price if available
            info_cols = st.columns(2)
            with info_cols[0]:
                if brand:
                    st.write(f"🏷️ **Brand:** {brand}")
            with info_cols[1]:
                if price:
                    st.write(f"💰 **Price:** {price}")
            
            # Display description
            if description_parts:
                st.write("📝 **Description:**")
                for desc in description_parts:
                    st.write(desc)

def display_product_image(product: dict):
    """Display a single product with its image prominently."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display image
        if product['images'] and product['images'] != 'nan' and product['images'].strip():
            # Get first image URL
            image_urls = product['images'].split(';')
            if image_urls:
                first_image = image_urls[0].strip()
                if first_image:
                    st.image(first_image, use_container_width=True)
                else:
                    st.info("No image available")
            else:
                st.info("No image available")
        else:
            st.info("No image available")
    
    with col2:
        # Product details
        st.markdown(f"### {product['name']}")
        
        # Brand and price in columns
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            if product['brand'] and product['brand'] != 'nan':
                st.markdown(f"**Brand:** {product['brand']}")
        with detail_col2:
            if product['price'] and product['price'] != 'nan':
                st.markdown(f"**Price:** {product['price']}")
        
        # Description
        if product['about'] and product['about'] != 'nan':
            description = clean_description(product['about'])
            if description:
                st.markdown("**Description:**")
                st.markdown(description)

# Main title
st.title("🛒 Multimodal RAG: Amazon Product Search")
st.markdown("Search Amazon products using text queries or upload images for visual search!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    st.subheader("🔑 API Configuration")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for GPT models",
        placeholder="sk-..."
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    # Fixed model settings
    st.subheader("🤖 System Configuration")
    st.info("**CLIP Model**: clip-ViT-B-32 (for multimodal embeddings)")
    st.info("**LLM Model**: openai/gpt-4o-mini (for answer generation)")
    
    # Search settings
    st.subheader("🔍 Search Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    # Initialize system
    if st.button("🚀 Initialize RAG System", type="primary"):
        try:
            if not openai_key:
                st.error("❌ OpenAI API key required. Please enter your API key above.")
                st.stop()
            
            with st.spinner("Initializing RAG system..."):
                rag = MultimodalRAG(
                    csv_path=None,  # Use artifacts only
                    clip_model="clip-ViT-B-32",
                    llm_model="openai/gpt-4o-mini",
                    top_k=top_k
                )
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
            
            st.success("✅ RAG system initialized successfully!")
            st.info(f"📊 Loaded {len(rag.df)} products")
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")

# Main content
if st.session_state.get('data_loaded', False):
    rag = st.session_state.rag_system
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Text Query", "🖼️ Image Query", "🖼️ Product Image Request"])
    
    with tab1:
        st.header("Text-based Product Search")
        user_question = st.text_input(
            "Enter your question about products:",
            placeholder="What are the dimensions of the Electronic Snap Circuits Mini Kits Classpack?"
        )
        
        if st.button("🔍 Search", key="text_search", type="primary"):
            if user_question:
                with st.spinner("Searching products..."):
                    try:
                        answer, sources = rag.answer_text_question(user_question)
                        
                        # Display answer
                        st.markdown("### 🤖 Answer")
                        st.markdown(answer)
                        
                        # Display sources with proper formatting
                        st.markdown("### 📦 Retrieved Products")
                        display_product_sources(sources[:3])
                                
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question to search")
    
    with tab2:
        st.header("Image-based Product Search")
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a product image to find similar items"
        )
        
        if uploaded_image:
            from PIL import Image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            image_question = st.text_input(
                "Ask a question about this image:",
                placeholder="Find products similar to this image"
            )
            
            if st.button("🔍 Analyze", key="image_search", type="primary"):
                if image_question:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        img_path = tmp_file.name
                    
                    with st.spinner("Analyzing image and searching..."):
                        try:
                            answer, sources = rag.answer_image_question(img_path, image_question)
                            
                            # Display answer
                            st.markdown("### 🤖 Answer")
                            st.markdown(answer)
                            
                            # Display sources with proper formatting
                            st.markdown("### 📦 Retrieved Products")
                            display_product_sources(sources[:3])
                                    
                        except Exception as e:
                            st.error(f"❌ Error processing image query: {str(e)}")
                        finally:
                            os.unlink(img_path)
                else:
                    st.warning("⚠️ Please enter a question about the image")
    
    with tab3:
        st.header("Request Specific Product Image")
        st.markdown("Search for a specific product by name to view its image and details.")
        
        product_name = st.text_input(
            "Enter product name:",
            placeholder="LEGO Minecraft Creeper BigFig"
        )
        
        if st.button("🔍 Find Product", key="product_search", type="primary"):
            if product_name:
                with st.spinner("Searching for product..."):
                    try:
                        matches = rag.search_product_by_name(product_name)
                        
                        if matches:
                            st.markdown(f"### 🎯 Found {len(matches)} matching product(s)")
                            
                            for i, product in enumerate(matches):
                                with st.expander(f"Product {i+1}: {product['name']}", expanded=(i==0)):
                                    display_product_image(product)
                                    
                        else:
                            st.warning(f"⚠️ No products found matching '{product_name}'. Try a different search term or check spelling.")
                            
                    except Exception as e:
                        st.error(f"❌ Error searching for product: {str(e)}")
            else:
                st.warning("⚠️ Please enter a product name to search")
else:
    st.info("👈 Please initialize the RAG system using the sidebar to start searching!") 
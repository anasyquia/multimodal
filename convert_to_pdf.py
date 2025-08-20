#!/usr/bin/env python3
"""
Convert Markdown documentation to PDF
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'markdown',
        'weasyprint',
        'pygments'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"{sys.executable} -m pip install {package}")

def markdown_to_pdf(markdown_file, output_file=None):
    """Convert markdown file to PDF"""
    
    # Install requirements if needed
    install_requirements()
    
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    
    # Set output file
    if not output_file:
        output_file = markdown_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'codehilite',
        'fenced_code',
        'tables',
        'toc'
    ])
    
    html_content = md.convert(markdown_content)
    
    # Add CSS styling
    css_style = """
    @page {
        size: A4;
        margin: 2cm;
        @top-center {
            content: "Multimodal RAG Technical Documentation";
            font-size: 12px;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page);
            font-size: 12px;
            color: #666;
        }
    }
    
    body {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: none;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    h1 {
        font-size: 28px;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    
    h2 {
        font-size: 24px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
    }
    
    h3 {
        font-size: 20px;
        color: #34495e;
    }
    
    h4 {
        font-size: 18px;
        color: #34495e;
    }
    
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    pre {
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 6px;
        overflow: auto;
        border-left: 4px solid #3498db;
        margin: 16px 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        font-size: 13px;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    
    th {
        background-color: #f8f9fa;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 16px 0;
        padding: 0 16px;
        color: #666;
    }
    
    ul, ol {
        margin: 16px 0;
        padding-left: 24px;
    }
    
    li {
        margin: 8px 0;
    }
    
    .toc {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 6px;
        margin: 20px 0;
    }
    
    .toc ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .toc li {
        margin: 5px 0;
    }
    
    .toc a {
        text-decoration: none;
        color: #3498db;
    }
    
    .highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    hr {
        border: none;
        border-top: 2px solid #ecf0f1;
        margin: 32px 0;
    }
    
    .page-break {
        page-break-before: always;
    }
    """
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Multimodal RAG Technical Documentation</title>
        <style>
        {css_style}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    font_config = FontConfiguration()
    html_doc = HTML(string=full_html)
    css_doc = CSS(string=css_style, font_config=font_config)
    
    print(f"Converting {markdown_file} to {output_file}...")
    html_doc.write_pdf(output_file, stylesheets=[css_doc], font_config=font_config)
    print(f"✅ PDF created successfully: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Get the markdown file path
    markdown_file = "Multimodal_RAG_Technical_Documentation.md"
    
    if not os.path.exists(markdown_file):
        print(f"❌ Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    # Convert to PDF
    try:
        pdf_file = markdown_to_pdf(markdown_file)
        print(f"🎉 Documentation successfully converted to PDF: {pdf_file}")
        print(f"📄 File size: {os.path.getsize(pdf_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {str(e)}")
        print("💡 Try installing dependencies manually:")
        print("pip install markdown weasyprint pygments")
        sys.exit(1) 
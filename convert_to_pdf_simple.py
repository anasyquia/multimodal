#!/usr/bin/env python3
"""
Convert Markdown documentation to PDF using a simpler approach
"""

import os
import sys
import re
from datetime import datetime

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'markdown',
        'reportlab'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"{sys.executable} -m pip install {package}")

def markdown_to_pdf_simple(markdown_file, output_file=None):
    """Convert markdown file to PDF using reportlab"""
    
    # Install requirements if needed
    install_requirements()
    
    import markdown
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, blue, black
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    
    # Set output file
    if not output_file:
        output_file = markdown_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_content = md.convert(markdown_content)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        textColor=blue,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=blue
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=16,
        textColor=Color(0.2, 0.3, 0.5)
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor=Color(0.3, 0.4, 0.6)
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        spaceAfter=6,
        spaceBefore=6,
        leftIndent=20,
        backgroundColor=Color(0.95, 0.95, 0.95)
    )
    
    # Parse content and create story
    story = []
    
    # Split content by lines
    lines = markdown_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        # Title (first # heading)
        if line.startswith('# ') and 'Multimodal RAG' in line:
            title_text = line[2:].strip()
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 20))
            
        # Main headings
        elif line.startswith('## '):
            heading_text = line[3:].strip()
            story.append(Paragraph(heading_text, heading1_style))
            
        elif line.startswith('### '):
            heading_text = line[4:].strip()
            story.append(Paragraph(heading_text, heading2_style))
            
        elif line.startswith('#### '):
            heading_text = line[5:].strip()
            story.append(Paragraph(heading_text, heading3_style))
            
        # Code blocks
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if code_lines:
                code_text = '<br/>'.join(code_lines)
                story.append(Paragraph(code_text, code_style))
            
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            list_text = line[2:].strip()
            list_text = f"• {list_text}"
            story.append(Paragraph(list_text, styles['Normal']))
            
        elif re.match(r'^\d+\.', line):
            list_text = line.strip()
            story.append(Paragraph(list_text, styles['Normal']))
            
        # Tables (simple handling)
        elif '|' in line and line.count('|') >= 2:
            # Skip table headers and separators for now
            if not line.startswith('|---'):
                table_text = line.replace('|', ' | ')
                story.append(Paragraph(table_text, styles['Normal']))
            
        # Regular paragraphs
        elif line and not line.startswith('---'):
            # Clean up markdown syntax
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
            text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
            
            story.append(Paragraph(text, styles['Normal']))
            
        i += 1
    
    # Add metadata
    story.insert(1, Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.insert(2, Paragraph("Repository: https://github.com/anasyquia/multimodal", styles['Normal']))
    story.insert(3, Spacer(1, 30))
    
    # Build PDF
    print(f"Converting {markdown_file} to {output_file}...")
    doc.build(story)
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
        pdf_file = markdown_to_pdf_simple(markdown_file)
        print(f"🎉 Documentation successfully converted to PDF: {pdf_file}")
        print(f"📄 File size: {os.path.getsize(pdf_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Error converting to PDF: {str(e)}")
        print("💡 Try installing dependencies manually:")
        print("pip install markdown reportlab")
        sys.exit(1) 
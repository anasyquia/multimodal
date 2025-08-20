#!/usr/bin/env python3
"""
Script to run evaluation and update documentation with real metrics
"""

import json
import re
from datetime import datetime
from evaluate_system import RAGEvaluator, MultimodalRAG

def run_evaluation_and_update_docs():
    """Run evaluation and update documentation with real metrics"""
    
    print("🚀 Running evaluation to generate real metrics...")
    
    try:
        # Run evaluation
        rag_system = MultimodalRAG()
        rag_system.load_artifacts()
        
        evaluator = RAGEvaluator(rag_system)
        results = evaluator.run_full_evaluation()
        
        # Update documentation
        update_documentation_with_real_metrics(results)
        
        print("✅ Documentation updated with real metrics!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def update_documentation_with_real_metrics(results):
    """Update the documentation with actual evaluation results"""
    
    doc_file = "Multimodal_RAG_Technical_Documentation.md"
    
    # Read current documentation
    with open(doc_file, 'r') as f:
        content = f.read()
    
    # Extract metrics from results
    perf = results['performance_metrics']
    acc = results['accuracy_metrics']
    qual = results['quality_metrics']
    db = results['database_coverage']
    
    # Create real metrics sections
    real_performance_section = f"""**Response Time Analysis:**
```
Query Type          | Avg Response Time | 95th Percentile
Text Query          | {perf.get('text_queries_avg', 0):.1f}s             | {perf.get('text_queries_95th_percentile', 0):.1f}s
Product Search      | {perf.get('product_name_queries_avg', 0):.1f}s             | {perf.get('product_name_queries_95th_percentile', 0):.1f}s
System Load         | {perf.get('system_load_time_avg', 0):.1f}s             | {perf.get('system_load_time_95th_percentile', 0):.1f}s
```"""

    real_accuracy_section = f"""**Accuracy Metrics:**
- **Query Success Rate**: {acc.get('success_rate', 0):.1%}
- **Result Relevance**: {acc.get('relevance_rate', 0):.1%} relevant results
- **Category Matching**: {acc.get('category_accuracy', 0):.1%} accuracy
- **Brand Recognition**: {acc.get('brand_accuracy', 0):.1%} accuracy"""

    real_quality_section = f"""**Quality Assessment:**
- **Response Coherence**: {qual.get('coherence_rate', 0):.1%} coherent responses
- **Product Inclusion**: {qual.get('product_inclusion_rate', 0):.1%} responses include products
- **Error Rate**: {qual.get('error_rate', 0):.1%} system errors
- **Avg Response Length**: {qual.get('avg_response_length', 0):.0f} characters"""

    real_database_section = f"""**Database Coverage:**
- **Total Products**: {db.get('total_products', 0):,} products indexed
- **Image Availability**: {db.get('image_coverage', 0):.1%} have valid images
- **Description Coverage**: {db.get('description_coverage', 0):.1%} have descriptions  
- **Price Information**: {db.get('price_coverage', 0):.1%} have price data
- **Categories**: {db.get('unique_categories', 0)} unique categories
- **Brands**: {db.get('unique_brands', 0)} unique brands"""

    # Replace the hypothetical metrics sections
    
    # Replace response time analysis
    old_response_pattern = r'\*\*Response Time Analysis:\*\*\n```[\s\S]*?```'
    content = re.sub(old_response_pattern, real_performance_section, content)
    
    # Replace accuracy metrics
    old_accuracy_pattern = r'\*\*Accuracy Metrics:\*\*\n- \*\*Text Query Relevance\*\*:[\s\S]*?- \*\*Product Name Matching\*\*:.*'
    content = re.sub(old_accuracy_pattern, real_accuracy_section, content)
    
    # Replace quality assessment
    old_quality_pattern = r'\*\*Search Result Evaluation:\*\*\n```python[\s\S]*?```'
    content = re.sub(old_quality_pattern, real_quality_section, content)
    
    # Add database coverage section
    if "**Database Coverage:**" not in content:
        # Find a good place to insert it (after quality metrics)
        quality_end = content.find("**Error Analysis:**")
        if quality_end != -1:
            content = content[:quality_end] + real_database_section + "\n\n" + content[quality_end:]
    
    # Add timestamp note
    timestamp_note = f"""
> **Note**: Metrics updated with actual evaluation results on {datetime.now().strftime('%B %d, %Y at %H:%M')}
"""
    
    # Insert timestamp after performance analysis section
    perf_section_end = content.find("### 8.2 Scalability Analysis")
    if perf_section_end != -1:
        content = content[:perf_section_end] + timestamp_note + content[perf_section_end:]
    
    # Write updated documentation
    with open(doc_file, 'w') as f:
        f.write(content)
    
    print(f"📝 Updated {doc_file} with real evaluation metrics")

if __name__ == "__main__":
    success = run_evaluation_and_update_docs()
    if success:
        print("\n🎉 Your documentation now contains real evaluation metrics!")
        print("📊 Check evaluation_results_YYYYMMDD_HHMMSS.json for detailed results")
    else:
        print("\n❌ Failed to generate real metrics. Check your system setup.") 
#!/usr/bin/env python3
"""
Evaluation script for Multimodal RAG system
Generates actual performance metrics and quality assessments
"""

import time
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Import your RAG system
from rag_backend import MultimodalRAG

class RAGEvaluator:
    def __init__(self, rag_system: MultimodalRAG):
        self.rag_system = rag_system
        self.results = {
            'performance_metrics': {},
            'accuracy_metrics': {},
            'quality_metrics': {},
            'error_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create a set of test queries for evaluation"""
        test_queries = [
            # Text queries
            {"type": "text", "query": "wireless headphones", "expected_category": "Electronics"},
            {"type": "text", "query": "LEGO building blocks", "expected_category": "Toys"},
            {"type": "text", "query": "kitchen knife set", "expected_category": "Kitchen"},
            {"type": "text", "query": "running shoes Nike", "expected_brand": "Nike"},
            {"type": "text", "query": "smartphone iPhone", "expected_brand": "Apple"},
            
            # Product name queries
            {"type": "product_name", "query": "iPhone 14", "expected_found": True},
            {"type": "product_name", "query": "Nintendo Switch", "expected_found": True},
            {"type": "product_name", "query": "nonexistent product xyz123", "expected_found": False},
            
            # Edge cases
            {"type": "text", "query": "", "expected_error": True},
            {"type": "text", "query": "a", "expected_results": "minimal"},
            {"type": "text", "query": "the best product ever made in the world", "expected_results": "variable"},
        ]
        return test_queries
    
    def measure_response_times(self, test_queries: List[Dict], num_runs: int = 5) -> Dict[str, float]:
        """Measure response times for different query types"""
        print("📊 Measuring response times...")
        
        response_times = {
            'text_queries': [],
            'product_name_queries': [],
            'system_load_time': []
        }
        
        # Measure system initialization time
        for i in range(3):
            start_time = time.time()
            # Simulate system reload
            self.rag_system.load_artifacts()
            load_time = time.time() - start_time
            response_times['system_load_time'].append(load_time)
            print(f"  System load {i+1}: {load_time:.2f}s")
        
        # Measure query response times
        for query_data in test_queries:
            if query_data.get('expected_error'):
                continue
                
            query_times = []
            for run in range(num_runs):
                start_time = time.time()
                
                try:
                    if query_data['type'] == 'text':
                        # Text query
                        results = self.rag_system.query(query_data['query'], top_k=5)
                    elif query_data['type'] == 'product_name':
                        # Product name search
                        results = self.rag_system.search_product_by_name(query_data['query'])
                    
                    response_time = time.time() - start_time
                    query_times.append(response_time)
                    
                except Exception as e:
                    print(f"  Error in query '{query_data['query']}': {e}")
                    continue
            
            if query_times:
                avg_time = np.mean(query_times)
                response_times[f"{query_data['type']}_queries"].extend(query_times)
                print(f"  Query '{query_data['query'][:30]}...': {avg_time:.3f}s avg")
        
        # Calculate summary statistics
        summary = {}
        for query_type, times in response_times.items():
            if times:
                summary[f"{query_type}_avg"] = np.mean(times)
                summary[f"{query_type}_median"] = np.median(times)
                summary[f"{query_type}_95th_percentile"] = np.percentile(times, 95)
                summary[f"{query_type}_std"] = np.std(times)
        
        return summary
    
    def evaluate_accuracy(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate accuracy of search results"""
        print("🎯 Evaluating accuracy...")
        
        accuracy_results = {
            'total_queries': 0,
            'successful_queries': 0,
            'relevant_results': 0,
            'category_matches': 0,
            'brand_matches': 0,
            'expected_not_found': 0
        }
        
        for query_data in test_queries:
            if query_data.get('expected_error'):
                continue
                
            accuracy_results['total_queries'] += 1
            
            try:
                if query_data['type'] == 'text':
                    results = self.rag_system.query(query_data['query'], top_k=5)
                    if results and len(results) > 0:
                        accuracy_results['successful_queries'] += 1
                        
                        # Check for expected category
                        if 'expected_category' in query_data:
                            for result in results:
                                if query_data['expected_category'].lower() in result.get('main_category', '').lower():
                                    accuracy_results['category_matches'] += 1
                                    break
                        
                        # Check for expected brand
                        if 'expected_brand' in query_data:
                            for result in results:
                                if query_data['expected_brand'].lower() in result.get('store', '').lower():
                                    accuracy_results['brand_matches'] += 1
                                    break
                        
                        # Basic relevance check (query terms in results)
                        query_terms = query_data['query'].lower().split()
                        for result in results:
                            result_text = f"{result.get('title', '')} {result.get('description', '')}".lower()
                            if any(term in result_text for term in query_terms):
                                accuracy_results['relevant_results'] += 1
                                break
                
                elif query_data['type'] == 'product_name':
                    results = self.rag_system.search_product_by_name(query_data['query'])
                    
                    if query_data.get('expected_found', True):
                        if results and len(results) > 0:
                            accuracy_results['successful_queries'] += 1
                    else:
                        if not results or len(results) == 0:
                            accuracy_results['expected_not_found'] += 1
                            accuracy_results['successful_queries'] += 1
                            
            except Exception as e:
                print(f"  Error evaluating '{query_data['query']}': {e}")
                continue
        
        # Calculate percentages
        total = accuracy_results['total_queries']
        if total > 0:
            accuracy_results['success_rate'] = accuracy_results['successful_queries'] / total
            accuracy_results['relevance_rate'] = accuracy_results['relevant_results'] / total
            accuracy_results['category_accuracy'] = accuracy_results['category_matches'] / total
            accuracy_results['brand_accuracy'] = accuracy_results['brand_matches'] / total
        
        return accuracy_results
    
    def evaluate_quality(self, sample_queries: List[str]) -> Dict[str, Any]:
        """Evaluate response quality"""
        print("✨ Evaluating response quality...")
        
        quality_metrics = {
            'response_lengths': [],
            'responses_with_products': 0,
            'responses_with_errors': 0,
            'coherent_responses': 0,
            'total_evaluated': 0
        }
        
        for query in sample_queries:
            try:
                result = self.rag_system.query(query, top_k=5)
                quality_metrics['total_evaluated'] += 1
                
                if result:
                    response_text = result.get('response', '')
                    products = result.get('products', [])
                    
                    # Response length
                    quality_metrics['response_lengths'].append(len(response_text))
                    
                    # Has products
                    if products and len(products) > 0:
                        quality_metrics['responses_with_products'] += 1
                    
                    # Basic coherence check (no obvious errors)
                    if response_text and len(response_text) > 10 and "error" not in response_text.lower():
                        quality_metrics['coherent_responses'] += 1
                
            except Exception as e:
                quality_metrics['responses_with_errors'] += 1
                print(f"  Error in quality evaluation for '{query}': {e}")
        
        # Calculate quality statistics
        if quality_metrics['response_lengths']:
            quality_metrics['avg_response_length'] = np.mean(quality_metrics['response_lengths'])
            quality_metrics['median_response_length'] = np.median(quality_metrics['response_lengths'])
        
        total = quality_metrics['total_evaluated']
        if total > 0:
            quality_metrics['product_inclusion_rate'] = quality_metrics['responses_with_products'] / total
            quality_metrics['coherence_rate'] = quality_metrics['coherent_responses'] / total
            quality_metrics['error_rate'] = quality_metrics['responses_with_errors'] / total
        
        return quality_metrics
    
    def analyze_database_coverage(self) -> Dict[str, Any]:
        """Analyze the product database coverage"""
        print("📚 Analyzing database coverage...")
        
        try:
            products = self.rag_system.artifacts.products
            
            coverage_stats = {
                'total_products': len(products),
                'products_with_images': 0,
                'products_with_descriptions': 0,
                'products_with_prices': 0,
                'unique_categories': set(),
                'unique_brands': set(),
                'price_range': {'min': float('inf'), 'max': 0}
            }
            
            for product in products:
                # Images
                if product.get('images') and product['images'] != 'nan':
                    coverage_stats['products_with_images'] += 1
                
                # Descriptions
                if product.get('description') and len(str(product['description'])) > 10:
                    coverage_stats['products_with_descriptions'] += 1
                
                # Prices
                price_str = str(product.get('price', ''))
                if price_str and price_str != 'nan':
                    coverage_stats['products_with_prices'] += 1
                    try:
                        # Extract numeric price
                        price_num = float(''.join(c for c in price_str if c.isdigit() or c == '.'))
                        coverage_stats['price_range']['min'] = min(coverage_stats['price_range']['min'], price_num)
                        coverage_stats['price_range']['max'] = max(coverage_stats['price_range']['max'], price_num)
                    except:
                        pass
                
                # Categories and brands
                if product.get('main_category'):
                    coverage_stats['unique_categories'].add(product['main_category'])
                if product.get('store'):
                    coverage_stats['unique_brands'].add(product['store'])
            
            # Convert sets to counts
            coverage_stats['unique_categories'] = len(coverage_stats['unique_categories'])
            coverage_stats['unique_brands'] = len(coverage_stats['unique_brands'])
            
            # Calculate percentages
            total = coverage_stats['total_products']
            if total > 0:
                coverage_stats['image_coverage'] = coverage_stats['products_with_images'] / total
                coverage_stats['description_coverage'] = coverage_stats['products_with_descriptions'] / total
                coverage_stats['price_coverage'] = coverage_stats['products_with_prices'] / total
            
            return coverage_stats
            
        except Exception as e:
            print(f"Error analyzing database: {e}")
            return {'error': str(e)}
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("🚀 Starting comprehensive RAG system evaluation...")
        print("=" * 50)
        
        # Create test data
        test_queries = self.create_test_queries()
        sample_queries = [
            "wireless bluetooth headphones",
            "kitchen appliances",
            "children's toys",
            "smartphone accessories",
            "home office furniture"
        ]
        
        # Run evaluations
        self.results['performance_metrics'] = self.measure_response_times(test_queries)
        print()
        
        self.results['accuracy_metrics'] = self.evaluate_accuracy(test_queries)
        print()
        
        self.results['quality_metrics'] = self.evaluate_quality(sample_queries)
        print()
        
        self.results['database_coverage'] = self.analyze_database_coverage()
        print()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def save_results(self):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {filename}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("📊 EVALUATION SUMMARY")
        print("=" * 50)
        
        # Performance
        perf = self.results['performance_metrics']
        print(f"⚡ Performance Metrics:")
        print(f"  • Text Query Avg Time: {perf.get('text_queries_avg', 0):.3f}s")
        print(f"  • Product Search Avg Time: {perf.get('product_name_queries_avg', 0):.3f}s")
        print(f"  • System Load Time: {perf.get('system_load_time_avg', 0):.3f}s")
        print()
        
        # Accuracy
        acc = self.results['accuracy_metrics']
        print(f"🎯 Accuracy Metrics:")
        print(f"  • Success Rate: {acc.get('success_rate', 0):.1%}")
        print(f"  • Relevance Rate: {acc.get('relevance_rate', 0):.1%}")
        print(f"  • Category Accuracy: {acc.get('category_accuracy', 0):.1%}")
        print()
        
        # Quality
        qual = self.results['quality_metrics']
        print(f"✨ Quality Metrics:")
        print(f"  • Coherence Rate: {qual.get('coherence_rate', 0):.1%}")
        print(f"  • Product Inclusion Rate: {qual.get('product_inclusion_rate', 0):.1%}")
        print(f"  • Error Rate: {qual.get('error_rate', 0):.1%}")
        print()
        
        # Database
        db = self.results['database_coverage']
        print(f"📚 Database Coverage:")
        print(f"  • Total Products: {db.get('total_products', 0):,}")
        print(f"  • Image Coverage: {db.get('image_coverage', 0):.1%}")
        print(f"  • Unique Categories: {db.get('unique_categories', 0)}")
        print(f"  • Unique Brands: {db.get('unique_brands', 0)}")

def main():
    """Main evaluation function"""
    if not os.path.exists('artifacts'):
        print("❌ No artifacts directory found. Please run setup first.")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system = MultimodalRAG()
        rag_system.load_artifacts()
        
        # Run evaluation
        evaluator = RAGEvaluator(rag_system)
        results = evaluator.run_full_evaluation()
        
        print("\n✅ Evaluation complete!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
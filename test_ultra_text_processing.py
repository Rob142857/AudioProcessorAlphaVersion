"""
Test Ultra-Enhanced Text Processing Capabilities

This script demonstrates the new ultra-enhanced text processing with:
1. Multi-pass processing (up to 6 passes)
2. Parallel processing across multiple CPU cores
3. Advanced sentence and paragraph segmentation
4. Better punctuation and capitalization
5. Quality assessment

Usage:
    python test_ultra_text_processing.py
"""

import time
import multiprocessing
from text_processor_ultra import create_ultra_processor, create_advanced_paragraph_formatter

def test_basic_processing():
    """Test basic ultra text processing."""
    print("🧪 Testing Basic Ultra Text Processing")
    print("=" * 50)
    
    # Sample raw transcription text (typical Whisper output)
    raw_text = """hello everyone welcome to todays lecture on machine learning 
    were going to talk about neural networks and deep learning algorithms 
    first lets discuss what machine learning is its a subset of artificial intelligence 
    that focuses on building systems that can learn from data without being explicitly programmed 
    there are three main types of machine learning supervised learning unsupervised learning 
    and reinforcement learning supervised learning uses labeled data to train models 
    unsupervised learning finds patterns in data without labels and reinforcement learning 
    learns through trial and error by receiving rewards or penalties for actions"""
    
    print("Raw text:")
    print(repr(raw_text))
    print()
    
    # Process with ultra processor
    processor = create_ultra_processor(max_workers=4)
    
    start_time = time.time()
    enhanced_text = processor.process_text_ultra(raw_text, passes=6)
    processing_time = time.time() - start_time
    
    print("Ultra-enhanced text:")
    print(enhanced_text)
    print()
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    return enhanced_text

def test_paragraph_formatting():
    """Test advanced paragraph formatting."""
    print("🧪 Testing Advanced Paragraph Formatting")
    print("=" * 50)
    
    # Long text that needs paragraph breaks
    long_text = """Machine learning is a powerful technology that has revolutionized many industries. 
    It allows computers to learn patterns from data and make predictions or decisions without being 
    explicitly programmed. However, implementing machine learning requires careful consideration of 
    many factors. First, you need high-quality data that is representative of the problem you're 
    trying to solve. The data should be clean, properly labeled, and free from bias. Second, you 
    need to choose the right algorithm for your specific use case. Different algorithms work better 
    for different types of problems. For example, neural networks are excellent for image recognition, 
    while decision trees work well for classification tasks with clear rules. Third, you need to 
    properly evaluate your model's performance using appropriate metrics and validation techniques. 
    This helps ensure that your model will work well on new, unseen data. Finally, you need to 
    consider the ethical implications of your machine learning system. This includes ensuring 
    fairness, transparency, and accountability in your algorithms."""
    
    formatter = create_advanced_paragraph_formatter(max_workers=2)
    
    start_time = time.time()
    formatted_text = formatter.format_paragraphs_advanced(long_text, target_length=400)
    formatting_time = time.time() - start_time
    
    print("Formatted text with intelligent paragraph breaks:")
    print(formatted_text)
    print()
    
    print(f"Formatting completed in {formatting_time:.2f} seconds")
    return formatted_text

def test_performance_comparison():
    """Test performance with different worker counts."""
    print("🧪 Testing Performance with Different Worker Counts")
    print("=" * 50)
    
    test_text = """this is a performance test of the ultra text processing system 
    we want to see how different numbers of worker threads affect processing speed 
    the text contains various issues that need to be fixed including punctuation 
    capitalization sentence boundaries and paragraph formatting the system should 
    handle all of these issues efficiently across multiple processing passes"""
    
    worker_counts = [1, 2, 4, min(8, multiprocessing.cpu_count())]
    
    results = []
    
    for workers in worker_counts:
        print(f"\nTesting with {workers} workers:")
        
        processor = create_ultra_processor(max_workers=workers)
        
        start_time = time.time()
        result = processor.process_text_ultra(test_text, passes=5)
        processing_time = time.time() - start_time
        
        results.append((workers, processing_time, len(result)))
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Output length: {len(result)} chars")
        
        # Show quality metrics if available
        if hasattr(processor, 'quality_metrics') and processor.quality_metrics:
            metrics = processor.quality_metrics
            if 'avg_processing_speed' in metrics:
                print(f"  Speed: {metrics['avg_processing_speed']:.0f} chars/sec")
    
    print("\n📊 Performance Summary:")
    print("Workers | Time (s) | Speed (chars/sec)")
    print("-" * 35)
    
    for workers, time_taken, output_len in results:
        speed = len(test_text) / time_taken if time_taken > 0 else 0
        print(f"{workers:7d} | {time_taken:8.3f} | {speed:13.0f}")

def test_quality_assessment():
    """Test quality assessment features."""
    print("🧪 Testing Quality Assessment")
    print("=" * 50)
    
    # Text with various quality issues
    poor_text = """this text has many problems no punctuation poor capitalization 
    its very difficult to read and understand the sentences run together without 
    proper breaks there are also some redundant redundant words and phrases that 
    should be cleaned up the overall readability is quite poor"""
    
    processor = create_ultra_processor(max_workers=2)
    
    print("Original text (poor quality):")
    print(poor_text)
    print()
    
    enhanced_text = processor.process_text_ultra(poor_text, passes=6)
    
    print("Enhanced text:")
    print(enhanced_text)
    print()
    
    # Show quality metrics
    if hasattr(processor, 'quality_metrics') and processor.quality_metrics:
        metrics = processor.quality_metrics
        print("Quality Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

def main():
    """Run all tests."""
    cpu_count = multiprocessing.cpu_count()
    print(f"🚀 Ultra Text Processing Test Suite")
    print(f"💻 System: {cpu_count} CPU cores available")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Basic processing
        test_basic_processing()
        print("\n" + "=" * 60 + "\n")
        
        # Test 2: Paragraph formatting
        test_paragraph_formatting()
        print("\n" + "=" * 60 + "\n")
        
        # Test 3: Performance comparison
        test_performance_comparison()
        print("\n" + "=" * 60 + "\n")
        
        # Test 4: Quality assessment
        test_quality_assessment()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("✨ Ultra-enhanced text processing is ready for use")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
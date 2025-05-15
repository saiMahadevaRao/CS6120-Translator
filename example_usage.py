#!/usr/bin/env python3
"""
Example usage script for the English-French Neural Machine Translation System.

This script demonstrates how to:
1. Load a pre-trained model
2. Translate individual sentences
3. Summarize and translate documents
4. Evaluate translation quality
"""

import torch
import logging
from pathlib import Path

# Import our translation system
from translator import (
    summarize_document, 
    translate_sentence, 
    translate_documents,
    calculate_rouge_scores,
    load_dictionary
)
from config import get_device, get_model_path, get_dictionary_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_dictionaries():
    """Load the trained model and language dictionaries."""
    try:
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load dictionaries
        dict_paths = get_dictionary_paths()
        input_dic = load_dictionary(str(dict_paths["input"]))
        output_dic = load_dictionary(str(dict_paths["output"]))
        
        logger.info(f"Loaded input dictionary with {input_dic.n_count} words")
        logger.info(f"Loaded output dictionary with {output_dic.n_count} words")
        
        # Load model (this would require the model architecture to be defined)
        # For now, we'll just return the dictionaries
        logger.warning("Model loading not implemented in this example - using dictionaries only")
        
        return input_dic, output_dic, device
        
    except FileNotFoundError as e:
        logger.error(f"Could not load model/dictionaries: {e}")
        logger.error("Please ensure you have trained the model first using translator.py")
        return None, None, None


def example_sentence_translation(input_dic, output_dic, device):
    """Example of translating individual sentences."""
    logger.info("=== Example: Sentence Translation ===")
    
    # Sample English sentences
    english_sentences = [
        "Hello, how are you today?",
        "The weather is beautiful today.",
        "I love learning new languages.",
        "Machine learning is fascinating.",
        "Have a great day!"
    ]
    
    logger.info("English sentences:")
    for i, sentence in enumerate(english_sentences, 1):
        logger.info(f"{i}. {sentence}")
    
    logger.info("\nNote: Full translation requires the trained model to be loaded.")
    logger.info("This example shows the preprocessing and tokenization steps.")
    
    # Show tokenization example
    if input_dic:
        logger.info(f"\nTokenization example for '{english_sentences[0]}':")
        # This would normally use the tokenize function from translator.py
        logger.info("(Tokenization would happen here with the trained model)")


def example_document_summarization():
    """Example of document summarization."""
    logger.info("\n=== Example: Document Summarization ===")
    
    # Sample long document
    document = """
    Machine learning is a subset of artificial intelligence that focuses on developing 
    systems that can learn from data. These systems improve their performance over time 
    without being explicitly programmed. The field has seen tremendous growth in recent 
    years, with applications ranging from image recognition to natural language processing.
    
    Deep learning, a subset of machine learning, uses neural networks with multiple 
    layers to model complex patterns in data. Transformers, a type of neural network 
    architecture, have revolutionized natural language processing tasks such as machine 
    translation, text summarization, and question answering.
    
    The success of machine learning systems depends heavily on the quality and quantity 
    of training data. Data preprocessing, feature engineering, and model selection are 
    crucial steps in building effective machine learning systems.
    """
    
    logger.info("Original document:")
    logger.info(document.strip())
    
    # Summarize the document
    summary = summarize_document(document, max_chars=150)
    logger.info(f"\nSummarized document (max 150 chars):")
    logger.info(summary)
    
    # Show different summary lengths
    for max_chars in [50, 100, 200]:
        summary = summarize_document(document, max_chars=max_chars)
        logger.info(f"\nSummary ({max_chars} chars): {summary}")


def example_evaluation():
    """Example of evaluation metrics."""
    logger.info("\n=== Example: Evaluation Metrics ===")
    
    # Sample reference and generated summaries
    reference_summaries = [
        "Machine learning systems learn from data to improve performance.",
        "Deep learning uses neural networks for complex pattern recognition.",
        "Data quality is crucial for machine learning success."
    ]
    
    generated_summaries = [
        "Machine learning improves performance through data learning.",
        "Neural networks enable complex pattern recognition in deep learning.",
        "Machine learning success depends on data quality."
    ]
    
    logger.info("Reference summaries:")
    for i, summary in enumerate(reference_summaries, 1):
        logger.info(f"{i}. {summary}")
    
    logger.info("\nGenerated summaries:")
    for i, summary in enumerate(generated_summaries, 1):
        logger.info(f"{i}. {summary}")
    
    # Calculate ROUGE scores
    try:
        rouge_scores = calculate_rouge_scores(generated_summaries, reference_summaries)
        logger.info(f"\nROUGE Scores:")
        for metric, score in rouge_scores.items():
            logger.info(f"{metric}: {score:.4f}")
    except Exception as e:
        logger.error(f"Could not calculate ROUGE scores: {e}")


def main():
    """Main function to run all examples."""
    logger.info("🚀 English-French Neural Machine Translation System - Examples")
    logger.info("=" * 70)
    
    # Load model and dictionaries
    input_dic, output_dic, device = load_model_and_dictionaries()
    
    # Run examples
    example_sentence_translation(input_dic, output_dic, device)
    example_document_summarization()
    example_evaluation()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Examples completed!")
    logger.info("\nTo use the full translation system:")
    logger.info("1. Train the model: python translator.py")
    logger.info("2. Use the trained model for actual translations")
    logger.info("3. Check the README.md for detailed usage instructions")


if __name__ == "__main__":
    main()

# src/data_processing/preprocess.py
import pandas as pd
import re
import os
import logging
import nltk
from typing import List, Dict, Any
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove special characters
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_chunks(text: str, max_length: int = 512) -> List[str]:
    """
    Split text into chunks respecting sentence boundaries.
    
    Args:
        text: Input text to split
        max_length: Maximum chunk length
        
    Returns:
        List of text chunks
    """
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        # Fallback to simple splitting
        sentences = text.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def prepare_training_data(input_file: str, output_file: str, task_type: str = "classification") -> None:
    """
    Prepare training data for fine-tuning.
    
    Args:
        input_file: Path to raw data
        output_file: Path to save processed data
        task_type: Type of task (classification or generation)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Apply cleaning and preprocessing
        logger.info("Cleaning and preprocessing text...")
        df['processed_text'] = df['text'].apply(clean_text)
        
        # Process based on task type
        if task_type == "classification":
            # Verify label column exists
            if 'label' not in df.columns:
                logger.warning("Label column not found for classification task")
        elif task_type == "generation":
            # For generation tasks like summarization
            if 'target' not in df.columns:
                logger.warning("Target column not found for generation task")
        
        # Save processed data
        logger.info(f"Saving processed data to {output_file}...")
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully processed {len(df)} documents")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except Exception as e:
        logger.error(f"Error during data processing: {e}")

def download_arxiv_dataset(output_file: str, num_examples: int = 1000) -> None:
    """
    Download ArXiv papers dataset from Hugging Face.
    
    Args:
        output_file: Path to save the dataset
        num_examples: Number of examples to download
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info(f"Downloading {num_examples} examples from ArXiv dataset...")
        
        try:
            from datasets import load_dataset
            
            # Load scientific papers dataset (ArXiv subset) with trust_remote_code=True
            dataset = load_dataset(
                "scientific_papers", 
                "arxiv", 
                split=f"train[:{num_examples}]",
                trust_remote_code=True  # This is the key parameter we were missing
            )
            logger.info(f"Successfully downloaded {len(dataset)} examples")
            
            # Extract abstracts and titles
            abstracts = []
            categories = []
            
            # Map ArXiv categories to numerical labels
            category_map = {}
            current_label = 0
            
            for item in dataset:
                abstract = item['abstract']
                
                # Extract category (usually in format cs.CL, math.AG, etc.)
                category = "other"
                if "." in item.get('section_name', ''):
                    category = item['section_name'].split('.')[0]
                
                # Map category to numerical label
                if category not in category_map:
                    category_map[category] = current_label
                    current_label += 1
                
                abstracts.append(abstract)
                categories.append(category_map[category])
            
            # Create dataframe
            df = pd.DataFrame({
                'text': abstracts,
                'label': categories,
                'category': [list(category_map.keys())[list(category_map.values()).index(cat)] for cat in categories]
            })
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset saved to {output_file} with {len(df)} examples and {len(category_map)} categories")
            logger.info(f"Category mapping: {category_map}")
            return True
            
        except ImportError:
            logger.error("The 'datasets' library is not installed. Install it with: pip install datasets")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def create_fallback_dataset(output_file: str, num_examples: int = 1000) -> None:
    """
    Create a fallback dataset if downloading fails.
    
    Args:
        output_file: Path to save the dataset
        num_examples: Number of examples to create
    """
    logger.info("Creating fallback dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Categories for technical documents
    categories = [
        "Machine Learning",
        "Web Development",
        "Database Systems",
        "Network Security",
        "Operating Systems"
    ]
    
    # Template texts for each category
    templates = {
        "Machine Learning": [
            "Deep learning models have revolutionized image recognition through the use of convolutional neural networks.",
            "Natural language processing techniques allow computers to understand and generate human language.",
            "Reinforcement learning enables agents to learn optimal behavior through interaction with an environment.",
            "Transfer learning allows models trained on one task to be applied to different but related tasks.",
            "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models."
        ],
        "Web Development": [
            "RESTful APIs provide a standardized way for web applications to communicate with each other.",
            "JavaScript frameworks like React and Vue have transformed front-end development practices.",
            "Responsive web design ensures websites display properly across various device sizes and screen resolutions.",
            "Web accessibility guidelines help developers create sites that can be used by people with diverse abilities.",
            "Content management systems provide non-technical users with the ability to update website content."
        ],
        "Database Systems": [
            "Relational databases organize data into tables with rows and columns linked by relationships.",
            "NoSQL databases provide flexible schema designs for handling unstructured and semi-structured data.",
            "Database indexing improves query performance by creating data structures that speed up data retrieval.",
            "Database transactions ensure data integrity by grouping operations that must succeed or fail as a unit.",
            "Sharding is a database partitioning strategy that distributes data across multiple servers."
        ],
        "Network Security": [
            "Firewalls monitor and filter network traffic based on predetermined security rules.",
            "Encryption protocols protect data by converting it into code that can only be deciphered with the correct key.",
            "Virtual private networks create secure connections over public networks by tunneling protocols.",
            "Intrusion detection systems monitor networks for suspicious activities and policy violations.",
            "Multi-factor authentication enhances security by requiring multiple verification methods."
        ],
        "Operating Systems": [
            "Process scheduling algorithms determine which processes receive CPU time and in what order.",
            "Memory management in operating systems handles the allocation and deallocation of memory resources.",
            "File systems organize and store data on storage devices using various structures and algorithms.",
            "System calls provide an interface between user applications and operating system services.",
            "Virtualization technology allows multiple operating systems to run on a single physical machine."
        ]
    }
    
    # Generate dataset
    texts = []
    labels = []
    category_names = []
    
    while len(texts) < num_examples:
        for i, category in enumerate(categories):
            category_templates = templates[category]
            
            # Generate variations of each template
            for template in category_templates:
                # Add variations with different wording
                variations = []
                
                # Original template
                variations.append(template)
                
                # Variation 1: Add an introductory phrase
                intros = ["It is well known that ", "Experts agree that ", "Research shows that ", 
                           "Studies indicate that ", "It has been demonstrated that "]
                variations.append(random.choice(intros) + template.lower())
                
                # Variation 2: Add explanatory conclusion
                conclusions = [" This has significant implications for practical applications.",
                               " This concept is fundamental to understanding the field.",
                               " This approach has proven effective in various scenarios.",
                               " This principle guides current best practices.",
                               " This insight has led to numerous innovations."]
                variations.append(template + random.choice(conclusions))
                
                # Add variations until we reach the desired number
                for var in variations:
                    if len(texts) < num_examples:
                        texts.append(var)
                        labels.append(i)
                        category_names.append(category)
    
    # Create dataframe with the desired number of examples
    df = pd.DataFrame({
        'text': texts[:num_examples],
        'label': labels[:num_examples],
        'category': category_names[:num_examples]
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Created fallback dataset with {len(df)} examples and {len(categories)} categories")
    logger.info(f"Categories: {categories}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument("--input_file", type=str, default='data/raw/arxiv_papers.csv', 
                      help="Path to raw data CSV (default: data/raw/arxiv_papers.csv)")
    parser.add_argument("--output_file", type=str, default='data/processed/arxiv_processed.csv', 
                      help="Path to save processed data CSV (default: data/processed/arxiv_processed.csv)")
    parser.add_argument("--task_type", choices=["classification", "generation"], default="classification", 
                      help="Type of task (classification or generation)")
    parser.add_argument("--num_examples", type=int, default=1000,
                      help="Number of examples to download (default: 1000)")
    parser.add_argument("--download_dataset", action="store_true",
                      help="Download ArXiv dataset")
    parser.add_argument("--use_fallback", action="store_true",
                      help="Use fallback dataset instead of downloading")
    
    args = parser.parse_args()
    
    if args.download_dataset or not os.path.exists(args.input_file):
        if args.use_fallback:
            create_fallback_dataset(args.input_file, args.num_examples)
        else:
            # Try downloading, fall back if it fails
            success = download_arxiv_dataset(args.input_file, args.num_examples)
            if not success:
                logger.info("Falling back to synthetic dataset creation...")
                create_fallback_dataset(args.input_file, args.num_examples)
    
    logger.info(f"Processing data from {args.input_file} to {args.output_file}...")
    prepare_training_data(args.input_file, args.output_file, args.task_type)
    logger.info("Data processing complete!")
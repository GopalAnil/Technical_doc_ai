# src/fine_tune_generator.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments  # Using standard TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_seq2seq_data(data_path):
    """Load and prepare data for sequence-to-sequence fine-tuning"""
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} examples from {data_path}")
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logger.info(f"Split into {len(train_df)} training and {len(val_df)} validation examples")
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    # This is a simple metric calculation for sequence-to-sequence tasks
    predictions, labels = eval_pred
    
    # Replace -100 with pad token id
    predictions = np.where(predictions != -100, predictions, 0)
    labels = np.where(labels != -100, labels, 0)
    
    # Calculate accuracy (simplified metric)
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy
    }

def create_documentation_dataset():
    """Create a synthetic dataset of technical documentation examples"""
    # Define technical documentation examples
    docs = [
        # NumPy examples
        {
            "input": "Define NumPy library clearly and completely for beginners.",
            "target": "NumPy (Numerical Python) is a fundamental Python library for scientific computing. It provides support for large, multi-dimensional arrays and matrices, along with a comprehensive collection of high-level mathematical functions to operate on these arrays. NumPy forms the foundation of Python's scientific computing ecosystem."
        },
        {
            "input": "List key components or aspects of NumPy library.",
            "target": "1. ndarray: N-dimensional array object for efficient data storage and operations\n2. Universal Functions (ufuncs): Element-wise operations on arrays with optimized performance\n3. Broadcasting: System for applying operations to arrays of different shapes/sizes\n4. Vectorization: Enables operations on entire arrays without explicit loops\n5. Linear Algebra Functions: Matrix operations, eigenvalues, determinants, etc."
        },
        # Pandas examples
        {
            "input": "Define pandas library clearly and completely for beginners.",
            "target": "Pandas is a powerful Python data analysis library built on top of NumPy. It provides flexible, high-performance data structures for working with structured data, particularly DataFrames which are similar to tables in a relational database or spreadsheets. Pandas excels at data manipulation, cleaning, exploration, and analysis."
        },
        {
            "input": "List key components or aspects of pandas library.",
            "target": "1. DataFrame: 2D tabular, size-mutable data structure with labeled axes\n2. Series: 1D labeled array for holding any data type\n3. Index: Labels for rows and columns that enable alignment and selection\n4. GroupBy: Functionality for split-apply-combine operations\n5. I/O Tools: Functions to read and write data in various formats (CSV, Excel, SQL, etc.)"
        },
        {
            "input": "Explain how pandas library works in detail for beginners.",
            "target": "Pandas works by extending NumPy's array capabilities with database-like features. DataFrames store data in columnar format for efficiency and provide SQL-like operations for data manipulation. Pandas implements automatic data alignment with labeled axes, handles missing data through NaN values, and provides extensive time series functionality. Operations are vectorized for performance while maintaining a simple, intuitive API."
        },
        # TensorFlow examples
        {
            "input": "Define TensorFlow library clearly and completely for beginners.",
            "target": "TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and community resources for building and deploying machine learning models. TensorFlow uses data flow graphs for numerical computations, where nodes represent mathematical operations and edges represent multidimensional data arrays (tensors) that flow between them."
        },
        {
            "input": "List key components or aspects of TensorFlow library.",
            "target": "1. Tensors: Multi-dimensional arrays that are the primary data structure\n2. Computational Graphs: Representation of calculations as directed graphs\n3. Eager Execution: Immediate evaluation of operations for more intuitive development\n4. Keras API: High-level interface for building and training models\n5. SavedModel Format: Standard serialization format for sharing and deploying models"
        },
        # Python examples
        {
            "input": "Define Python programming language clearly and completely for beginners.",
            "target": "Python is a high-level, interpreted programming language known for its readability and simplicity. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with significant use of whitespace. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's comprehensive standard library and active ecosystem make it suitable for various applications from web development to data science."
        },
        {
            "input": "List key components or aspects of Python programming language.",
            "target": "1. Interpreter: Executes Python code line by line without prior compilation\n2. Dynamic Typing: Variables are assigned types at runtime rather than declaration\n3. Extensive Standard Library: Built-in modules providing diverse functionality\n4. Package Ecosystem: Rich collection of third-party packages via PyPI\n5. Memory Management: Automatic garbage collection for memory allocation"
        },
        # Git examples
        {
            "input": "Define Git version control system clearly and completely for beginners.",
            "target": "Git is a distributed version control system that tracks changes in computer files and coordinates work among multiple people. Developed by Linus Torvalds in 2005, Git allows developers to maintain a complete history of their work, create branches to work on features separately, and collaborate efficiently with others. Git's distributed nature means each developer has a full copy of the repository, enabling offline work and providing natural backup of the codebase."
        }
    ]
    
    # Create dataframe
    df = pd.DataFrame(docs)
    
    # Save to CSV
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/tech_docs_examples.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Created documentation dataset with {len(df)} examples at {output_file}")
    
    return output_file

def fine_tune_generator(
    data_path=None,
    model_name="google/flan-t5-base",
    output_dir="models/fine_tuned_generator",
    batch_size=2,
    learning_rate=5e-5,
    epochs=3,
):
    """Fine-tune a T5 model for technical documentation generation"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic dataset if not provided
    if data_path is None:
        data_path = create_documentation_dataset()
    
    # Load data
    train_dataset, val_dataset = load_seq2seq_data(data_path)
    
    # Load model and tokenizer
    logger.info(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Apply LoRA for efficient fine-tuning
    logger.info("Configuring LoRA adapter")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Tokenize the datasets
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input"], 
            padding="max_length", 
            truncation=True,
            max_length=512
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples["target"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=512
    )
    
    # Set up training arguments - SIMPLIFIED VERSION like your original code
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        eval_steps=100
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    logger.info("Evaluating fine-tuned model")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model for technical documentation")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to processed data CSV")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                       help="Pretrained model name (default: google/flan-t5-base)")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned_generator",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    
    args = parser.parse_args()
    
    fine_tune_generator(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
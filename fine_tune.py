# src/model/fine_tune.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load and prepare data for fine-tuning"""
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} examples from {data_path}")
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'] if 'label' in df.columns else None)
    
    logger.info(f"Split into {len(train_df)} training and {len(val_df)} validation examples")
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy
    }

def fine_tune(data_path, model_name="distilbert-base-uncased", output_dir="models/fine_tuned", 
              batch_size=16, learning_rate=5e-5, epochs=3):
    """Fine-tune a pretrained model on our data"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_dataset, val_dataset = load_data(data_path)
    
    # Count number of labels
    if 'label' in train_dataset.column_names:
        num_labels = len(set(train_dataset['label']))
        logger.info(f"Detected {num_labels} unique labels")
    else:
        num_labels = 2  # Default to binary classification
        logger.warning("No 'label' column found, defaulting to binary classification")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text" if "text" in examples else "processed_text"], 
            padding="max_length", 
            truncation=True,
            max_length=512
        )
    
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments - SIMPLIFIED VERSION
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        # Remove problematic parameters
        save_steps=100,
        eval_steps=100,
        # evaluation_strategy and save_strategy might be problematic depending on version
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
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
    
    parser = argparse.ArgumentParser(description="Fine-tune a model on technical documentation data")
    parser.add_argument("--data_path", type=str, default="data/processed/arxiv_processed.csv",
                       help="Path to processed data CSV")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                       help="Pretrained model name (default: distilbert-base-uncased)")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    
    args = parser.parse_args()
    
    fine_tune(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
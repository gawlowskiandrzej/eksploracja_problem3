import os

import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_from_disk
import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"

def read_datasets():
    print("Reading datasets from parquet files...")
    train_dataset = pd.read_parquet(config.traindatapath)
    val_dataset = pd.read_parquet(config.valdatapath)
    test_dataset = pd.read_parquet(config.testdatapath)
    return train_dataset, val_dataset, test_dataset

class CodeBert:
    def __init__(self, task_subset='A', max_length=512, model_name="microsoft/codebert-base", num_labels=2):
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        
    def init_model_and_tokenizer(self):
        print(f"Initializing {self.model_name} model and tokenizer...")
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
        print(f"Model initialized with {self.num_labels} labels")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        print(f"Model on: {next(self.model.parameters()).device}")

    def load_model(self, model_dir):
        print(f"Loading fine-tuned model from {model_dir}...")

        self.model = RobertaForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)

        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            padding=True, 
            max_length=self.max_length)

    def prepare_dataset(self, df):
        print("Preparing dataset for traning...")

        dataset = Dataset.from_pandas(df[['code', 'label']])

        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )

        return dataset
    
    def prepare_datasetv1(self, train_df, val_df, num_proc=1):
        print("Preparing dataset for training...")
        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=num_proc
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=num_proc
        )
        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

        return {
            "accuracy": accuracy, 
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        print("Starting training...")

        print(f"Total training samples: {len(train_dataset)}")
        print(f"Total validation samples: {len(val_dataset)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            # logging_dir='./logs',
            logging_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
            report_to="none",
            run_name="codebert_2e-5"
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()

        trainer.save_model(output_dir + "/final_model")
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed. Model saved to {output_dir}")
        
        return trainer

    def evaluate_model(self, trainer, val_dataset):
        print("Evaluating model...")
        
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        return predictions

def train_model(num_labels, train_data, val_data):
    codebert_base = CodeBert(max_length=512, model_name="microsoft/codebert-base", num_labels=num_labels)
    codebert_base.init_model_and_tokenizer()
    train_dataset, val_dataset = codebert_base.prepare_datasetv1(train_data, val_data)
    train_dataset.save_to_disk("tokenized_train_dataset")
    val_dataset.save_to_disk("tokenized_val_dataset")

    trainer = codebert_base.train(
        train_dataset,
        val_dataset,
        output_dir="./working/",
        num_epochs=1,
        batch_size=16,
        learning_rate=2e-5
    )
    return trainer

def test_model(test_data, val_data, trainer= None):
    code_bert = CodeBert()
    code_bert.load_model("./working/final_model")
    val_dataset = code_bert.prepare_dataset(val_data)
    test_dataset = code_bert.prepare_dataset(test_data)
    data_collator = DataCollatorWithPadding(tokenizer=code_bert.tokenizer)

    if (trainer is None):
        trainer = Trainer(
            model=code_bert.model,
            data_collator=data_collator,
            processing_class=code_bert.tokenizer,
            compute_metrics=code_bert.compute_metrics
    )
    print("Evaluating on evaluation set...")
    predictions = trainer.predict(val_dataset)

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['human', 'machine'], digits=5))

    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['human', 'machine'], digits=5))

def create_model():
    datasets = read_datasets()
    train_data, val_data, test_data = datasets
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    print(f"Loaded {len(test_data)} testing samples")

    num_labels = train_data['label'].nunique()
    print(num_labels)
    
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    print("Starting model training...")
    trainer = train_model(num_labels, train_data, val_data)
    test_model(test_data, val_data, trainer)

if __name__ == "__main__":
    create_model()
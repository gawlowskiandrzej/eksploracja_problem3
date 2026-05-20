import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_from_disk
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

def read_datasets():
    train_dataset = pd.read_parquet(config.traindatapath)
    val_dataset = pd.read_parquet(config.valdatapath)
    test_dataset = pd.read_parquet(config.testdatapath)
    return train_dataset, val_dataset, test_dataset

class BaseModel:
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
        print(f"Model initialized with {self.num_labels} labels")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

    def prepare_dataset(self, train_df, val_df, num_proc=4):
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

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        print("Starting training...")
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Total validation samples: {len(val_dataset)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,         # log mỗi 100 step
            eval_strategy="steps",
            eval_steps=10000,          # valid mỗi 10.000 step
            save_strategy="steps",
            save_steps=10000,            # save mỗi 10.000 step
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=4
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.save_model()
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
def create_model():
    number_of_labels = 2
    codebert_base = BaseModel(max_length=512, model_name="microsoft/codebert-base", num_labels=number_of_labels)
    codebert_base.init_model_and_tokenizer()
    datasets = read_datasets()
    train_dataset, val_dataset = codebert_base.prepare_dataset(datasets[0], datasets[1])
    train_dataset.save_to_disk("tokenized_train_dataset")
    val_dataset.save_to_disk("tokenized_val_dataset")
    train_dataset = load_from_disk("/kaggle/input/notebook8a7e039295/tokenized_train_dataset")
    val_dataset = load_from_disk("/kaggle/input/notebook8a7e039295/tokenized_val_dataset")

    trainer = codebert_base.train(
    train_dataset.select(range(250000)),
    val_dataset.select(range(50000)),
    output_dir="/kaggle/working/",
    num_epochs=1,
    batch_size=2,
    learning_rate=2e-5
    )

if __name__ == "__main__":
    create_model()
import pandas as pd
import os
import numpy as np
from datetime import datetime

from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold, ParameterGrid

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import Trainer, TrainingArguments

train_val = pd.read_csv('./data/train_cleaned.csv')

train_val.columns


X = list(train_val['cleaned_full_text'])
y = list(train_val['score'])

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
config = RobertaConfig.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

# Freeze RoBERTa parameters
for param in model.parameters():
    param.requires_grad = False

# Define dataset class
class RegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = float(self.targets[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # return_tensor=True # instead of 'pt', use True
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"]).flatten(),
            "attention_mask": torch.tensor(encoding["attention_mask"]).flatten(),
            "targets": torch.tensor(target, dtype=torch.float)
        }


# Modify RoBERTa model for regression (add a regression head)
class RobertaRegressor(nn.Module):
    def __init__(self, config):
        super(RobertaRegressor, self).__init__()
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # the pooled output - the entire input obtained by applying pooling operations over the hidden states
        pooled_output = outputs.pooler_output
        
        # regularization via dropout
        pooled_output = self.dropout(pooled_output)
        
        # pass the pooled output through a linear layer
        logits = self.linear(pooled_output)
        
        # return the prediction
        return logits


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.prediction.flatten()
    return {"mse": mean_squared_error(label, preds)}

# Initialize Trainer
class RobertaTrainer(Trainer):
    def __init__(self, model, tokenizer, args, train_dataset=None, eval_dataset=None, compute_metrics=None):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
        self.tokenizer = tokenizer
        self.loss_fn = nn.MSELoss()
        
    def training_step(self, model, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        targets = inputs["targets"]
        
        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = self.loss_fn(logits, targets)
        
        return loss

# Define hyperparameter grid
hyperparameters = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "num_train_epochs": [3, 4, 5],
    "per_device_train_batch_size": [4],
    "per_device_eval_batch_size": [4],
    "max_len": [32, 64, 128]
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
log_file_path = './log/hyperparameter_log.txt'

best_score = None
best_params = None

for params in ParameterGrid(hyperparameters):
    print("Hyperparameters:", params)
    scores = []
    
    for train_index, val_index in kf.split(X):
        train_texts, val_texts = [X[i] for i in train_index], [X[i] for i in val_index]
        train_targets, val_targets = [y[i] for i in train_index], [y[i] for i in val_index]
        
        model = RobertaRegressor(config)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        training_args = TrainingArguments(
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            num_train_epochs=params['num_train_epochs'],
            evaluation_strategy='epoch',
            logging_dir='./logs',
            output_dir='./output'
        )
        
        # Prepare training and validation datasets
        train_dataset = RegressionDataset(train_texts, train_targets, tokenizer, params["max_len"])
        val_dataset = RegressionDataset(val_texts, val_targets, tokenizer, params["max_len"])
        
        trainer = RobertaTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()
        
        # Evaluate model
        eval_metrics = trainer.evaluate()
        mse_score = eval_metrics["eval_mse"]
        
        scores.append(mse_score)
        
    # Calculate average score across folds
    avg_score = np.mean(scores)
    print("Average Score:", avg_score)
        
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log the performace for the hyperparameter setting
    with open(log_file_path, "a") as log_file:
        # Write hyperparameters and performance metrics to log file
        log_file.write(f"Hyperparameters: {hyperparameters}\n")
        log_file.write(f"MSE Score: {avg_score}\n")
        log_file.write("\n")  # Add newline for readability

    # Close log file
    log_file.close()
        
    if best_score is None or avg_score > best_score:
        best_score = avg_score
        best_params = params
        
# Print best hyperparameters and score
print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)



import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.optim import AdamW as TorchAdamW 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F


def resample_data(df, undersample_threshold, oversample_threshold):

    resampled_dfs = []

    counts = df['new_category'].value_counts()

    for sub_cat, count in counts.items():
        sub_cat_data = df[df['new_category'] == sub_cat]

        if count > undersample_threshold:
            undersampled_data = sub_cat_data.sample(undersample_threshold, random_state=42)
            resampled_dfs.append(undersampled_data)
        elif count < oversample_threshold:
            oversampled_data = sub_cat_data.sample(oversample_threshold, replace=True, random_state=42)
            resampled_dfs.append(oversampled_data)
        else:
            resampled_dfs.append(sub_cat_data)

    balanced_df = pd.concat(resampled_dfs, ignore_index=True)
    
    return balanced_df


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    average_loss = np.mean(losses)

    return {'accuracy': accuracy, 'loss': average_loss}

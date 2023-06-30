import argparse
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    return p.parse_args()

sg = parse_args()

train_tuple = pd.read_csv(sg.train)
dev_tuple = pd.read_csv(sg.dev)
test_tuple = pd.read_csv(sg.test)


def convert_to_list(df):
    sentences = []
    
    for i, row in df.iterrows():
        try: 
            sentences.append(eval(row['sentence']))
        except:
            continue
        
    return sentences


train_list = convert_to_list(train_tuple)
dev_list = convert_to_list(dev_tuple)
test_list = convert_to_list(test_tuple)


def get_labels(data):
    labels = set()
    for sentence in data:
        for _, tag in sentence:
            labels.add(tag)
    labels = sorted(list(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    return label2id


label_map = get_labels(train_list + dev_list + test_list)

# Remove
print(label_map)


# Load the XLM-Roberta tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large')

# Modify the last layer of the model to have 10 output labels
num_labels = len(label_map)
model.num_labels = num_labels
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_labels)

# Load your training, validation, and test data in a format that the model can understand
# The data should be in the format of list of sentences where each sentence is a list of (word, label) pairs
train_data = train_list
val_data = dev_list
test_data = test_list


# Convert the data into a format that XLM-Roberta can read
def convert_data_to_features(data, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []
    pad_label_id = -100
    for sentence in data:
        token_ids = []
        label_ids = []
        for word, label in sentence:
            tokenized_word = tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(tokenized_word)
            label_ids.extend([label_map[label]] * len(tokenized_word))
        
        # Add special tokens and pad the sequences to a fixed length
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        label_ids = [pad_label_id] + label_ids + [pad_label_id]
        attention_mask = [1] * len(token_ids)
        padding_length = tokenizer.model_max_length - len(token_ids)
        token_ids = token_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        label_ids = label_ids + [pad_label_id] * padding_length
        
        input_ids.append(token_ids)
        attention_masks.append(attention_mask)
        labels.append(label_ids)
    
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)


train_inputs, train_masks, train_labels = convert_data_to_features(train_data, tokenizer)
val_inputs, val_masks, val_labels = convert_data_to_features(val_data, tokenizer)
test_inputs, test_masks, test_labels = convert_data_to_features(test_data, tokenizer)

# Create PyTorch DataLoader objects to efficiently load and batch the data
batch_size = 2
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model onto the device
model.to(device)

# Set up the training parameters
learning_rate = 2e-5
adam_epsilon = 1e-8
epochs = 30

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Train the model
model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        #input_ids, attention_mask, labels = batch
        # torch.cuda.synchronize()
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
#         print('shape of input ids', len(input_ids), len(input_ids[0]))
#         print('the input ids look like', input_ids)
#         print('shape of attention_mask', len(attention_mask), len(attention_mask[0]))
#         print('the attention_mask look like', attention_mask)
#         print('shape of labels', len(labels), len(labels[0]))
#         print('the labels look like', labels)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()


# Evaluate the model on the validation set
model.eval()
predictions, true_labels = [], []
for batch in val_dataloader:
    #input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Convert logits to label predictions
    predicted_labels = torch.argmax(logits, dim=2)
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    

    # Remove padding and add to overall predictions and true labels
    for i in range(len(labels)):
        preds = []
        true = []
        for j in range(len(labels[i])):
        # IF NOT WORKING, REVERT -100 to 0
            if labels[i][j] != -100:
                preds.append(predicted_labels[i][j])
                true.append(labels[i][j])
        predictions.append(preds)
        true_labels.append(true)
        

# Remove
#print('the gold labels are', true_labels)
#print('the predicted labels are', predictions)


# Compute evaluation metrics
predictions_flat = [label for sentence in predictions for label in sentence]
true_labels_flat = [label for sentence in true_labels for label in sentence]


#print('true labels flat looks like', true_labels_flat)
#print('predictions flat looks like', predictions_flat)

for index in range(len(true_labels_flat) - 1, -1, -1):
    if true_labels_flat[index] == -100:
        del true_labels_flat[index]
        del predictions_flat[index]
    elif true_labels_flat[index] == label_map['O']:
        del true_labels_flat[index]
        del predictions_flat[index]


f1 = f1_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
precision = precision_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
recall = recall_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
f1_macro = f1_score(true_labels_flat, predictions_flat, average='macro', zero_division=1)

true_labels_flat_string = [key for value in true_labels_flat for key, val in label_map.items() if val == value]
predictions_flat_string = [key for value in predictions_flat for key, val in label_map.items() if val == value]

# Calculate per-label metrics
classification_rep = classification_report(true_labels_flat_string, predictions_flat_string, zero_division=1)

print("metrics for validation")
print("F1 score:", f1)
print("Precision:", precision)
print("Recall:", recall)
#print("true labels", true_labels_flat)
#print("predictions:", predictions_flat)
print("Macro F1 score:", f1_macro)
print("Per-label Metrics:")
print(classification_rep)


# Test the model on the test set
predictions, true_labels = [], []
for batch in test_dataloader:
#     input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # Change if not working
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         labels = labels.to(device)
        logits = outputs.logits
    
    # Convert logits to label predictions
    predicted_labels = torch.argmax(logits, dim=2)
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Remove padding and add to overall predictions and true labels
    for i in range(len(labels)):
        preds = []
        true = []
        for j in range(len(labels[i])):
            # IF NOT WORKING, REVERT -100 TO 0
            if labels[i][j] != -100:
                preds.append(predicted_labels[i][j])
                true.append(labels[i][j])
        predictions.append(preds)
        true_labels.append(true)


# Compute evaluation metrics
predictions_flat = [label for sentence in predictions for label in sentence]
true_labels_flat = [label for sentence in true_labels for label in sentence]


for index in range(len(true_labels_flat) - 1, -1, -1):
    if true_labels_flat[index] == -100:
        del true_labels_flat[index]
        del predictions_flat[index]
    elif true_labels_flat[index] == label_map['O']:
        del true_labels_flat[index]
        del predictions_flat[index]
        

f1 = f1_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
precision = precision_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
recall = recall_score(true_labels_flat, predictions_flat, average='weighted', zero_division=1)
f1_macro = f1_score(true_labels_flat, predictions_flat, average='macro', zero_division=1)

true_labels_flat_string = [key for value in true_labels_flat for key, val in label_map.items() if val == value]
predictions_flat_string = [key for value in predictions_flat for key, val in label_map.items() if val == value]

# Calculate per-label metrics
classification_rep = classification_report(true_labels_flat_string, predictions_flat_string, zero_division=1)

print("metrics for test")
print("F1 score:", f1)
print("Precision:", precision)
print("Recall:", recall)
#print("true labels", true_labels_flat)
#print("predictions:", predictions_flat)
print("Macro F1 score:", f1_macro)
print("Per-label Metrics:")
print(classification_rep)


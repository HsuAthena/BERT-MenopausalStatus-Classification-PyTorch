from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict, load_metric
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import random
from sklearn.metrics import classification_report


# Checking for device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to set seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(255)


# Setting up hyperparameters and configurations
NUM_EPOCHS = 100
BATCH_SIZE = 8 
PATIENCE = 5
LEARNING_RATE = 2e-5 
WEIGHT_DECAY = 0.01
METRIC = 'eval_loss'
EVAL_STRATEGY = 'steps'

# Get the max length of the tokenizer
max_length = 512
# Defining the model checkpoint
checkpoint = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
# Get the tokenizer for this pretrained model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Specifify the location to save the model
output_dir = '/content/drive/MyDrive/menopause_status_model/model'


# Source for the annotated data
source_data_filename = '/content/MenopauseStatusResult.csv'
# Setting up hyperparameters and configurations
source_ds = load_dataset('csv', data_files=source_data_filename,encoding='ISO-8859-1')
# Converting the dataset to pandas DataFrame for data exploration
source_df = source_ds['train'].to_pandas()
# Encoding the labels as integers for the model
source_ds = source_ds.class_encode_column('labels')

# Extracting features and labels from the dataset
features = source_ds['train'].features
# Get a set of the unique labels in this dataset
label_ids = list(set(source_ds['train']['labels']))
labels = [features['labels'].int2str(idx) for idx in label_ids]
# Get the number of unique labels
num_labels = len(labels)
# Map the IDs to labels dictionary
id2label = {idx:features['labels'].int2str(idx) for idx in range(num_labels)}
# Map the labels to IDs dictionary
label2id = {v:k for k, v in id2label.items()}

# Calculating class weights to handle class imbalance
class_weights = (1- (source_df['labels'].value_counts().sort_index() / len(source_df))).values
# Change the weights to Pytorch tensors and send to GPU
class_weights = torch.from_numpy(class_weights).float().to(device)


# Splitting the dataset into train(70%), validation(15%), and test(15%) sets.
# All the 70-30 split, then 15-15 split from test dataset
train_ds = source_ds['train'].train_test_split(test_size=0.3)
# Split the test in half for 15-15 test validation
test_val = train_ds['test'].train_test_split(test_size=0.5)
# Join all of the datasets back together
source_ds = DatasetDict({
  'train': train_ds['train'],
  'test': test_val['test'],
  'validation': test_val['train']
})


# Calculate the logging steps, which is once per epoch
logging_steps = len(source_ds['train']) // BATCH_SIZE


# Create the training arguments
training_args = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = NUM_EPOCHS,
    learning_rate = LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    weight_decay = WEIGHT_DECAY,
    evaluation_strategy = EVAL_STRATEGY,
    logging_steps = logging_steps,
    fp16 = True,
    push_to_hub = False,
    eval_steps = logging_steps,
    save_total_limit = PATIENCE,
    save_steps = logging_steps,
    load_best_model_at_end = True)


#Tokenizes a record in the dataset, with token ids, attention mask, and token type ids
def tokenize_function(example):

  return tokenizer(example["text"], truncation=True, max_length=max_length)

# Tokenize the dataset
tokenized_ds = source_ds.map(tokenize_function, batched=True)
tokenized_ds = tokenized_ds.remove_columns(["text"])


# Function to compute metrics for evaluation
def compute_metrics(pred):

  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average='weighted')
  accuracy = accuracy_score(y_true=labels, y_pred=preds)

  return {"accuracy": accuracy, "f1": f1}


# Create the model from the checkpoint
def model_init():

  return AutoModelForSequenceClassification.from_pretrained(
      checkpoint,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id
    ).to(device)


# calculate the loss
class WeightedLossTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):

    outputs = model(**inputs)
    logits = outputs.get('logits')
    labels = inputs.get('labels')
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_func(logits, labels)

    return (loss, outputs) if return_outputs else loss


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

# Train
trainer.train()


# Making predictions on the test set
predictions = trainer.predict(tokenized_ds["test"])
preds = np.argmax(predictions.predictions, axis=-1)


# Preparing the test DataFrame
test_df = pd.DataFrame(source_ds['test'])
test_df['preds'] = preds
test_df['labels_text'] = [id2label[id] for id in test_df.labels]
test_df['preds_text'] = [id2label[id] for id in test_df.preds]
print(classification_report(test_df.labels_text, test_df.preds_text))
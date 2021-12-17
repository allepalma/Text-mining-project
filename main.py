import os
from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor
from models.models import *
from models.config import CustomBertConfig
from torch import nn


seed = 13
dataset_file = 'dataset.txt'
max_length = 512

# Create dataset if it has not been created yet
if not os.path.isfile(dataset_file):
    data_dir = 'cadec'
    c = DatasetCreator()
    c.create_dataset(data_dir, dataset_file)


###### Example on how to use the BERT model with linear head for prediction ######
# Class containing the data
data_loader = DataProcessor(filename=dataset_file, model='bert-base-uncased', seed=seed, max_length=max_length,
                            batch_size=8)
print(data_loader.label2id)
# Configure parameters
config = CustomBertConfig(clf_type='baseline', num_labels=9, embedding_size=256, vocab_size=30522, hidden_size=128)

# Initialize model
model = Baseline(config)
batch = [b for b in data_loader.train_dataloader][0]

# Predict
res = model(input_ids=batch[0], attention_mask = batch[1], labels = batch[2])

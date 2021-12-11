import os
from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor
from models.models import *
from models.configuration_bert import CustomBertConfig
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
# Configure parameters
config = CustomBertConfig(model='bert-base-uncased', clf_type='linear', num_labels=11,
                          dropout=0.1, hidden_size=768, num_clf_hidden_layers=0, num_neurons=(),
                          activation=nn.ReLU)

# Initialize model
bert_model = BertCRF(config)
batch = [b for b in data_loader.train_dataloader][0]

# Predict
res = bert_model(input_ids=batch[0], attention_mask = batch[1], labels = batch[2])

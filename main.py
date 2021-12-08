import os

from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor

seed = 13
dataset_file = 'dataset.txt'
max_length = 512

# Create dataset if it has not been created yet
if not os.path.isfile(dataset_file):
    data_dir = 'cadec'
    c = DatasetCreator(max_length=max_length, split_messages=True)
    c.create_dataset(data_dir, dataset_file)

data_processor = DataProcessor(filename=dataset_file, model='bert-base-uncased', seed=seed, max_length=max_length)

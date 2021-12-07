import os

from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor

dataset_file = 'dataset.txt'

# Create dataset if it has not been created yet
if not os.path.isfile(dataset_file):
    data_dir = 'cadec'
    c = DatasetCreator()
    c.create_dataset(data_dir, dataset_file)

data_processor = DataProcessor(filename=dataset_file, model = 'bert-base-uncased', seed=13, max_length = 512)

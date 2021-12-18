import os
from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor
from traintest import TrainTest
from models.models import *
from models.config import CustomBertConfig
from torch import nn


def baseline(seed, max_length, dataset_file):
    # Class containing the data
    data_loader = DataProcessor(filename=dataset_file, model='bert-base-uncased', seed=seed, max_length=max_length,
                                batch_size=4)
    # Configure parameters
    config = CustomBertConfig(clf_type='baseline', num_labels=9, embedding_size=256, vocab_size=30522, hidden_size=128)
    # Initialize model
    bert_model = Baseline(config)
    # Train and test the model
    TrainTest(bert_model, 'Baseline', data_loader, learning_rate=1e-3)

def bertmodels(seed, max_length, dataset_file):
    transformers = ['bert-base-uncased', 'dmis-lab/biobert-v1.1']
    heads = {
        'Linear': BertLinear,
        'CRF': BertCRF,
        'LSTM': BertLSTM
    }
    for transformer in transformers:
        for model in heads.keys():
            # Class containing the data
            data_loader = DataProcessor(filename=dataset_file, model=transformer, seed=seed, max_length=max_length,
                                        batch_size=4)
            # Configure parameters
            config = CustomBertConfig(model=transformer, clf_type='linear', num_labels=9,
                                    dropout=0.1, hidden_size=768, num_clf_hidden_layers=0, num_neurons=(),
                                    activation=nn.ReLU)
            # Initialize model
            bert_model = heads[model](config)
            # Train and test the model
            model_name = f'{transformer} + {model}'
            TrainTest(bert_model, model_name, data_loader)


if __name__ == '__main__':
    seed = 13
    dataset_file = 'dataset.txt'
    max_length = 512

    # Create dataset if it has not been created yet
    if not os.path.isfile(dataset_file):
        data_dir = 'cadec'
        c = DatasetCreator()
        c.create_dataset(data_dir, dataset_file)
    
    baseline(seed, max_length, dataset_file)
    # bertmodels(seed, max_length, dataset_file)
from utils import *
from dataset_creation import DatasetCreator
from embedding_extractor import EmbeddingExtractor

import os

seed = 13
dataset_file = 'dataset.txt'
max_length = 512

# Define which methods to execute
train_baseline = False
train_berts = False
extract_embeddings = True
show_tsne = True

# Create dataset if it has not been created yet
if not os.path.isfile(dataset_file):
    data_dir = 'cadec'
    DatasetCreator(data_dir, dataset_file)

# Create necessary/missing directories (saved_models/logging/embeddings)
create_dirs()

# Train and test baseline
if train_baseline:
    baseline(seed, max_length, dataset_file)

# Train and test BERT models
if train_berts:
    bertmodels(seed, max_length, dataset_file)


# Specify best performing BERT model
best_bert = 'emilyalsentzer/Bio_ClinicalBERT'
best_transformer = 'BioClinicalBert'
best_head = 'CRF'
model_path = os.path.join('saved_models','BioClinicalBert_CRF')

# Initialize dataloader for best BERT
data_loader = DataProcessor(filename=dataset_file, model=best_bert, seed=seed, max_length=max_length,
                                    batch_size=4)

# Extract and save BERT embeddings
if extract_embeddings:
    EmbeddingExtractor(best_transformer, best_head, model_path, dataset_file, seed)

# Calculate/get t-SNE and plot
# (Can only be done with either saved BERT embeddings or saved t-SNE embeddings)
if show_tsne:
    datatypes = ['before', 'after']
    for datatype in datatypes:
        # Calculate t-SNE and save embeddings in embedding folder (if they don't exist yet)
        reduced_embs, labels, dataset_encoding = get_tsne(datatype, save=True)
        # Plot t-SNE
        plot_tsne(data_loader, reduced_embs, labels, dataset_encoding, datatype)

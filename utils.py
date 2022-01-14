from bert_data_creation import DataProcessor
from traintest import TrainTest
from models.models import *
from models.config import CustomBertConfig

import os
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

def create_dirs():
    # Create directory to store trained models
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    # Create directory to store logging during training
    if not os.path.exists('logging'):
        os.makedirs('logging')
    # Create directory to store BERT embeddings and t-SNE
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

def baseline(seed, max_length, dataset_file):
    # Class containing the data
    data_loader = DataProcessor(filename=dataset_file, model='bert-base-uncased', seed=seed, max_length=max_length,
                                batch_size=4)
    # Configure parameters
    config = CustomBertConfig(clf_type='baseline', num_labels=9, embedding_size=256, vocab_size=30522, hidden_size=128)
    # Initialize model
    bert_model = Baseline(config)
    # Train and test the model
    TrainTest(bert_model, 'Baseline', data_loader, learning_rate=0.001)

def bertmodels(seed, max_length, dataset_file):
    transformers = {
        'Bert': 'bert-base-uncased', 
        'BioBert': 'dmis-lab/biobert-v1.1', 
        'BioClinicalBert': 'emilyalsentzer/Bio_ClinicalBERT'
        }
    heads = {
        'Linear': BertLinear,
        'CRF': BertCRF,
        'LSTM': BertLSTM
    }
    for t_name, transformer in transformers.items():
        for h_name, head in heads.items():
            # Class containing the data
            data_loader = DataProcessor(filename=dataset_file, model=transformer, seed=seed, max_length=max_length,
                                        batch_size=4)
            # Configure parameters
            config = CustomBertConfig(model=transformer, clf_type='linear', num_labels=9,
                                      dropout=0.1, hidden_size=768, num_clf_hidden_layers=0, num_neurons=(),
                                      activation=nn.ReLU)
            # Initialize model
            bert_model = head(config)
            # Train and test the model
            model_name = f'{t_name}_{h_name}'
            TrainTest(bert_model, model_name, data_loader)

def get_tsne(datatype, save=False):
    
    # Read t-SNEmbeddings if they are saved
    tsne_path = os.path.join('embeddings', f'tsne_{datatype}_tuning.pkl')
    if os.path.exists(tsne_path):
        with open(tsne_path, 'rb') as f:
            data = pkl.load(f)
        reduced_embs = data[0]
    # Calculate t-SNE (and save if specified)
    else:
        print(f'Calculating t-SNE from BERT embeddings {datatype} fine-tuning...')
        with open(os.path.join('embeddings', f'data_{datatype}_tuning.pkl'), 'rb') as f:
            data = pkl.load(f)
        reduced_embs = TSNE(n_components=2).fit_transform(data[0])
        if save:
            obj = [reduced_embs, data[1], data[2]]
            with open(tsne_path, 'wb') as f:
                pkl.dump(obj, f)
    return reduced_embs, data[1], data[2]

def plot_tsne(dataloader, reduced_embs, labels, dataset_encoding, datatype):
    # Get labels as strings and order them for the legend
    str_labels, hue_order = [], []
    for label in labels:
        str_labels.append(dataloader.id2label[label])
    for label in np.unique(labels):
        hue_order.append(dataloader.id2label[label])
    # Define colors
    colors = [
        'darkgreen', 'chartreuse',
        'blue', 'aqua',
        'darkviolet', 'violet',
        'maroon', 'red'
    ]
    # Plot
    plt.figure(figsize=[12.8, 9.6])
    plt.rcParams['font.size'] = '16'
    p = sns.scatterplot(
        x = reduced_embs[:,0],
        y = reduced_embs[:,1],
        hue = str_labels,
        hue_order = hue_order,
        palette = colors,
        s = 15,
        # style = dataset_encoding,
        alpha = 0.6,
        legend = 'full'
    )
    title = f't-SNE {datatype} fine-tuning'
    p.set_title(title, fontsize = 30)
    p.set(xlabel = 'Embedding 1', ylabel = 'Embedding 2')
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.tight_layout()
    plt.show()

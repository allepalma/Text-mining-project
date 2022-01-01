import torch
import sklearn
import numpy as np
import seaborn
import os
from dataset_creation import DatasetCreator
from bert_data_creation import DataProcessor
from models.models import *
from models.config import CustomBertConfig
from torch import nn
import pickle as pkl

class EmbeddingExtractor:
    def __init__(self, bert_model, clf_head, model_path, data_path, seed):
        """
        Class extracting the embeddings from named entities
        :param bert_model: the name of the bert model to use (Bert, BioBert, BioClinicalBert)
        :param clf_head: the name of the classification head to use (Linear, CRF, LSTM)
        :param model_path: the path with the weights of the trained model
        :param data_path: the path with the dataset
        """
        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.bert_model = bert_model
        self.clf_head = clf_head
        # The transformers and the heads
        self.transformers = {'Bert': 'bert-base-uncased',
                             'BioBert': 'dmis-lab/biobert-v1.1',
                             'BioClinicalBert': 'emilyalsentzer/Bio_ClinicalBERT'}
        # The Bert model options
        self.heads = {'Linear': BertLinear,
                      'CRF': BertCRF,
                      'LSTM': BertLSTM}
        self.model_path = model_path
        self.data_path = data_path
        self.seed = seed

        # Create the data object
        self.data_object = self.load_data()

        # Initialize non-fine-tuned model
        self.config = CustomBertConfig(model=self.transformers[self.bert_model], clf_type='linear', num_labels=9,
                                       dropout=0.1, hidden_size=768, num_clf_hidden_layers=0, num_neurons=(),
                                       activation=nn.ReLU)
        self.model = self.heads[self.clf_head](self.config).to(self.device)
        self.embeddings_to_plot_pre_training, self.labels_to_plot_pre_training, self.ids_to_plot_pre_training = self.extract_embeddings()

        # Parametrize the model
        self.parametrize_model()

        # Extract embeddings after parametrization
        self.embeddings_to_plot_post_training, self.labels_to_plot_post_training, self.ids_to_plot_post_training = self.extract_embeddings()

        # Save the objects
        self.save_embeddings(self.embeddings_to_plot_pre_training, self.embeddings_to_plot_pre_training, self.embeddings_to_plot_pre_training,
                             'data_before_tuning.pkl')

        self.save_embeddings(self.embeddings_to_plot_post_training, self.embeddings_to_plot_post_training, self.embeddings_to_plot_post_training,
                             'data_after_tuning.pkl')

    def load_data(self):
        """
        Given the path of the data, it loads the data into a loader class
        :return: Torch loader class containing the data
        """
        #Check if the dataset is already in the folder
        if not os.path.exists(self.data_path):
            data_dir = 'cadec'
            c = DatasetCreator()
            c.create_dataset(data_dir, self.data_path)
        # The data object
        data_object = DataProcessor(self.data_path, self.transformers[self.bert_model], seed=self.seed,
                                    batch_size=4, max_length=512)
        return data_object

    def extract_embeddings(self):
        """
        Extract the embeddings given the model and process them such that:
        - The embeddings associated to O labels are not considered
        - The embeddings of word pieces of the same word are averaged
        :return: The processed embeddings of training, test and validation sets
        """
        dataset_id = 0  # 0 for training, 1 for validation and 2 for testing
        embeddings_to_plot, labels_to_plot, dataset_encoding = [], [], []  # Contain the final embeddings
        # For each data loader
        for loader in [self.data_object.train_dataloader, self.data_object.val_dataloader, self.data_object.test_dataloader]:
            # For all the batches in the loader
            for batch in loader:
                # Inputs to the BERT model
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                word_ids = batch[3].to(self.device)  # The ids of the words before breaking them into pieces (used for averaging)
                # Apply the model
                with torch.no_grad():
                    output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                # For each batch, fetch the last embedding layer
                last_layer_embeddings = output['hidden_states'][12]
                # For each embedding of the batch, perform the extraction and averaging of the embeddings
                for i in range(last_layer_embeddings.shape[0]):
                    # Get the evaluated embeddings, labels and ids
                    emb = last_layer_embeddings[i, :, :]
                    lab = labels[i, :]
                    ids = word_ids[i, :]
                    # Derive the embeddings not associated to an O's and averaged by word piece
                    embeddings_to_keep, labels_to_keep = self.select_embeddings(emb, lab, ids)
                    embeddings_to_plot += embeddings_to_keep
                    labels_to_plot += labels_to_keep
            # Add the dataset encoding (0 for training, 1 for valid, 2 for test)
            dataset_encoding += [dataset_id for i in range(len(embeddings_to_plot))]
            dataset_id += 1
        return np.array(embeddings_to_plot), np.array(labels_to_plot), np.array(dataset_encoding)

    def select_embeddings(self, embeddings, labels, word_ids):
        """
        Averages across word pieces and excludes O's embeddings
        :param layer_embeddings: The flattened embeddings by BERT
        :param labels: The flattened labels associated to the embedding observations
        :param word_ids: The flattened ids of the words
        :return: The embeddings selected to be represented and the associated labels
        """
        # Lists of embeddings without O's and averaged across word pieces
        embeddings_to_keep = []
        labels_to_keep = []
        # The maximum ID of the words
        max_word = max(word_ids)
        # For each original word id, fetch all embeddings of the word pieces corresponding to it and average them
        for i in range(1, max_word):
            # Keep only the embeddings whose label is different from 0 and that correspond to the ith original word
            mask = torch.where(torch.logical_and(word_ids == i, labels != 0))
            wordpiece_embeddings = embeddings[mask]
            label_wordpiece = labels[mask]
            # Average different word pieces
            if len(wordpiece_embeddings) > 0:
                mean_wordpiece_embedding = wordpiece_embeddings.mean(dim=0)
                # Store the results
                embeddings_to_keep.append(mean_wordpiece_embedding.tolist())
                labels_to_keep.append(label_wordpiece[0].item())
        return embeddings_to_keep, labels_to_keep

    def parametrize_model(self):
        """
        Given the path of the weights, it parameterizes the BERT model using them
        """
        self.model.load_state_dict(torch.load(self.model_path))  # Add the weights to Bert

    def save_embeddings(self, embeddings, labels, sentences, path):
        """
        Saves the embeddings
        """
        obj = [embeddings, labels, sentences]
        with open(path, 'wb') as file:
            pkl.dump(obj, file)




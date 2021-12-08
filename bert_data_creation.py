import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

'''
Read the data from a pre-processed CADEC dataset and process them into a format compatible with BERT
'''

class DataProcessor():
    '''
    Loads the data from a pre-processed CADEC named-entity dataset and creates a BERT dataset
    '''
    def __init__(self, filename, model, seed, batch_size = 32, max_length = 512):
        # Set the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize attribute variables
        self.max_length = max_length
        self.filename = filename
        self.seed = seed  # For test and train split
        self.model = model
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        print('Parsing the data file...')
        # Obtain sentences and labels
        self.tokens, self.labels = self.sentence_parser()

        # Split sentences if their associated wordpiece encoding is longer than max_length
        self.split_tokens, self.split_labels = [], []
        for tok, lab in zip(self.tokens, self.labels):
            split_tok, split_lab = self.split_sentences(tok, lab)
            self.split_tokens.extend(split_tok)
            self.split_labels.extend(split_lab)

        # Create ids for labels and split into training and test set
        self.label2id = self.get_label_encoding_dict()  # Initialize mapping of labels to ids
        self.tokens_train, self.tokens_test, self.labels_train, self.labels_test = self.train_test_split()

        # Tokenize for BERT
        # Training set
        self.tokenized_input_train = self.tokenizer(self.tokens_train, truncation=True, is_split_into_words=True,
                                                    add_special_tokens=True, padding='max_length', max_length=self.max_length)
        self.train_tags = self.get_bert_labels(self.tokenized_input_train, self.labels_train)

        # Test set
        self.tokenized_input_test = self.tokenizer(self.tokens_test, truncation=True, is_split_into_words=True,
                                                    add_special_tokens=True, padding='max_length', max_length=self.max_length)
        self.test_tags = self.get_bert_labels(self.tokenized_input_test, self.labels_test)

        # Prepare the data so it is compatible with torch
        self.y_train = torch.tensor(self.train_tags).to(self.device)
        self.y_test = torch.tensor(self.test_tags).to(self.device)

        self.train_dataloader = self.create_data_loaders(self.tokenized_input_train, self.y_train)
        self.test_dataloader = self.create_data_loaders(self.tokenized_input_test, self.y_test)

    def sentence_parser(self):
        '''
        Read the content of filename and parses it into labels and tokens
        :return: tokens and labels: two lists containing the tokens and the labels in the dataset
        '''
        with open(self.filename, 'r') as f:
            data_raw = f.read()
        sentences = [sent.split('\n') for sent in data_raw.split('\n\n')[:-1]]
        tokens = [[pair.split('\t')[0] for pair in sent] for sent in sentences]
        labels = [[pair.split('\t')[1] for pair in sent] for sent in sentences]
        return tokens, labels

    def split_sentences(self, sentence, labels):
        '''
        Read the tokenized sentences and split them if they are longer than a maximum length
        :param: An input tokenized sentence
        :return: The tokenized sentence
        '''
        # The BERT encoding of the period token
        period_tok = '.'
        # Recursion takes place only if the split has to be performed
        if len(self.tokenizer.encode(sentence, is_split_into_words=True)) > self.max_length:
            idx_half = len(sentence)//2
            # Dictionary with position associated to how far each period (if any) is from the middle of the sentence
            period_offsets = {pos: abs(idx_half - pos) for pos in range(len(sentence)) if sentence[pos] == period_tok}
            if period_offsets != {}:
                # If there is a period, sort based on the distance from the central point
                period_offsets_sorted = sorted(period_offsets.items(), key=lambda x: x[1])
                split_point = period_offsets_sorted[0][0]
            else:
                # If there is no period, take the middle index
                split_point = idx_half
            # Define the splits based on the found splitting point
            sent1, sent2 = sentence[:split_point+1], sentence[split_point+1:]
            lab1, lab2 = labels[:split_point+1], labels[split_point+1:]
            split1, split2 = self.split_sentences(sent1, lab1), self.split_sentences(sent2, lab2)  # Recursive call
            return split1[0]+split2[0], split1[1]+split2[1]
        else:
            return [sentence], [labels]

    def train_test_split(self):
        '''
        Splits the dataset into training and test observations
        :return: Training and test data and labels
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.split_tokens, self.split_labels, test_size=0.20,
                                                            random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def get_label_encoding_dict(self):
        '''
        Given the training data, associate each distinct label to an id
        :return: lab2id: a dictionary mapping unique labels to ids
        '''
        labels = []  # list of unique labels
        for sent in self.labels:
            for label in sent:
                if label not in labels and label != 'O':
                    labels.append(label)
        # Sort labels by the first letter after B- and I- in the BIO tag
        labels = ['O'] + sorted(labels, key=lambda x: x[2:])
        lab2id = {lab: id for lab, id in zip(labels, range(len(labels)))}
        return lab2id

    def get_bert_labels(self, tokenized_words, labels):
        '''
        Align labels with the pre-processed token sequences
        :return: A list of label sequences for sentences
        '''
        labels_bert = []
        for i, label in enumerate(labels):  # Loop over token sentences
            # Map each tokenized word to its ID in the original sentence
            word_ids = tokenized_words.word_ids(batch_index=i)
            # Contains the label ids for a sentence
            label_ids = []
            for word_idx in word_ids:
                # Special characters ([CLS], [SEP], [PAD]) set to -100
                if word_idx is None:
                    label_ids.append(-100)
                # If a word is broken by wordpiece, just add as many labels as word chunk
                else:
                    label_ids.append(self.label2id[label[word_idx]])
            labels_bert.append(label_ids)
        return labels_bert

    def create_data_loaders(self, bert_ds, labels):
        '''
        Create a dataset compatoble with torch
        :param bert_ds: A tokenized object containing both input_ids and mask ids
        :param labels: The label sequence asociated to the tokens
        :return: A torch DataLoader object
        '''
        # Create the DataLoader for our training set
        # So now only use the inputs, not the original data anymore
        data = TensorDataset(torch.tensor(bert_ds['input_ids']), torch.tensor(bert_ds['attention_mask']), labels)
        sampler = RandomSampler(data)
        # For each data loader we need the data, a sampler and a batch size
        data_loader = DataLoader(dataset = data, sampler = sampler, batch_size = self.batch_size)
        return data_loader


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

'''
Read the data from a pre-processed CADEC dataset and process them into a format compatible with BERT
'''


class DataProcessor():
    """
    Loads the data from a pre-processed CADEC named-entity dataset and creates a BERT dataset
    """
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
        self.label2id, self.id2label = self.get_label_encoding_dict()  # Initialize mapping of labels to ids
        # Test set: 0.2
        self.tokens_train, self.tokens_test, self.labels_train, self.labels_test = self.train_test_split(test_size=0.20)
        # Validation set: 0.125 of 0.8 = 0.1 of total
        self.tokens_train, self.tokens_val, self.labels_train, self.labels_val = self.train_test_split(test_size=0.125)

        print('Tokenize sentences...')
        # Tokenize for BERT
        # Training set
        self.tokenized_input_train = self.tokenizer(self.tokens_train, truncation=True, is_split_into_words=True,
                                                    add_special_tokens=True, padding=True)
        self.tokenized_input_train = self.add_word_ids(self.tokenized_input_train)
        self.train_tags = self.get_bert_labels(self.tokenized_input_train, self.labels_train)
        self.train_max_length = len(self.tokenized_input_train['input_ids'])  # The length of the longest training message

        # Validation set
        self.tokenized_input_val = self.tokenizer(self.tokens_val, truncation=True, is_split_into_words=True,
                                                  add_special_tokens=True, padding=True, max_length = self.train_max_length)
        self.tokenized_input_val = self.add_word_ids(self.tokenized_input_val)
        self.val_tags = self.get_bert_labels(self.tokenized_input_val, self.labels_val)

        # Test set
        self.tokenized_input_test = self.tokenizer(self.tokens_test, truncation=True, is_split_into_words=True,
                                                   add_special_tokens=True, padding=True, max_length = self.train_max_length)
        self.tokenized_input_test = self.add_word_ids(self.tokenized_input_test)
        self.test_tags = self.get_bert_labels(self.tokenized_input_test, self.labels_test)

        print('Preparing the dataset...')
        # Prepare the data so it is compatible with torch
        self.y_train = torch.tensor(self.train_tags).to(self.device)
        self.y_val = torch.tensor(self.val_tags).to(self.device)
        self.y_test = torch.tensor(self.test_tags).to(self.device)

        self.train_dataloader = self.create_data_loaders(self.tokenized_input_train, self.y_train)
        self.val_dataloader = self.create_data_loaders(self.tokenized_input_val, self.y_val)
        self.test_dataloader = self.create_data_loaders(self.tokenized_input_test, self.y_test)

    def sentence_parser(self):
        '''
        Read the content of filename and parses it into labels and tokens
        :return: tokens and labels: two lists containing the tokens and the labels in the dataset
        '''
        with open(self.filename, 'r') as f:
            data_raw = f.read()
        sentences = [sent.split('\n') for sent in data_raw.split('\n\n')[:-1]]  # Read the sentences
        tokens = [[pair.split('\t')[0] for pair in sent] for sent in sentences]  # Colect labels and tokens
        labels = [[pair.split('\t')[1] for pair in sent] for sent in sentences]
        labels = [[lab if lab not in ('I-Finding', 'B-Finding') else 'O' for lab in sent] for sent in labels]
        return tokens, labels

    def split_sentences(self, sentence, labels):
        '''
        Read the tokenized sentences and split them if they are longer than a maximum length (by default, 512)
        :param: An input tokenized sentence
        :param: The labels corresponding to the tokenized sentence
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
                # If there is a period, sort period locations based on the distance from the central point
                period_offsets_sorted = sorted(period_offsets.items(), key=lambda x: x[1])
                split_point = period_offsets_sorted[0][0]  # The period location closest to the centre of the sequence
            else:
                # If there is no period, take the middle index
                split_point = idx_half
            # Define the splits based on the found splitting point
            sent1, sent2 = sentence[:split_point+1], sentence[split_point+1:]
            lab1, lab2 = labels[:split_point+1], labels[split_point+1:]
            split1, split2 = self.split_sentences(sent1, lab1), self.split_sentences(sent2, lab2)  # Recursive call
            return split1[0]+split2[0], split1[1]+split2[1]  # Compose lists of sub-lists of split sentences
        else:
            return [sentence], [labels]

    def train_test_split(self, test_size):
        '''
        Splits the dataset into training and test observations
        :return: Training and test data and labels
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.split_tokens, self.split_labels, test_size=test_size,
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
        id2lab = labels
        return lab2id, id2lab

    def add_word_ids(self, tokenized_data):
        """
        Adds to the tokenized object the original word ids of the token to reconstruct from wordpiece
        :param tokenized_data: A dictionary object of tokenized data
        :return: The same tokenized data with the word ids for each sentence
        """
        word_ids = []
        for i in range(len(tokenized_data['input_ids'])):
            batch_word_id = tokenized_data.word_ids(batch_index=i)
            # Convert Nones to 0 and augment all IDs by 1 (used when we create tensors)
            batch_word_id = [i+1 if i!=None else 0 for i in batch_word_id]
            word_ids.append(batch_word_id)
        tokenized_data['word_ids'] = word_ids
        return tokenized_data

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
                    label_ids.append(self.label2id['O'])  # Assign the O label to the special characters
                # If a word is broken by wordpiece, just add as many labels as word chunk
                else:
                    label_ids.append(self.label2id[label[word_idx]])
            labels_bert.append(label_ids)
        return labels_bert

    def create_data_loaders(self, bert_ds, labels):
        '''
        Create a dataset compatible with torch
        :param bert_ds: A tokenized object containing both input_ids and mask ids
        :param labels: The label sequence associated to the tokens
        :return: A torch DataLoader object
        '''
        # Create the DataLoader for our training set
        # So now only use the inputs, not the original data anymore
        data = TensorDataset(torch.tensor(bert_ds['input_ids']), torch.tensor(bert_ds['attention_mask']), labels,
                             torch.tensor(bert_ds['word_ids']))
        sampler = RandomSampler(data)
        # For each data loader we need the data, a sampler and a batch size
        data_loader = DataLoader(dataset=data, sampler=sampler, batch_size=self.batch_size)
        return data_loader


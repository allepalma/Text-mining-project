import os
import logging
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

class TrainTest():
    def __init__(self,
                 model,
                 model_name,
                 dataloader,
                 optimizer=torch.optim.Adam,
                 learning_rate=2e-5,
                 early_stopping=True,
                 max_epochs=30,
                 logging=True
                 ):
        # Initialize logger to store results
        self.logging = logging
        if self.logging:
            self.logger = self.initialize_logger('training_logger.log')

        # Define model and dataloader
        self.model = model
        self.model_name = model_name
        self.dataloader = dataloader

        # Check for GPU
        if torch.cuda.is_available():    
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        # Initialize training parameters
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs

        # Train and test
        print(f'\nStart training for {self.model_name}...\n')
        if self.logging:
            self.logger.info(f'\nStart training for {self.model_name}...\n')
        self.train()
        print(f'\nTesting trained model {self.model_name}...\n')
        if self.logging:
            self.logger.info(f'\nTesting trained model {self.model_name}...\n')
        self.evaluate('Test')

    def initialize_logger(self, log_file):
        '''
        Initialize loggerfile
        '''
        log_format = '%(asctime)s : %(message)s'
        logging.basicConfig(format=log_format, filename=log_file, level=logging.DEBUG)
        return logging.getLogger()

    def log_metrics(self, avg_loss, f1, precision, recall, data):
        '''
        Log metrics during training/validating/testing
        '''
        self.logger.info(f'{data} avg. loss:\t{avg_loss}')
        self.logger.info(f'{data} F1-score:\t{f1}')
        self.logger.info(f'{data} precision:\t{precision}')
        self.logger.info(f'{data} recall:\t{recall}')

    def get_added_tokens(self):
        '''
        Return IDs of padding/separator/cls tokens
        '''
        padding = self.dataloader.tokenizer.vocab['[PAD]']
        separator = self.dataloader.tokenizer.vocab['[SEP]']
        classifier = self.dataloader.tokenizer.vocab['[CLS]']
        return padding, separator, classifier

    def get_metrics(self, y_true, y_pred):
        '''
        Return F1-score, precision and recall
        '''
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        return f1, precision, recall

    def train_epoch(self):
        '''
        Train model for a single epoch and return loss averaged over batches
        '''
        total_loss = 0
        # Train mode
        self.model.train()
        for batch in tqdm(self.dataloader.train_dataloader):
            # Get data
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            # Set gradient to zero
            self.model.zero_grad()
            # Get model output
            output = self.model(input_ids = input_ids, 
                              attention_mask = attention_mask,
                              labels = labels)
            loss = output['loss']
            total_loss += loss.item()
            # Compute gradient
            loss.backward()
            # Update weights
            self.optimizer.step()
        return total_loss/len(self.dataloader.train_dataloader)

    def train(self):
        '''
        Train model and stop on basis of early stopping criterion
        '''
        no_improve_cnt = 0
        prev_loss = 10000
        for i in range(self.max_epochs):
            print(f'\nEpoch {i+1}')
            # Train and get avg. training loss over batches
            avg_loss_tr = self.train_epoch()
            print(f'Train avg. loss:\t{avg_loss_tr}')
            if self.logging:
                self.logger.info(f'Epoch: {i+1}')
                self.logger.info(f'Train avg. loss:\t{avg_loss_tr}')
            # Validate model
            loss_val = self.evaluate('Validation')
            # Check and update early stopping
            if self.early_stopping:
                # Save model if it has improved enough over current epoch
                if loss_val < (prev_loss - 0.005):
                    no_improve_cnt = 0
                    prev_loss = loss_val
                    torch.save(self.model.state_dict(), os.path.join('saved_models', self.model_name))
                else:
                    no_improve_cnt += 1
                # Early stop with no (significant) improvement for 3 epochs
                if no_improve_cnt == 3:
                    print(f'\nEarly stopping: no improvement for {no_improve_cnt} epochs')
                    print(f'Model is saved at "saved_models/{self.model_name}"')
                    break
    
    def evaluate_batch(self, batch):
        '''
        Evaluate a single batch and return loss, 
        and predicted- and true labels of non-special tokens
        '''
        # Get data
        input_ids = batch[0].to(self.device)
        attention_mask = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        # Get model output
        with torch.no_grad():
            output = self.model(input_ids = input_ids, 
                                attention_mask = attention_mask,
                                labels = labels)
        loss = output['loss'].item()
        # Get the label IDs predicted by the model
        if 'predictions' in output.keys():
            predictions = output['predictions']
        else:
            logits = output['logits']
            predictions = torch.argmax(logits, dim=2)
        # Get rid of special tokens (pad/sep/cls)
        padding, separator, cls_token = self.get_added_tokens()
        mask = ((input_ids != padding) & (input_ids != separator) & (input_ids != cls_token))
        pred_list = torch.masked_select(predictions, mask).tolist()
        label_list = torch.masked_select(labels, mask).tolist()
        return loss, pred_list, label_list

    def evaluate(self, datatype):
        '''
        Test the model (on validation or test set),
        print metrics and return F1-score
        '''
        # Get either validation or test dataloader
        if datatype == 'Validation':
            data = self.dataloader.val_dataloader
        else:
            data = self.dataloader.test_dataloader
            self.model.load_state_dict(torch.load(os.path.join('saved_models', self.model_name)))
        total_loss = 0
        pred_list, y_list = [], []
        # Set model to evaluation mode
        self.model.eval()
        # Evaluate each batch individually
        for batch in data:
            loss, preds, labels = self.evaluate_batch(batch)
            total_loss += loss
            # Save predictions and true labels for metrics
            pred_list += preds
            y_list += labels
        # Calculate and print metrics
        avg_loss = total_loss/len(data)
        f1, precision, recall = self.get_metrics(y_list, pred_list)
        print(f'{datatype} avg. loss:\t{avg_loss}')
        print(f'{datatype} F1-score:\t{f1}')
        print(f'{datatype} precision:\t{precision}')
        print(f'{datatype} recall:\t{recall}')
        if datatype == 'Test':
            print(f'Classification report:\n{classification_report(y_list, pred_list, target_names=self.dataloader.id2label)}')
        if self.logging:
            self.log_metrics(avg_loss, f1, precision, recall, datatype)
        return avg_loss
        
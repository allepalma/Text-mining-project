from transformers import BertPreTrainedModel, AutoModel
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
import torch.nn.functional as f

"""
BERT model with linear classification head 
"""


class BertLinear(BertPreTrainedModel):
    """
    Module implementing a BERT classifier with a linear layer of output heads with different numbers of neurons
    The config object must contain:
    - num_labels: number of output labels
    - dropout: the dropout rate
    - hidden_size: the hidden_size of the bert model (768)
    - num_layers: number of hidden layers of the linear classifier
    - num_neurons: a list containing number of neurons for as many layers as indicated by num_layers
    - activation: activation function of the linear layers
    """
    def __init__(self, config):
        super().__init__(config)
        # The number of exit labels for each element of the sequence
        self.num_labels = config.num_labels
        #Initialize dropout layer
        self.dropout = nn.Dropout(p=config.dropout)
        # The BERT pre-trained model
        self.bert = AutoModel.from_pretrained(config.model)
        self.hidden_size = config.hidden_size  # The hidden layer size of the BERT model (fixed at 768)
        # Linear head
        self.num_clf_hidden_layers = config.num_clf_hidden_layers  # number of layers
        self.num_neurons = config.num_neurons
        self.activation = config.activation
        # Implement the sequential network
        self.classifier = nn.Sequential()
        if self.num_clf_hidden_layers > 0:
            self.classifier.add_module(f'linear 0', nn.Linear(self.hidden_size, self.num_neurons[0]))
            self.classifier.add_module(f'Activation 0', self.activation())
            for i in range(1, self.num_clf_hidden_layers):
                self.classifier.add_module(f'linear {i}', nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
                self.classifier.add_module(f'Activation {i}', self.activation())
            # Add output layer
            self.classifier.add_module('Output layer', nn.Linear(self.num_neurons[-1], self.num_labels))
        else:
            self.classifier.add_module('Output layer', nn.Linear(self.hidden_size, self.num_labels))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True
    ):
        # Apply BERT to the input labels
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        sequence_output = outputs[0]  # Select last BERT embedding

        # Dropout
        sequence_output = self.dropout(sequence_output)
        # Apply classification to layers to output
        logits = self.classifier(sequence_output)
        loss = None

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = attention_mask.view(-1) == 1  # flattened boolean vector
        active_logits = logits.view(-1, self.num_labels)  # flattened logits vector
        # Apply the mask to specific labels whenever their attention is equal to 0
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        # Compute loss function
        loss = loss_fct(active_logits, active_labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions}


"""
BERT model with CRF classification head 
"""


class BertCRF(BertPreTrainedModel):
    """
    Module implementing a BERT classifier with a CRF layer of output heads with different numbers of neurons
    The config object must contain:
    - num_labels: number of output labels
    - dropout: the dropout rate
    - hidden_size: the hidden_size of the bert model (768)
    """
    def __init__(self, config):
        super().__init__(config)
        # The number of exit labels for each element of the sequence
        self.num_labels = config.num_labels
        #Initialize dropout layer
        self.dropout = nn.Dropout(p=config.dropout)
        # The BERT pre-trained model
        self.bert = AutoModel.from_pretrained(config.model)
        self.hidden_size = config.hidden_size  # The hidden layer size of the BERT model
        # Linear layer to map BERT output to label size
        self.linear = nn.Linear(self.hidden_size, self.num_labels)
        # CRF head
        self.crf = CRF(self.num_labels, batch_first=True)
        # Initialize log-softmax for prediction
        self.log_soft = f.log_softmax

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True
    ):
        # Apply BERT to the input labels
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        sequence_output = outputs[0]  # Select last BERT embedding

        # Dropout and linear layer prediction
        bert_output = self.dropout(sequence_output)
        probs = self.linear(bert_output)

        # Transform the results to probabilities
        logits = self.log_soft(probs, 2)
        attention_mask = attention_mask.type(torch.uint8)

        # Compute the CRF loss as the negative log-likelihood (which is the output of forward)
        loss = -self.crf(logits, labels, mask=attention_mask, reduction='token_mean')

        # Predict the best sequence
        pred_list = self.crf.decode(logits, mask=attention_mask)
        preds = torch.zeros_like(input_ids).long()
        for i, pred in enumerate(pred_list):
            preds[i, :len(pred)] = torch.LongTensor(pred)

        return {
            'loss': loss,
            'predictions': preds,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions}


"""
BERT model with LSTM classification head 
"""


class BertLSTM(BertPreTrainedModel):
    """
    Module implementing a BERT classifier with a LSTM layer of output heads with different numbers of neurons
    The config object must contain:
    - num_labels: number of output labels
    - dropout: the dropout rate
    - hidden_size: the hidden_size of the bert model (768)
    """
    def __init__(self, config):
        super().__init__(config)
        # The number of exit labels for each element of the sequence
        self.num_labels = config.num_labels
        #Initialize dropout layer
        self.dropout = nn.Dropout(p=config.dropout)
        # The BERT pre-trained model
        self.bert = AutoModel.from_pretrained(config.model)
        self.hidden_size = config.hidden_size  # The hidden layer size of the BERT model
        # Linear layer to map BERT output to label size
        self.linear = nn.Linear(self.hidden_size, self.num_labels)
        # ultimo embedding di LSTM
        self.final_lstm = nn.LSTM(self.num_labels,
                                  self.num_labels,
                                  2,
                                  dropout=config.dropout,
                                  batch_first=True)
        # Initialize log-softmax for prediction
        self.log_soft = f.log_softmax

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        # Apply BERT to the input labels
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=output_hidden_states)
        sequence_output = outputs[0]  # Select last BERT embedding

        # Dropout and linear layer prediction
        bert_output = self.dropout(sequence_output)
        scores = self.linear(bert_output)
        attention_mask = attention_mask.type(torch.uint8)

        # Apply LSTM to output of the linear layer
        lstm_out, _ = self.final_lstm(scores)
        lstm_out = lstm_out.contiguous()

        # Compute loss
        loss = None
        loss_fct = CrossEntropyLoss()

        # Only keep active parts of the loss
        active_loss = attention_mask.view(-1) == 1  # flattened boolean vector
        active_logits = lstm_out.view(-1, self.num_labels)  # flattened logits vector
        # Apply the mask to specific labels whenever their attention is equal to 0
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        # Compute loss function
        loss = loss_fct(active_logits, active_labels)

        return {
            'loss': loss,
            'logits': lstm_out,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions}


'''
Baseline LSTM-crf network 
'''


class Baseline(nn.Module):
    """
    Module implementing a baseline model with an LSTM followed by a CRF layer
    The config object must contain:
    - embedding_size: the dimensionality of the embedded words
    - voocab_size: the vocabulary size of BERT (hint: leave as default)
    - num_labels: number of output labels
    - hidden_size: the hidden_size of the bert model (768)
    """
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.embedding_size = config.embedding_size  # The size of the initial word embeddings
        self.vocab_size = config.vocab_size  # The number of words present in the BERT vocabulary
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size  # The hidden size of the LSTM

        # Embedding layer to compute a representation of the sequence of words
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # The LSTM layer
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            num_layers=2,
                            dropout=config.dropout,
                            batch_first=True)
        # A linear layer that maps the sequence to the possible number of labels
        self.hidden2tag = nn.Linear(self.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # Log-softmax function (between the LSTM and the CRF)
        self.log_soft = f.log_softmax

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        # Fix type of the attention mask
        attention_mask = attention_mask.type(torch.uint8)
        # Apply embedding layer to the input ids to the input labels
        embedded_input = self.embedding(input_ids)
        # Apply the lstm to the result of the embedding
        lstm_out, _ = self.lstm(embedded_input)
        # Apply Linear layer to map to the label size
        linear_out = self.hidden2tag(lstm_out)
        logits = self.log_soft(linear_out, 2)
        # Compute the CRF loss as the negative log-likelihood (which is the output of forward)
        loss = -self.crf(logits, labels, mask=attention_mask, reduction='token_mean')

        # Predict the best sequence
        pred_list = self.crf.decode(logits, mask=attention_mask)
        preds = torch.zeros_like(input_ids).long()
        for i, pred in enumerate(pred_list):
            preds[i, :len(pred)] = torch.LongTensor(pred)

        return {
            'loss': loss,
            'predictions': preds}
    

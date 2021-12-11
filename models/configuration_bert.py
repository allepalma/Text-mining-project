from transformers import BertPreTrainedModel, AutoModel
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class CustomBertConfig(PretrainedConfig):
    """
    Configuration class to store the model parameters
    - model: Type of BERT model
    - clf_type: linear, lstm, crf
    - num_labels:  Number of classification labels per item
    - dropout:  Dropout between BERT and classifier
    - hidden_size:  Different from None only if model is linear
    - num_layers:  Number of hidden layers if linear model
    - num_neurons:  If linear, contains the number of neurons
    - activation: activation function classifier
    """
    def __init__(self,
                 model='bert-base-uncased',
                 clf_type='linear',
                 num_labels=9,
                 dropout=0.1,
                 hidden_size=768,
                 num_clf_hidden_layers=0,
                 num_neurons=(),
                 activation=nn.ReLU,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.model = model
        self.clf_type = clf_type
        self.num_labels = num_labels
        self.dropout = dropout
        self.hidden_size = hidden_size
        if self.clf_type == 'linear':
            self.num_clf_hidden_layers = num_clf_hidden_layers
            self.activation = activation
            self.num_neurons = num_neurons






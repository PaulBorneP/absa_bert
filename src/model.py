
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, bert_name: str, hidden_size: int = 768, num_classes: int = 3, dr_rate: float = None):
        """
        Initialize the BertClassifier model.

        Args:
            bert_name (str): The name of the pre-trained BERT model.
            hidden_size (int): The size of the hidden layer in the classifier. Default is 768.
            num_classes (int): The number of classes for classification. Default is 3.
            dr_rate (float): The dropout rate for the dropout layer. Default is None.
        """
        super(BertClassifier, self).__init__()
        self.bert = BertModel(BertConfig.from_pretrained(bert_name))
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        # freeze the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False



    def forward(self, x, mask):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        pooler = self.bert(x, attention_mask=mask)[1]        
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)